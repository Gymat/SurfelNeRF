import json
import os
import pickle
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
# import matplotlib.pyplot as plt
import pdb
import argparse
import copy
from tqdm import tqdm

# Global parameters
# IMAGE_

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_folder",
        default="/x/",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--data_path",
        default="/x/",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--output_dict",
        default="configs/scene0000_01.yaml",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--output_dir",
        default="configs/scene0000_01.yaml",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--depth_trunc",
        default=15,
        type=float,
        help="path to config file")
    parser.add_argument(
        "--height",
        default=480,
        type=int,
        help="path to config file")
    parser.add_argument(
        "--width",
        default=640,
        type=int,
        help="path to config file")
    parser.add_argument(
        "--begin_id",
        default=0,
        type=int,
        help="path to config file")
    parser.add_argument(
        "--end_id",
        default=10,
        type=int,
        help="path to config file")
    parser.add_argument(
        "--step",
        default=10,
        type=int,
        help="path to config file")
    return parser


def get_extrinsic(pos, lookat, u):
    """
    pos is the position;
    lookat is
    """
    # Here, we follow the pangolin code
    m = np.zeros((4,4))
    # 1
    L = lookat - pos
    # 2
    L_norm = norm_np(L)
#     print(L_norm)
    # We follow the code in Pangolin, insteasd of the website;
    # So, we first cross product the vecs ,and then normalize them
    # 3
    s = np.cross(L_norm, u)
    # 4
    y = np.cross(L_norm, s)
    # 5
    s_norm = norm_np(s)
    y_norm = norm_np(y)
    # R
    R = np.concatenate([s_norm[np.newaxis,:], y_norm[np.newaxis,:], 1*L_norm[np.newaxis,:]], axis=0)
    #
    t = -1 * np.matmul(R, pos)
    #
    m[:3, :3] = R
    m[:3, 3] = t
    m[3,3] = 1
    return m


def norm_np(v):
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v


class Create_from_depth_map():
    # This class is implemented via np package, so this can only handle batch_size=1.
    # If want to handle multi-batch_size, we need to implement this via torch package.
    def __init__(self, intrinsic_matrix, height=960, width=1280, depth_trunc=12.0):
        '''

        :param intrinsic_matrix: np.array size of (3,3)
        :param height:
        :param width:
        '''
        self.intrinsic_matrix = intrinsic_matrix
        print(intrinsic_matrix)
        self.focal_length = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1])
        self.principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.height = height
        self.width = width
        self.depth_trunc = depth_trunc
        self.cam_pos_cam_coord = np.asarray([0, 0, 0])
        # init useful params
        # We first build the coordinate matrix, which size is (H, W, 2), the value is the (x,y) coordinate of corresponding index
        i_coords, j_coords = np.meshgrid(range(height), range(width), indexing='ij')
        coord_mat = np.concatenate((j_coords[..., None], i_coords[..., None]), axis=-1)
        #
        principal_point_mat = np.array([self.principal_point[0], self.principal_point[1]])
        principal_point_mat = principal_point_mat.reshape(1, 1, 2)
        focal_length_mat = np.asarray([self.focal_length[0], self.focal_length[1]])
        focal_length_mat = focal_length_mat.reshape(1, 1, 2)
        self.one_mat = np.ones((height, width, 1))  # This one_mat is to build the 4_d vector (x, y, z, 1) which represents a point in the camera coordinate
        #
        self.cam_coord_part = (coord_mat - principal_point_mat) / focal_length_mat


    def project(self, depth_map, extrinsic_c2w_matrix):
        '''
        This for one image once.
        :param depth_map: np.array (H, W)
        :param extrinsic_c2w_matrix: np.array (4,4)
        :return:
        '''
        z_mat = depth_map.reshape(self.height, self.width, 1)
        masked_depth_map = np.zeros_like(z_mat)
        masked_depth_map[(z_mat < self.depth_trunc) & (z_mat > 0)] = 1  # true is 1!
        # pdb.set_trace()
        wrong_depth = 1 - masked_depth_map  # wrong is 1, true is 0
        if np.sum(wrong_depth) > 0:
            print("NUMBER {} Points are needed to be filtered since the depth value is too large or zero!".format(
                  np.sum(wrong_depth)))
        # pdb.set_trace()
        point_mask = masked_depth_map  # h, w, 1

        # do the computation
        cam_coord_xy = self.cam_coord_part * z_mat
        cam_coord_xyz = np.concatenate([cam_coord_xy, z_mat], axis=-1)  # (h, w, 3)
        cam_coord_xyz1 = np.concatenate([cam_coord_xyz, self.one_mat], axis=-1)  # (h, w, 4)
        # obtain the world_coordinate
        xyz1_cam_coord = np.transpose(cam_coord_xyz1, (2, 0, 1))  # (4, h, w)
        xyz1_cam_coord = xyz1_cam_coord.reshape(4, self.height * self.width)  # (4, h*w)
        # print(xyz1_cam_coord.shape, c2w_mat.shape)
        world_coord_mat = np.matmul(extrinsic_c2w_matrix, xyz1_cam_coord)  # (4, h*w)
        # print(world_coord_mat.shape)
        world_coord_mat = world_coord_mat[:3, :]  # (3, h*w)
        world_coord_mat = world_coord_mat.T  # (h*w, 3)
        points_dir_world_coord = self.obtain_point_dir(cam_coord_xyz, extrinsic_c2w_matrix)  # (h*w, 3)
        # reshape the size of point_mask from (h, w, 1) to (h*w, 1)
        point_mask = point_mask.reshape(self.height*self.width, 1)
        # print(world_coord_mat.shape)
        return world_coord_mat, points_dir_world_coord, point_mask

    def obtain_point_dir(self, cam_coord_xyz, extrinsic_c2w_m):
        h, w, _ = cam_coord_xyz.shape
        points_dir = cam_coord_xyz  # (h, w, 3)
        points_dir = points_dir.reshape(h*w, 3)
        points_dir_normed = norm_np(points_dir)
        points_dir_normed = points_dir_normed.T  # (3, h*w)
        # change the dir under world coordinates
        cam2world_mat3x3 = extrinsic_c2w_m[:3, :3]
        points_dir_world_coor = np.matmul(cam2world_mat3x3, points_dir_normed)  # (3, h*w)
        # points_dir_world_coor = points_dir_world_coor.reshape((3, h, w))
        # points_dir_world_coor = np.transpose(points_dir_world_coor, (1, 2, 0))  # ()
        return points_dir_world_coor.T  # (h*w, 3)

    def read_depth(self, filepath):
        depth_im = cv2.imread(filepath, -1).astype(np.float32)
        depth_im /= 1000
        # depth_im[depth_im > 8.0] = 0
        # depth_im[depth_im < 0.3] = 0
        return depth_im


def obtain_coord_map(h, w):
    H_array = np.arange(h)
    W_array = np.arange(w)
    H_array = H_array[..., None]
    W_array = W_array[None,]
    w_array_re = np.repeat(W_array, h, axis=0)
    h_array_re = np.repeat(H_array, w, axis=1)
    coord_map = np.concatenate((h_array_re[..., None], w_array_re[..., None]), axis=-1)  # H x W x 2
    return coord_map

def obtain_dis_camcenter(coord_map, cam_center):
    """
    obtain the weights for each point
    params: coord_map np.array: shape (H, W, 2)
    params: cam_center: np.array: shape(2)
    """
    dis = coord_map - cam_center
    dis_s2 = dis * dis
    dis_s2_sum = np.sum(dis_s2, axis=-1)
    dis_s2_sqrt = np.sqrt(dis_s2_sum)
    normed_dis = dis_s2_sqrt / np.max(dis_s2_sqrt)
    twosigmasquared = 0.72
    weights = np.exp(-1 * (normed_dis * normed_dis) / twosigmasquared)
    return weights


def obtain_weights_pixel(h, w, cam_center):
    coord_map = obtain_coord_map(h, w)
    weights = obtain_dis_camcenter(coord_map, cam_center)
    return weights

def read_pose_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    c2w_list = []
    for line in lines:
        words = line.strip().split(" ")
        row = [float(x) for x in words]
        c2w_list.append(row)
    c2w = np.array(c2w_list)
    return c2w


def load_intrinsic_depth(data_path, scene_name):
    txt_path = os.path.join(data_path, scene_name, 'intrinsic', 'intrinsic_depth.txt')
    intrinsic_matrix = read_pose_file(txt_path)
    return intrinsic_matrix[:3, :3]


def load_c2w_mat(data_path, img_id, scnen_name):
    txt_path = os.path.join(data_path, scnen_name, 'pose',"{}.txt".format(img_id))
    c2w_mat = read_pose_file(txt_path)
    w2c_mat = np.linalg.inv(c2w_mat)
    return c2w_mat, w2c_mat


def read_img_path(data_path, scene_name, img_id, height, width):
    img_path = os.path.join(data_path, scene_name, 'color', "{}.jpg".format(img_id))
    # pdb.set_trace()
    img = Image.open(img_path)
    img = img.resize((width, height), )
    img = np.array(img) / 255.0  #  H x W x 3

    return img


def main(args):
    print("!!! Make sure to check the direction of <up> !!!")

    # First read the json file
    # Opening JSON file
    json_list = os.listdir(args.json_folder)
    json_list.sort()
    print(json_list)
    # setup
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #
    for i in tqdm(range(args.begin_id, args.end_id)):
        json_item_name = json_list[i]
        scene_name = json_item_name.split('.')[0]
        scene_output_path = os.path.join(args.output_dir, scene_name)
        if not os.path.exists(scene_output_path):
            os.mkdir(scene_output_path)
        json_path = os.path.join(args.json_folder, json_item_name)
        with open(json_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        output_data = copy.deepcopy(json_object)
        # output_data['up'] *= -1  # !!! For ScanNet!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        width = args.width
        #
        intrinsic_matrix = load_intrinsic_depth(args.data_path, scene_name)
        depth_to_pcd = Create_from_depth_map(intrinsic_matrix, height=args.height, width=args.width,
                                             depth_trunc=args.depth_trunc)
        camcenter = np.array([intrinsic_matrix[0, 2], intrinsic_matrix[1, 2], ])
        for data_iter in tqdm(enumerate(json_object.keys()), ):
            index, imgid = data_iter
            assert index == int(imgid)
            if index % args.step != 0:
                # filter the non key frame
                continue
            item_data = json_object[imgid]
            # img_path = os.path.join(args.data_path, 'color', item_data['rgb_img_name'])
            depth_path = os.path.join(args.data_path, scene_name, 'depth', item_data['depth_img_name'])
            #
            depth = depth_to_pcd.read_depth(depth_path)
            # pdb.set_trace()
            lookat_item = item_data['look_at']
            campos = item_data['campos']
            # up = item_data['up'] * -1  # Note !!!
            #
            c2w_mat, w2c_mat = load_c2w_mat(args.data_path, scnen_name=scene_name, img_id=imgid)
            pcd_w, pcd_dir, point_mask = depth_to_pcd.project(depth, c2w_mat)
            pcd_item = o3d.geometry.PointCloud()
            pcd_item.points = o3d.utility.Vector3dVector(pcd_w)
            # estimate the normals via open3d and orient normals towards camera location.
            pcd_item.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                  max_nn=30))
            # check norm dir
            C = np.array(campos).reshape(3)
            pcd_item.orient_normals_towards_camera_location(C)
            # Obtain weights
            weights_item = obtain_weights_pixel(args.height, args.width, camcenter)
            rgb_image = read_img_path(args.data_path, img_id=imgid, height=args.height, width=args.width,
                                      scene_name=scene_name)
            # new dict for update
            item_dict = {
                'points_pos': pcd_w,
                'normals': np.asarray(pcd_item.normals),
                'colors': rgb_image.reshape(args.height * args.width, 3),
                'point_mask': point_mask,
                'depth': depth,
                'weights': weights_item,
                'pcd_dirs': pcd_w - C[None, :]
            }

            ori_dict = output_data[imgid]
            # pdb.set_trace()
            ori_dict['up'] = [x*-1 for x in ori_dict['up']]
            ori_dict.update(item_dict)
            #
            # pdb.set_trace()
            item_saved_path = os.path.join(scene_output_path, "{}.pkl".format(str(imgid).zfill(4)))
            # Serializing json
            with open(item_saved_path, 'wb') as f:
                pickle.dump(ori_dict, f)
    #
    # Writing to sample.json
    # with open(args.output_dict, "w") as outfile:
    #     outfile.write(output_data)

if __name__ == '__main__':
    print("For ScanNet only!")
    args = setup_argparser().parse_args()
    print(args)
    main(args)
