import sys
sys.path.append('.')
from tqdm import tqdm
import time
import pickle
import argparse
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import copy

import pdb


"""
The target of this file is to generate the data, containing:
    (1) fragments info: a. key frame id; b. fragment id
    (2) world coordinate of each pixel in each image, view as size (H*W, 3), 

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Fuse ground truth tsdf')
    parser.add_argument("--dataset", default='scannet')
    parser.add_argument("--data_path", metavar="DIR",
                        help="path to raw dataset", default='/data/scannet/output/')
    parser.add_argument("--save_path", metavar="DIR",
                        help="file name", default='all_tsdf')

    #
    parser.add_argument('--sample_rate', default=0.05, type=float, help="rate to uniform sample the key frame from "
                                                                        "the sequence")
    parser.add_argument('--key_frame_num', default=3, type=int)

    # parser.add_argument('--test', action='store_true',
    #                     help='prepare the test set')
    # parser.add_argument('--max_depth', default=3., type=float,
    #                     help='mask out large depth values since they are noisy')
    # parser.add_argument('--num_layers', default=3, type=int)
    # parser.add_argument('--margin', default=3, type=int)
    parser.add_argument('--voxel_size', default=0.005, type=float)

    # parser.add_argument('--window_size', default=9, type=int)
    # parser.add_argument('--min_angle', default=15, type=float)
    # parser.add_argument('--min_distance', default=0.1, type=float)

    # ray multi processes
    # parser.add_argument('--n_proc', type=int, default=16, help='#processes launched to process scenes.')
    # parser.add_argument('--n_gpu', type=int, default=2, help='#number of gpus')
    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--loader_num_workers', type=int, default=8)
    return parser.parse_args()


class Create_from_depth_map():
    # This class is implemented via np package, so this can only handle batch_size=1.
    # If want to handle multi-batch_size, we need to implement this via torch package.
    def __init__(self, intrinsic_matrix, height=960, width=1280, depth_trunc=10.0):
        '''

        :param intrinsic_matrix: np.array size of (3,3)
        :param height:
        :param width:
        '''
        self.intrinsic_matrix = intrinsic_matrix
        self.focal_length = (intrinsic_matrix[0, 0], intrinsic_matrix[1, 1])
        self.principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.height = height
        self.width = width
        self.depth_trunc = depth_trunc
        self.cam_pos_cam_coord = np.asarray([0,0,0])
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
        self.cam_coord_part = (coord_mat - principal_point_mat)/focal_length_mat


    def project(self, depth_map, extrinsic_c2w_matrix):
        '''
        This for one image once.
        :param depth_map: np.array (H, W)
        :param extrinsic_c2w_matrix: np.array (4,4)
        :return:
        '''
        z_mat = depth_map.reshape(self.height, self.width, 1)
        masked_depth_map = np.zeros_like(z_mat)
        masked_depth_map[(z_mat < self.depth_trunc) & (z_mat > 0)] = 1
        wrong_depth = 1 - masked_depth_map
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


def main(args):
    #
    data_path = args.data_path
    #
    traj_txt = os.path.join(data_path, "traj.txt")
    traj_mat = read_traj(traj_txt)
    #
    img_root = os.path.join(data_path, "results")
    rgb_list, depth_list = prepare_data(img_root)
    # sample the key frame
    total_num = len(rgb_list)
    key_frame_step = int(total_num) * args.sample_rate
    local_frag_list = obtain_key_frame_list(total_num, key_frame_step, args.key_frame_num)
    #
    fragments = build_fragment(local_frag_list, ext_mat_list=traj_mat, rgb_list=rgb_list, dep_list=depth_list,
                               args=args)
    depth_map = obtain_depth_map_list(rgb_list, depth_list)
    fragments['depth_map'] = depth_map
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(os.path.join(args.save_path, "frags.pkl"), 'wb') as f:
        pickle.dump(fragments, f)

    pass

def obtain_depth_map_list(rgb_list, depth_list):
    output = dict()
    # fragments = []
    # intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)  # Hard code here!
    # intrinsic_matrix_np = intrinsic_matrix.intrinsic_matrix
    #
    # Pcd_from_dep = Create_from_depth_map(intrinsic_matrix_np, height=height, width=width)
    # saved_id_dict = dict()
    for index in tqdm((range(len(rgb_list)))):
        rgb_name = rgb_list[index]
        dep_name = depth_list[index]
        # build the rgbd image
        rgb_path = os.path.join(args.data_path, "results", rgb_name)
        dep_path = os.path.join(args.data_path, "results", dep_name)
        #
        color_raw = o3d.io.read_image(rgb_path)
        depth_raw = o3d.io.read_image(dep_path)
        #
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                        depth_scale=6553.5 * 10,
                                                                        convert_rgb_to_intensity=False, )
        depth_value = np.asarray(rgbd_image.depth)
        output[index] = depth_value

    return output



def obtain_key_frame_list(total_num, key_frame_step, num_key_frame):
    local_frag_list = []
    frame_id_list = [x for x in range(0, int(total_num), int(key_frame_step))]
    temp_list = []
    for f_id in frame_id_list:
        temp_list.append(f_id)
        if len(temp_list) == num_key_frame:
            local_frag_list.append(temp_list)
            temp_list = [f_id]
    # for the remaining frames
    if len(temp_list) > 0 and temp_list[-1] < total_num - 1:
        temp_list.append(total_num - 1)
        local_frag_list.append(temp_list)
    print(local_frag_list)
    return local_frag_list


def line2matrix(line):
    mc2w = np.zeros((4, 4))
    line = line.split(" ")
    for i, num in enumerate(line):
        dim0 = i // 4
        dim1 = i % 4
        mc2w[dim0, dim1] = float(num)
    #     mc2w[:3, 1] *= -1
    #     mc2w[:3, 2] *= -1
    mc2w[:3, 3] *= 0.1  # re-scale the position in the Replica dataset only!
    return mc2w


def read_traj(txt):
    output = []
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            mat = line2matrix(line)
            output.append(mat)
    return output


def prepare_data(image_root):
    rgb_list = []
    depth_list = []
    for img_name in os.listdir(image_root):
        if img_name.startswith("depth"):
            depth_list.append(img_name)
        else:
            rgb_list.append(img_name)
    # sort by id
    depth_list.sort(key=lambda x: int(x[5:11]))
    rgb_list.sort(key=lambda x: int(x[5:11]))
    print(rgb_list[:10])
    return rgb_list, depth_list

def build_fragment(local_frag_list, ext_mat_list, rgb_list, dep_list, width=600, height=340, args=None):
    output = dict()
    fragments = []
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(600, 340, 300, 300, 299.5, 169.5)  # Hard code here!
    intrinsic_matrix_np = intrinsic_matrix.intrinsic_matrix
    #
    Pcd_from_dep = Create_from_depth_map(intrinsic_matrix_np, height=height, width=width)
    # saved_id_dict = dict()
    saved_id_pcd = dict()
    saved_id_point_dir = dict()
    saved_id_color = dict()
    point_mask_dict = dict()
    for index, lfg in tqdm(enumerate(local_frag_list)):
        # integration
        # volume = o3d.integration.ScalableTSDFVolume(
        #     voxel_length=args.voxel_size,
        #     sdf_trunc=0.005,
        #     color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        for fid in lfg:
            rgb_name = rgb_list[fid]
            dep_name = dep_list[fid]
            # build the rgbd image
            rgb_path = os.path.join(args.data_path, "results", rgb_name)
            dep_path = os.path.join(args.data_path, "results", dep_name)
            #
            color_raw = o3d.io.read_image(rgb_path)
            depth_raw = o3d.io.read_image(dep_path)
            #
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                            depth_scale=6553.5 * 10,
                                                                            convert_rgb_to_intensity=False, )
            depth_value = np.asarray(rgbd_image.depth)
            color_value = np.asarray(rgbd_image.color) / 255.0
            c2w_m = ext_mat_list[int(fid)]
            pcd, pcd_dir, point_mask = Pcd_from_dep.project(depth_value, c2w_m)
            """
            pcd: np.array, size of (h*w, 3), where 3 is the position xyz coordinates
            pcd_dir: np.array, size of (h*w, 3), where 3 is the position xyz coordinates in the world coordinate
            """
            saved_id_pcd[fid] = pcd
            saved_id_point_dir[fid] = pcd_dir
            point_mask_dict[fid] = point_mask
            saved_id_color[fid] = color_value.reshape((width*height, 3))

        dict_ = {
            "frag_id": index,
            "key_frame_id": lfg,
            "voxel_size": args.voxel_size,
        }
        # pdb.set_trace()
        fragments.append(dict_)
    #
    output['fragments'] = fragments
    output['rgb_list'] = rgb_list
    output['depth_list'] = dep_list
    output['point_position'] = saved_id_pcd
    output['point_dir'] = saved_id_point_dir
    output['point_color'] = saved_id_color
    output['point_mask'] = point_mask_dict
    return output

# -------------------------------------------------------------------------
# Util Func
# -------------------------------------------------------------------------
def norm_np(v, theta=1e-10):
    if np.sum(v**2) == 0:
        # For safe operation
        normalized_v = v / theta
    else:
        length = (np.sqrt(np.sum(v ** 2, axis=-1)))
        if len(v.shape) == 2:
            length = length[:, None]
        normalized_v = v / (length + theta)
    return normalized_v



if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Finish!")
