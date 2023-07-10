import os
import pdb
import sys
sys.path.append(".")

import numpy as np
from numpy import dot
from math import sqrt
import matplotlib.pyplot as plt
import torch
import pickle
import time

from utils import format as fmt
import PointExt

from models.diff_rendering.diff_ray_marching import near_far_linear_ray_generation
from utils.simple_func import strlist_to_floatlist
from config.setup import setup_config
# X = torch.cuda.FloatTensor(8)




class lighting_fast_querier():

    def __init__(self, device, opt):

        print("querier device", device, device.index)
        self.device = device
        self.opt = opt
        self.claim_occ = PointExt.torch_launch_claim_occ
        self.map_coor2occ = PointExt.torch_launch_map_coor2occ
        self.fill_occ2pnts = PointExt.torch_launch_fill_occ2pnts
        self.mask_raypos = PointExt.torch_launch_mask_raypos
        self.get_shadingloc = PointExt.torch_launch_get_shadingloc
        self.query_along_ray = PointExt.torch_launch_query_along_ray

        self.count = 0
        self.vox_scale = strlist_to_floatlist(opt.MODEL.NEURAL_POINTS.VOX_SCALE)
        self.kernel_size = strlist_to_floatlist(opt.MODEL.NEURAL_POINTS.KERNEL_SIZE)
        self.ranges = strlist_to_floatlist(opt.DATA.RANGES)
        self.vox_size = strlist_to_floatlist(opt.MODEL.NEURAL_POINTS.VOX_SIZE)


    def get_hyperparameters(self, point_xyz_w_tensor, ranges=None):
        '''
        params vsize_np: 0.005
        '''
        # TODO: double check the hyperparameters !
        vsize_np = np.array(self.vox_size)
        self.vsize_np = vsize_np

        min_xyz, max_xyz = torch.min(point_xyz_w_tensor, dim=1)[0][0], torch.max(point_xyz_w_tensor, dim=1)[0][0]
        vscale_np = np.array(self.vox_scale, dtype=np.int32)  # 2 2 2
        scaled_vsize_np = (vsize_np * vscale_np).astype(np.float32)  # what is this ???
        # pdb.set_trace()
        if ranges is not None:
            # print("min_xyz", min_xyz.shape)
            # print("max_xyz", max_xyz.shape)
            # print("ranges", ranges)
            # pdb.set_trace()
            min_xyz, max_xyz = torch.max(
                torch.stack([min_xyz, torch.as_tensor(ranges[:3], dtype=torch.float32, device=min_xyz.device)], dim=0),
                dim=0)[0], torch.min(
                torch.stack([max_xyz, torch.as_tensor(ranges[3:], dtype=torch.float32, device=min_xyz.device)], dim=0),
                dim=0)[0]
        min_xyz = min_xyz - torch.as_tensor(scaled_vsize_np * self.kernel_size / 2, device=min_xyz.device,
                                            dtype=torch.float32)
        max_xyz = max_xyz + torch.as_tensor(scaled_vsize_np * self.kernel_size / 2, device=min_xyz.device,
                                            dtype=torch.float32)

        ranges_np = torch.cat([min_xyz, max_xyz], dim=-1).cpu().numpy().astype(np.float32)
        # print("ranges_np",ranges_np)
        vdim_np = (max_xyz - min_xyz).cpu().numpy() / vsize_np

        scaled_vdim_np = np.ceil(vdim_np / vscale_np).astype(np.int32)

        # print("grid voxel dim num: ", scaled_vdim_np)
        # pdb.set_trace()
        # To tensor
        ranges_tensor = torch.from_numpy(ranges_np).to(self.device)
        scaled_vox_size_tensor = torch.from_numpy(scaled_vsize_np).to(self.device)
        scaled_vox_grid_num_tensor = torch.from_numpy(scaled_vdim_np).to(self.device)
        kernel_size_tensor = torch.from_numpy(np.asarray(self.kernel_size, dtype=np.int32)).to(torch.int32).to(self.device)

        radius_limit_np = self.opt.MODEL.NEURAL_POINTS.RADIUS_LIMIT_SCALE * max(vsize_np[0], vsize_np[1])

        return np.asarray(radius_limit_np).astype(np.float32), ranges_np, vsize_np, scaled_vsize_np, \
               scaled_vdim_np, vscale_np, ranges_tensor, \
               scaled_vox_size_tensor, scaled_vox_grid_num_tensor, kernel_size_tensor

    def query_points(self, point_xyz_w_tensor, actual_numpoints_tensor, near_depth, far_depth,
                     ray_dirs_tensor, cam_pos_tensor, cam_rot_tensor, **kwargs):
        # For the right dim for campos and camdir
        # cam_pos_tensor = cam_pos_tensor.squeeze(0)
        # ray_dirs_tensor = ray_dirs_tensor.squeeze(0)
        # cam_rot_tensor = cam_rot_tensor.squeeze(0)
        # original code
        near_depth, far_depth = np.asarray(near_depth).item(), np.asarray(far_depth).item()
        radius_limit_np, ranges_np, vsize_np, scaled_vsize_np, scaled_vdim_np, vscale_np, \
        ranges_tensor, scaled_vox_size_tensor, scaled_vox_grid_num_tensor, kernel_size_tensor = self.get_hyperparameters(
            point_xyz_w_tensor, ranges=self.ranges)  # range means wrong !
        # print("self.opt.ranges", self.opt.ranges, range_gpu, ray_dirs_tensor)
        # pdb.set_trace()
        raypos_tensor, _, _, _ = near_far_linear_ray_generation(cam_pos_tensor, ray_dirs_tensor,
                                                                self.opt.MODEL.NEURAL_POINTS.Z_DEPTH_INTERVAL,
                                                                near=near_depth, far=far_depth,
                                                                jitter=0.0 if self.opt.IS_TRAIN else 0.)

        sample_pidx_tensor, sample_loc_w_tensor, ray_mask_tensor = self.query_grid_point_index(
                                                                                               raypos_tensor,
                                                                                               point_xyz_w_tensor,
                                                                                               actual_numpoints_tensor,
                                                                                               kernel_size_tensor,
                                                                                               self.opt.MODEL.NEURAL_POINTS.SR,
                                                                                               self.opt.MODEL.NEURAL_POINTS.K,
                                                                                               scaled_vdim_np,
                                                                                               self.opt.MODEL.NEURAL_POINTS.MAX_O,
                                                                                               self.opt.MODEL.NEURAL_POINTS.P,
                                                                                               radius_limit_np,
                                                                                               ranges_tensor,
                                                                                               scaled_vox_size_tensor,
                                                                                               scaled_vox_grid_num_tensor,)
        masked_select_ray = torch.masked_select(ray_dirs_tensor, ray_mask_tensor[..., None] > 0)
        sample_ray_dirs_tensor = masked_select_ray.reshape(ray_dirs_tensor.shape[0], -1, 3)[..., None, :]
        sample_ray_dirs_tensor = sample_ray_dirs_tensor.expand(-1, -1, self.opt.MODEL.NEURAL_POINTS.SR, -1).contiguous()
        # print("sample_ray_dirs_tensor", sample_ray_dirs_tensor.shape)
        sample_loc_cam_coor_tensor = self.w2pers(sample_loc_w_tensor, cam_rot_tensor,
                                                 cam_pos_tensor)  # this also tensor
        return sample_pidx_tensor, sample_loc_cam_coor_tensor, \
               sample_loc_w_tensor, sample_ray_dirs_tensor, ray_mask_tensor, vsize_np, ranges_np, raypos_tensor

    def w2pers(self, point_xyz_w, camrotc2w, campos):
        #     point_xyz_pers    B X M X 3
        xyz_w_shift = point_xyz_w - campos[:, None, :]
        xyz_c = torch.sum(xyz_w_shift[..., None, :] * torch.transpose(camrotc2w, 1, 2)[:, None, None, ...], dim=-1)
        z_pers = xyz_c[..., 2]
        x_pers = xyz_c[..., 0] / xyz_c[..., 2]
        y_pers = xyz_c[..., 1] / xyz_c[..., 2]
        return torch.stack([x_pers, y_pers, z_pers], dim=-1)



    def switch_pixel_id(self, pixel_idx_tensor, h):
        pixel_id = torch.cat([pixel_idx_tensor[..., 0:1], h - 1 - pixel_idx_tensor[..., 1:2]], dim=-1)
        # print("pixel_id", pixel_id.shape, torch.min(pixel_id, dim=-2)[0], torch.max(pixel_id, dim=-2)[0])
        return pixel_id

    def build_occ_vox(self, point_xyz_w_tensor, actual_numpoints_tensor, B, N, P, max_o, scaled_vdim_np,
                      scaled_vsize_gpu, scaled_vdim_gpu, kernel_size_gpu, d_coord_shift):
        # scaled_vdim_gpu (x,y,z) is the number of voxel in the Bbox
        device = point_xyz_w_tensor.device
        # point_xyz_w_tensor = point_xyz_w_tensor.to(torch.float32)

        coor_occ_tensor = torch.zeros([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], dtype=torch.int32,
                                      device=device)
        # coor_occ_tensor indicates that whether the coordinate of (Batch_id, x, y, z) has a occupancy.
        occ_2_pnts_tensor = torch.full([B, max_o, P], -1, dtype=torch.int32, device=device)
        occ_2_coor_tensor = torch.full([B, max_o, 3], -1, dtype=torch.int32, device=device)
        occ_numpnts_tensor = torch.zeros([B, max_o], dtype=torch.int32, device=device)
        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1,
                                       dtype=torch.int32, device=device)
        occ_idx_tensor = torch.zeros([B], dtype=torch.int32, device=device)
        seconds = time.time()
        assert B == 1, "Batch size must be 1!!!"
        grid_size_vol = 0

        self.claim_occ(
            point_xyz_w_tensor,
            actual_numpoints_tensor,
            np.int32(B),
            np.int32(N),
            d_coord_shift,  # 3, Bbox min;
            scaled_vsize_gpu,  # voxel size; *d_voxel_size
            scaled_vdim_gpu,  # 3 max grid for the total BBox i.e., (max_xyz - min_xyz)/ voxel_size; *d_grid_size
            np.int32(grid_size_vol),
            np.int32(max_o),
            occ_idx_tensor,
            coor_2_occ_tensor,
            occ_2_coor_tensor,
            np.uint(seconds),
            )
        """ 
        The important thing is the coor_2_occ_tensor & occ_2_coor_tensor
        scaled_vdim_gpu (x,y,z) is the number of voxel in the Bbox
        1. coor_2_occ (size of [B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]]), tell that 
            whether each voxel (i.e., coordinates) grid in the BBox has a occlued point of
        2. occ_2_coor (size of [B, max_o, 3]) store the 3-dim coordinates of each occupancy  
        """

        # torch.cuda.synchronize()
        # print("debug: ", B)

        coor_2_occ_tensor = torch.full([B, scaled_vdim_np[0], scaled_vdim_np[1], scaled_vdim_np[2]], -1,
                                       dtype=torch.int32, device=device)

        self.map_coor2occ(
            np.int32(B),
            np.int32(N),
            scaled_vdim_gpu,
            kernel_size_gpu,
            np.int32(grid_size_vol),
            np.int32(max_o),
            occ_idx_tensor,
            coor_occ_tensor,
            coor_2_occ_tensor,
            occ_2_coor_tensor,
            )
        # torch.cuda.synchronize()
        seconds = time.time()


        self.fill_occ2pnts(
            point_xyz_w_tensor,
            actual_numpoints_tensor,
            np.int32(B),
            np.int32(N),
            np.int32(P),
            d_coord_shift,
            scaled_vsize_gpu,
            scaled_vdim_gpu,
            np.int32(grid_size_vol),
            np.int32(max_o),
            coor_2_occ_tensor,
            occ_2_pnts_tensor,
            occ_numpnts_tensor,
            np.uint(seconds),
            )
        # torch.cuda.synchronize()

        return coor_occ_tensor, occ_2_coor_tensor, coor_2_occ_tensor, occ_idx_tensor, occ_numpnts_tensor, occ_2_pnts_tensor

    def query_grid_point_index(self, raypos_tensor, point_xyz_w_tensor, actual_numpoints_tensor,
                               kernel_size_tensor, SR, K, scaled_vdim_np,
                               max_o, P, radius_limit_np, ranges_tensor, scaled_vox_size_tensor,
                               scaled_vox_grid_num_tensor, ):
        # TODO: check here!! why sample_pinx is 0 in .size(1)
        if len(point_xyz_w_tensor.size()) == 2:
            point_xyz_w_tensor = point_xyz_w_tensor.unsqueeze(0)
        device = point_xyz_w_tensor.device
        point_xyz_w_tensor = point_xyz_w_tensor.to(torch.float32)
        B, N = point_xyz_w_tensor.shape[0], point_xyz_w_tensor.shape[1]
        pixel_size = scaled_vdim_np[0] * scaled_vdim_np[1]
        grid_size_vol = pixel_size * scaled_vdim_np[2]
        d_coord_shift = ranges_tensor[:3]  # ranges of bbox, :3 means the min box
        R, D = raypos_tensor.shape[1], raypos_tensor.shape[2]  # D means depth interval
        # R = pixel_idx_tensor.reshape(B, -1, 2).shape[1]

        # gridSize = int((B * N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        # print("grid Size: {}".format(gridSize))
        coor_occ_tensor, occ_2_coor_tensor, coor_2_occ_tensor, occ_idx_tensor, occ_numpnts_tensor, occ_2_pnts_tensor = \
            self.build_occ_vox(point_xyz_w_tensor, actual_numpoints_tensor, B, N, P, max_o, scaled_vdim_np,
                               scaled_vox_size_tensor, scaled_vox_grid_num_tensor, kernel_size_tensor,
                               d_coord_shift)

        # torch.cuda.synchronize()
        # print("coor_occ_tensor", torch.min(coor_occ_tensor), torch.max(coor_occ_tensor), torch.min(occ_2_coor_tensor), torch.max(occ_2_coor_tensor), torch.min(coor_2_occ_tensor), torch.max(coor_2_occ_tensor), torch.min(occ_idx_tensor), torch.max(occ_idx_tensor), torch.min(occ_numpnts_tensor), torch.max(occ_numpnts_tensor), torch.min(occ_2_pnts_tensor), torch.max(occ_2_pnts_tensor), occ_2_pnts_tensor.shape)
        # print("occ_numpnts_tensor", torch.sum(occ_numpnts_tensor > 0), ranges_np)
        # vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor)

        raypos_mask_tensor = torch.zeros([B, R, D], dtype=torch.int32, device=device)

        self.mask_raypos(
            np.int32(N),
            raypos_tensor,  # [1, 2048, 400, 3]
            coor_occ_tensor,  # [1, 2048, 400, 3]
            np.int32(B),
            np.int32(R),
            np.int32(D),
            np.int32(grid_size_vol),
            d_coord_shift,  # ranges
            scaled_vox_grid_num_tensor,  # d_grid_size
            scaled_vox_size_tensor,  # d_voxel_size
            raypos_mask_tensor,  # raypos_mask
        )


        # torch.cuda.synchronize()
        # print("raypos_mask_tensor", raypos_mask_tensor.shape, torch.sum(coor_occ_tensor), torch.sum(raypos_mask_tensor))
        # save_points(raypos_tensor.reshape(-1, 3), "./", "rawraypos_pnts")
        # raypos_masked = torch.masked_select(raypos_tensor, raypos_mask_tensor[..., None] > 0)
        # save_points(raypos_masked.reshape(-1, 3), "./", "raypos_pnts")

        ray_mask_tensor = torch.max(raypos_mask_tensor, dim=-1)[0] > 0  # B, R
        # R_item = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()

        # if R_item < 2:
        #     pdb.set_trace()
        R = torch.max(torch.sum(ray_mask_tensor.to(torch.int32))).cpu().numpy()
        if R == 0:
            print(fmt.YELLOW + "No valid ray !!! " + fmt.END)
            # pdb.set_trace()
        sample_loc_tensor = torch.zeros([B, R, SR, 3], dtype=torch.float32, device=device)
        sample_pidx_tensor = torch.full([B, R, SR, K], -1, dtype=torch.int32, device=device)
        if R > 0:
            raypos_tensor = torch.masked_select(raypos_tensor,
                                                ray_mask_tensor[..., None, None].expand(-1, -1, D, 3)).reshape(B, R, D,
                                                                                                               3)
            raypos_mask_tensor = torch.masked_select(raypos_mask_tensor,
                                                     ray_mask_tensor[..., None].expand(-1, -1, D)).reshape(B, R, D)
            # print("R", R, raypos_tensor.shape, raypos_mask_tensor.shape)

            raypos_maskcum = torch.cumsum(raypos_mask_tensor, dim=-1).to(torch.int32)
            # cumsum
            raypos_mask_tensor = (raypos_mask_tensor * raypos_maskcum * (raypos_maskcum <= SR)) - 1
            sample_loc_mask_tensor = torch.zeros([B, R, SR], dtype=torch.int32, device=device)
            self.get_shadingloc(
                np.int32(N),
                raypos_tensor,  # [1, 2048, 400, 3]
                raypos_mask_tensor,
                np.int32(B),
                np.int32(R),
                np.int32(D),
                np.int32(SR),
                sample_loc_tensor,
                sample_loc_mask_tensor,
            )

            # torch.cuda.synchronize()
            # print("shadingloc_mask_tensor", torch.sum(sample_loc_mask_tensor, dim=-1), torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0), torch.sum(sample_loc_mask_tensor > 0))
            # shadingloc_masked = torch.masked_select(sample_loc_tensor, sample_loc_mask_tensor[..., None] > 0)
            # save_points(shadingloc_masked.reshape(-1, 3), "./", "shading_pnts{}".format(self.count))

            seconds = time.time()
            self.query_along_ray(
                np.int32(N),
                point_xyz_w_tensor,
                np.int32(B),
                np.int32(SR),
                np.int32(R),
                np.int32(max_o),
                np.int32(P),
                np.int32(K),
                np.int32(grid_size_vol),
                np.float32(radius_limit_np ** 2),
                d_coord_shift,
                scaled_vox_grid_num_tensor,
                scaled_vox_size_tensor,
                kernel_size_tensor,
                occ_numpnts_tensor,
                occ_2_pnts_tensor,
                coor_2_occ_tensor,
                sample_loc_tensor,
                sample_loc_mask_tensor,
                sample_pidx_tensor,
                np.uint(seconds),
                np.int32(2),  # hard code here!!! since this is useless!
                )
            # torch.cuda.synchronize()
            # print("point_xyz_w_tensor",point_xyz_w_tensor.shape)
            # queried_masked = point_xyz_w_tensor[0][sample_pidx_tensor.reshape(-1).to(torch.int64), :]
            # save_points(queried_masked.reshape(-1, 3), "./", "queried_pnts{}".format(self.count))
            # print("valid ray",  torch.sum(torch.sum(sample_loc_mask_tensor, dim=-1) > 0))
            #
            masked_valid_ray = torch.sum(sample_pidx_tensor.view(B, R, -1) >= 0, dim=-1) > 0
            R = torch.max(torch.sum(masked_valid_ray.to(torch.int32), dim=-1)).cpu().numpy()
            ray_mask_tensor.masked_scatter_(ray_mask_tensor, masked_valid_ray)
            sample_pidx_tensor = torch.masked_select(sample_pidx_tensor,
                                                     masked_valid_ray[..., None, None].expand(-1, -1, SR, K)).reshape(B,
                                                                                                                      R,
                                                                                                                      SR,
                                                                                                                      K)
            sample_loc_tensor = torch.masked_select(sample_loc_tensor,
                                                    masked_valid_ray[..., None, None].expand(-1, -1, SR, 3)).reshape(B,
                                                                                                                     R,
                                                                                                                     SR,
                                                                                                                     3)
        # self.count+=1
        return sample_pidx_tensor, sample_loc_tensor, ray_mask_tensor.to(torch.int8)


def load_pnts(point_path, point_num):
    with open(point_path, 'rb') as f:
        print("point_file_path################", point_path)
        all_infos = pickle.load(f)
        point_xyz = all_infos["point_xyz"]
    print(len(point_xyz), point_xyz.dtype, np.mean(point_xyz, axis=0), np.min(point_xyz, axis=0),
          np.max(point_xyz, axis=0))
    np.random.shuffle(point_xyz)
    return point_xyz[:min(len(point_xyz), point_num), :]


def save_points(xyz, dir, filename):
    if xyz.ndim < 3:
        xyz = xyz[None, ...]
    filename = "{}.txt".format(filename)
    os.makedirs(dir, exist_ok=True)
    filepath = os.path.join(dir, filename)
    print("save at {}".format(filepath))
    if torch.is_tensor(xyz):
        np.savetxt(filepath, xyz.cpu().reshape(-1, xyz.shape[-1]), delimiter=";")
    else:
        np.savetxt(filepath, xyz.reshape(-1, xyz.shape[-1]), delimiter=";")






def w2img(point_xyz, transform_matrix, focal):
    camrot = transform_matrix[:3, :3]  # world 2 cam
    campos = transform_matrix[:3, 3]  #
    point_xyz_shift = point_xyz - campos[None, :]
    # xyz = np.sum(point_xyz_shift[:,None,:] * camrot.T, axis=-1)
    xyz = np.sum(camrot[None, ...] * point_xyz_shift[:, :, None], axis=-2)
    # print(xyz.shape, np.sum(camrot[None, None, ...] * point_xyz_shift[:,:,None], axis=-2).shape)
    xper = xyz[:, 0] / -xyz[:, 2]
    yper = xyz[:, 1] / xyz[:, 2]
    x_pixel = np.round(xper * focal + 400).astype(np.int32)
    y_pixel = np.round(yper * focal + 400).astype(np.int32)
    print("focal", focal, np.tan(.5 * 0.6911112070083618))
    print("pixel xmax xmin:", np.max(x_pixel), np.min(x_pixel), "pixel ymax ymin:", np.max(y_pixel), np.min(y_pixel))
    print("per xmax xmin:", np.max(xper), np.min(xper), "per ymax ymin:", np.max(yper), np.min(yper), "per zmax zmin:",
          np.max(xyz[:, 2]), np.min(xyz[:, 2]))
    print("min perx", -400 / focal, "max perx", 400 / focal)
    background = np.ones([800, 800, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .2

    plt.figure()
    plt.imshow(background)

    return np.stack([xper, yper, -xyz[:, 2]], axis=-1)


def render_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
    pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
    print(pixel_xy_inds.shape)
    y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .5
    plt.figure()
    plt.imshow(background)


# def save_mask_pers_points(queried_point_xyz, vsize, ranges, w, h):
#     pixel_xy_inds = np.floor((queried_point_xyz[:, :2] - ranges[None, :2]) / vsize[None, :2]).astype(np.int32)
#     print(pixel_xy_inds.shape)
#     y_pixel, x_pixel = pixel_xy_inds[:, 1], pixel_xy_inds[:, 0]
#     background = np.ones([h, w, 3], dtype=np.float32)
#     background[y_pixel, x_pixel, :] = .5
#     image_dir = os.path.join(self.opt.checkpoints_dir, opt.name, 'images')
#     image_file = os.path.join(image_dir)


def render_pixel_mask(pixel_xy_inds, w, h):
    y_pixel, x_pixel = pixel_xy_inds[0, :, 1], pixel_xy_inds[0, :, 0]
    background = np.ones([h, w, 3], dtype=np.float32)
    background[y_pixel, x_pixel, :] = .0
    plt.figure()
    plt.imshow(background)

def vis_vox(ranges_np, scaled_vsize_np, coor_2_occ_tensor):
    print("ranges_np", ranges_np, scaled_vsize_np)
    mask = coor_2_occ_tensor.cpu().numpy() > 0
    xdim, ydim, zdim = coor_2_occ_tensor.shape[1:]
    x_ = np.arange(0, xdim)
    y_ = np.arange(0, ydim)
    z_ = np.arange(0, zdim)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    xyz = np.stack([x,y,z], axis=-1).reshape(-1,3).astype(np.float32)
    xyz = ranges_np[None, :3] + (xyz + 0.5) * scaled_vsize_np[None, :]
    xyz = xyz[mask.reshape(-1)]
    save_points(xyz, "./", "occ_xyz")
    print(xyz.shape)

def save_queried_points(point_xyz_tensor, point_xyz_pers_tensor, sample_pidx_tensor, pixel_idx_tensor, pixel_idx_cur_tensor, vdim, vsize, ranges):
    B, R, SR, K = sample_pidx_tensor.shape
    # pixel_inds = torch.as_tensor([3210, 3217,3218,3219,3220, 3221,3222,3223,3224,3225,3226,3227,3228,3229,3230, 3231,3232,3233,3234,3235, 3236,3237,3238,3239,3240], device=sample_pidx_tensor.device, dtype=torch.int64)
    point_inds = sample_pidx_tensor[0, :, :, :]
    # point_inds = sample_pidx_tensor[0, pixel_inds, :, :]
    mask = point_inds > -1
    point_inds = torch.masked_select(point_inds, mask).to(torch.int64)
    queried_point_xyz_tensor = point_xyz_tensor[0, point_inds, :]
    queried_point_xyz = queried_point_xyz_tensor.cpu().numpy()
    print("queried_point_xyz.shape", B, R, SR, K, point_inds.shape, queried_point_xyz_tensor.shape,
          queried_point_xyz.shape)
    print("pixel_idx_cur_tensor", pixel_idx_cur_tensor.shape)
    render_pixel_mask(pixel_idx_cur_tensor.cpu().numpy(), vdim[0], vdim[1])

    render_mask_pers_points(point_xyz_pers_tensor[0, point_inds, :].cpu().numpy(), vsize, ranges, vdim[0], vdim[1])

    plt.show()

def load_init_points(scan, data_dir="/home/xharlie/user_space/data/nrData/nerf/nerf_synthetic_colmap"):
    points_path = os.path.join(data_dir, scan, "colmap_results/dense/fused.ply")
    # points_path = os.path.join(self.data_dir, self.scan, "exported/pcd_te_1_vs_0.01_jit.ply")
    assert os.path.exists(points_path)
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(points_path)
    # plydata (PlyProperty('x', 'double'), PlyProperty('y', 'double'), PlyProperty('z', 'double'), PlyProperty('nx', 'double'), PlyProperty('ny', 'double'), PlyProperty('nz', 'double'), PlyProperty('red', 'uchar'), PlyProperty('green', 'uchar'), PlyProperty('blue', 'uchar'))
    print("plydata", plydata.elements[0])
    x,y,z=torch.as_tensor(plydata.elements[0].data["x"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["y"].astype(np.float32), device="cuda", dtype=torch.float32), torch.as_tensor(plydata.elements[0].data["z"].astype(np.float32), device="cuda", dtype=torch.float32)
    points_xyz = torch.stack([x,y,z], dim=-1).to(torch.float32)
    return points_xyz

def try_build(ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, near_depth, far_depth, shading_count, split=["train"], imgidx=0, gpu=0, NN=2):
    # point_path = os.path.join(point_dir, point_file)
    # point_xyz = load_pnts(point_path, 819200000)  # 81920   233872
    # point_xyz = load_init_points(obj)
    point_xyz = np.random.rand(10000, 3) * 1
    # imgs, poses, _, _, _, intrinsic = load_blender_data(
    #     os.path.expandvars("${nrDataRoot}") + "/nerf/nerf_synthetic/{}".format(obj), split, half_res=False, testskip=1)
    # H, W, focal = hwf
    H, W, focal = 680, 1200, 600
    intrinsic = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
    # plt.figure()
    # plt.imshow(imgs[imgidx])
    point_xyz_w_tensor = torch.as_tensor(point_xyz, device="cuda:{}".format(gpu))[None, ...]
    print("point_xyz_w_tensor", point_xyz_w_tensor[0].shape, torch.min(point_xyz_w_tensor[0], dim=0)[0],
          torch.max(point_xyz_w_tensor[0], dim=0)[0])
    # plt.show()
    actual_numpoints_tensor = torch.ones([1], device=point_xyz_w_tensor.device, dtype=torch.int32) * len(
        point_xyz_w_tensor[0])
    # range_gpu, vsize_gpu, vdim_gpu, vscale_gpu, kernel_size_gpu = np_to_gpuarray(ranges, scaled_vsize, scaled_vdim, vscale, kernel_size)
    pixel_idx_tensor = torch.as_tensor(pixel_idx, device=point_xyz_w_tensor.device, dtype=torch.int32)[None, ...]
    c2w = np.array([[-3.20569622e-01,  9.47224956e-01,  8.15535109e-17,  6.76303974e-02],
                    [ 4.48055195e-01,  1.51635452e-01, -8.81052344e-01, -1.09304628e-01],
                    [-8.34554767e-01, -2.82438617e-01, -4.73018782e-01,  3.29090427e-01],
                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    print("c2w", c2w.shape, pixel_idx.shape)
    from data.data_utils import get_dtu_raydir
    cam_pos, camrot = c2w[:3, 3], c2w[:3, :3]
    ray_dirs_tensor, cam_pos_tensor = torch.as_tensor(get_dtu_raydir(pixel_idx, intrinsic, camrot, True),
                                                      device=pixel_idx_tensor.device,
                                                      dtype=torch.float32), torch.as_tensor(cam_pos,
                                                                                            device=pixel_idx_tensor.device,
                                                                                            dtype=torch.float32)
    cam_rot_tensor = torch.as_tensor(camrot,
                                     device=pixel_idx_tensor.device,
                                     dtype=torch.float32)

    from collections import namedtuple
    opt_construct = namedtuple('opt',
                               'inverse vsize vscale kernel_size radius_limit_scale depth_limit_scale max_o P SR K gpu_maxthr NN ranges z_depth_dim query_size')
    opt = opt_construct(inverse=0, vscale=vscale, vsize=vsize, kernel_size=kernel_size, radius_limit_scale=0,
                        depth_limit_scale=0, max_o=max_o, P=P, SR=SR, K=K, gpu_maxthr=1024, NN=NN, ranges=ranges,
                        z_depth_dim=400, query_size="3 3 3")

    opt = setup_config()


    querier = lighting_fast_querier(point_xyz_w_tensor.device, opt)
    print("actual_numpoints_tensor", actual_numpoints_tensor)

    querier.query_points(pixel_idx_tensor, point_xyz_w_tensor, actual_numpoints_tensor, H, W,
                         near_depth, far_depth, ray_dirs_tensor[None, ...], cam_pos_tensor[None, ...],
                         cam_rot_tensor[None, ...])


if __name__ == "__main__":
    obj = "lego"
    # point_file = "{}.pkl".format(obj)
    # point_dir = os.path.expandvars("${nrDataRoot}/nerf/nerf_synthetic_points/")
    r = 0.36000002589322094
    ranges = np.array([-1., -1.3, -1.2, 1., 1.3, 1.2], dtype=np.float32)
    vdim = np.array([400, 400, 400], dtype=np.int32)
    # vsize = np.array([2 * r / vdim[0], 2 * r / vdim[1], 4. / vdim[2]], dtype=np.float32)
    vsize = np.array([0.005, 0.005, 0.005], dtype=np.float32)
    vscale = np.array([2, 2, 2], dtype=np.int32)
    SR = 24
    P = 128
    K = 8
    NN = 2
    ray_num = 2048
    kernel_size = np.array([5, 5, 5], dtype=np.int32)
    radius_limit = 0  # r / 400 * 5 #r / 400 * 5
    depth_limit = 0  # 4. / 400 * 1.5 # r / 400 * 2
    max_o = 500000
    near_depth, far_depth = 2., 6.
    shading_count = 400

    xrange = np.arange(0, 800, 1, dtype=np.int32)
    yrange = np.arange(0, 800, 1, dtype=np.int32)
    xv, yv = np.meshgrid(xrange, yrange, sparse=False, indexing='ij')
    inds = np.arange(len(xv.reshape(-1)), dtype=np.int32)
    np.random.shuffle(inds)
    inds = inds[:ray_num, ...]
    pixel_idx = np.stack([xv, yv], axis=-1).reshape(-1, 2)[inds]  # 20000 * 2
    gpu = 0
    imgidx = 3
    split = ["train"]

    try_build(ranges, vsize, vdim, vscale, max_o, P, kernel_size, SR, K, pixel_idx, obj,
              radius_limit, depth_limit, near_depth, far_depth, shading_count, split=split, imgidx=imgidx, gpu=0, NN=NN)