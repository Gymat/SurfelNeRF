import pdb

import numpy as np
import torch
import random
from PIL import Image
import os
import copy
from torchvision import transforms
import pickle
import json
from pytorch3d.renderer.cameras import look_at_view_transform
from collections import OrderedDict

from data.base import BaseDataset
from .build import DATASET_REGISTRY
from .data_utils import get_dtu_raydir


@DATASET_REGISTRY.register()
class Scannet_S1_multi_scenes(BaseDataset):
    def __init__(self, config, **kwargs):
        super(Scannet_S1_multi_scenes, self).__init__()
        cfg = config.DATA
        self.cfg = config
        self.data_dir = cfg.DATA_DIR
        self.is_train = config.IS_TRAIN
        #
        self.img_wh = (int(cfg.IMG_WH[0]), int(cfg.IMG_WH[1]))
        # self.ranges = strlist_to_floatlist(cfg.RANGES)
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])
        no_build = kwargs.get('no_build', False)
        self.transformers = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        if no_build:
            print("Using base dataset.")
            pass
        else:
            #
            self.rgb_image_path = cfg.RGB_Image_DIR  # root image path for all scenes
            self.depth_image_path = cfg.Depth_Image_DIR  # root depth path for all scenes
            self.scannet_json_path = cfg.ALL_JSON_PATH  # New
            self.scannet_pkl_path = cfg.SCANNET_PKL_PATH  # New

            self.bg_color = (1, 1, 1)
            self.to_tensor = transforms.ToTensor()
            self.get_item_type = cfg.GET_ITEM_TYPE  # New
            # --------------------------------------------------------------------------
            self.load_meta_data()
            print("Dataset is ready!")

    def _load_rgb_intrinsics(self, scene_name,):
        intrinsic_path = os.path.join(self.data_dir, scene_name, 'intrinsic', 'intrinsic_color.txt')
        np_mat = read_pose_file(intrinsic_path)
        mat = np_mat[:3, :3]
        # re-scale
        scale_x = self.img_wh[0] / 1296
        scale_y = self.img_wh[1] / 968
        mat[0, :] = mat[0, :] * scale_x
        mat[1, :] = mat[1, :] * scale_y
        return mat

    def _load_depth_intrinsics(self, scene_name,):
        intrinsic_path = os.path.join(self.data_dir, scene_name, 'intrinsic', 'intrinsic_depth.txt')
        np_mat = read_pose_file(intrinsic_path)
        mat = np_mat[:3, :3]
        # re-scale
        scale_x = self.img_wh[0] / 640
        scale_y = self.img_wh[1] / 480
        mat[0, :] = mat[0, :] * scale_x
        mat[1, :] = mat[1, :] * scale_y
        return mat

    def _get_fov_h(self, h, fy):
        fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))
        return fov_y

    def load_meta_data(self):
        self.scene_name_list = []
        self.scene_num_list = []
        self.intrinsic_dict = OrderedDict()
        self.fov_dict = OrderedDict()
        scenes_list = os.listdir(self.scannet_pkl_path)
        scenes_list.sort()
        scene_name_dict = OrderedDict()
        scene_kf_dict = OrderedDict()
        scene_non_kf_dict = OrderedDict()
        for scene_name in scenes_list:
            scene_len = len(list(os.listdir(os.path.join(self.data_dir, scene_name,  'color'))))
            key_f_id_list = [i for i in range(scene_len) if i % 10 == 0]
            nonkey_f_id_list = [i for i in range(scene_len) if i % 10 != 0]
            name_list = [i for i in range(scene_len)]
            #
            self.scene_num_list.append(scene_len)
            self.scene_name_list.append(scene_name)
            scene_kf_dict[scene_name] = key_f_id_list
            scene_non_kf_dict[scene_name] = nonkey_f_id_list
            scene_name_dict[scene_name] = name_list
            self.intrinsic_dict[scene_name] = self._load_rgb_intrinsics(scene_name)
            self.fov_dict[scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[scene_name][1, 1])
        self.scene_kf_dict = scene_kf_dict
        self.scene_non_kf_dict = scene_non_kf_dict
        self.scene_name_dict = scene_name_dict
        self.num_key_list = [len(scene_kf_dict[sc]) - 1 for sc in scene_kf_dict.keys()]


    def __len__(self):
        total_kf_num = sum(self.num_key_list)
        if self.get_item_type == 1:
            return sum(self.scene_num_list) - 1
        return total_kf_num

    def parsing_all_id_frame(self, id):
        cum_num = np.cumsum(self.scene_num_list)
        for index, csum in enumerate(cum_num):
            if id < csum:
                break
        index_before = index - 1
        if index_before < 0:
            cum_num_before = 0
        else:
            cum_num_before = cum_num[index_before]
        id_in_scene = id - cum_num_before
        scene_name = self.scene_name_list[index]
        return scene_name, id_in_scene


    def parsing_input_id_key_frame(self, id):
        # Note: return key frame id !!!

        cum_num = np.cumsum(self.num_key_list)
        for index, csum in enumerate(cum_num):
            if id < csum:
                break
        index_before = index - 1
        if index_before < 0:
            cum_num_before = 0
        else:
            cum_num_before = cum_num[index_before]
        id_in_scene = id - cum_num_before
        name = self.scene_name_list[index]
        kf_id = self.scene_kf_dict[name][id_in_scene]
        return kf_id, name, id_in_scene

    def __getitem__(self, id):
        if self.get_item_type == 1:
            raise NotImplementedError()
        elif self.get_item_type == 2:
            raise NotImplementedError()
        elif self.get_item_type == 3:
            kf_id, scene_name, id_kf_list = self.parsing_input_id_key_frame(id)
            scene_img_name = scene_name
            # Provide a local fragment to train
            key_frame_id1 = self.scene_kf_dict[scene_name][id_kf_list]
            key_frame_id2 = self.scene_kf_dict[scene_name][id_kf_list + 1]
            assert key_frame_id1 == kf_id
            gt_frame_id = np.random.randint(key_frame_id1, key_frame_id2 + 1)
            #
            data_k1 = self.get_item_data(key_frame_id1, scene_name)
            data_k2 = self.get_item_data(key_frame_id2, scene_name)
            data_gt = self.get_train_data(gt_frame_id, scene_name)
            # key frame 1
            img_path_k1 = os.path.join(self.rgb_image_path, scene_img_name, 'color', "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, _ = self._get_RT_lookat(lookat=data_k1['look_at'], cam_pos=data_k1['campos'],
                                                        up=torch.Tensor(data_k1['up']))
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1, intrinsic_matrix=data_k1['intrinsic'],
                                                                            camrot=rot_c2w_k1)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1
            }
            # key frame 2
            img_path_k2 = os.path.join(self.rgb_image_path, scene_img_name, 'color', "{}.jpg".format(data_k2['rgb_img_name'].split('.')[0]))
            input_img_k2, img_k2 = self.read_rgb_image(img_path_k2)
            rot_c2w_k2, campos_k2, _ = self._get_RT_lookat(lookat=data_k2['look_at'], cam_pos=data_k2['campos'],
                                                        up=torch.Tensor(data_k2['up']))
            gt_k2, pixelcoords_k2, raydir_k2, pxpy_k2 = self._obtain_train_info(img_k2,
                                                                                intrinsic_matrix=data_k2['intrinsic'],
                                                                                camrot=rot_c2w_k2)
            cam_para_k2 = {'R': rot_c2w_k2, 'T': campos_k2, 'fov': data_k2['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_k2 = {
                'data': data_k2, 'gt': gt_k2, 'rgb': input_img_k2, 'sampled_pidx': pxpy_k2, 'sampled_ray': raydir_k2,
                'camera': cam_para_k2
            }
            # train frame
            img_path_item = os.path.join(self.rgb_image_path, scene_img_name, 'color',
                                         "{}.jpg".format(data_gt['rgb_img_name'].split('.')[0]))
            input_img_item, img_item = self.read_rgb_image(img_path_item)
            rot_c2w_item, campos_item, _ = self._get_RT_lookat(lookat=data_gt['look_at'], cam_pos=data_gt['campos'],
                                                        up=torch.Tensor(data_gt['up']))
            gt_item, pixelcoords_item, raydir_item, pxpy_item = self._obtain_train_info(img_item,
                                                                                intrinsic_matrix=data_gt['intrinsic'],
                                                                                camrot=rot_c2w_item)
            cam_para_item = {'R': rot_c2w_item, 'T': campos_item, 'fov': data_gt['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_item = {
                'gt': gt_item, 'rgb': input_img_item, 'sampled_pidx': pxpy_item,
                'sampled_ray': raydir_item,
                'camera': cam_para_item
            }
            output = {0: output_data_k1, 1: output_data_k2, 2: output_data_item, }
            return output
        else:
            raise RuntimeError("Get_item_type is wrong.")

    def get_item_data(self, id, scene_name=None):
        assert scene_name is not None
        pkl_path = os.path.join(self.scannet_pkl_path,
                                scene_name,
                                "{}.pkl".format(str(self.scene_name_dict[scene_name][id]).zfill(4)))
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        data['pcd_dirs'] = data['pcd_dirs'] / (np.linalg.norm(data['pcd_dirs'], axis=-1, keepdims=True) + 1e-6)

        data['look_at'] = np.array(data['look_at'])
        data['look_at'] = data['look_at'][None, ]
        data['campos'] = np.array(data['campos'])
        data['campos'] = data['campos'][None,]
        data['up'] = np.array(data['up'])[None, ]
        data['intrinsic'] = self.intrinsic_dict[scene_name]
        data['weights'] = data['weights'].reshape(-1, 1)
        data['id'] = id
        data['scene'] = scene_name
        data['fov'] = self.fov_dict[scene_name]
        return data

    def get_train_data(self, id, scene_name):
        json_path = os.path.join(self.scannet_json_path, "{}.json".format(scene_name))  # new
        with open(json_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        data = dict()
        up = json_object[str(id)]['up']
        if up[-1] < 0:
            up = [x*-1 for x in up]
        # data['pcd_dirs'] = data['pcd_dirs'] / np.linalg.norm(data['pcd_dirs'], axis=-1)
        data['up'] = np.array(up)[None, ]
        data['look_at'] = np.array(json_object[str(id)]['look_at'])[None, ]
        data['campos'] = np.array(json_object[str(id)]['campos'])[None, ]
        data['intrinsic'] = self.intrinsic_dict[scene_name]
        data['rgb_img_name'] = json_object[str(id)]['rgb_img_name']
        data['depth_img_name'] = json_object[str(id)]['depth_img_name']
        data['fov'] = self.fov_dict[scene_name]
        return data

    def read_rgb_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((self.img_wh[0], self.img_wh[1]))  # NOTE: PIL.resize is to (w, H)

        img1 = self.transformers(img)
        # to tensor only
        totensor = transforms.ToTensor()
        img_tensor = totensor(img)
        img_tensor = img_tensor.permute(1, 2, 0)  # (H, W, 3)
        return (img1, img_tensor)

    def _get_RT_lookat(self, lookat, cam_pos, up, **kwargs):
        skip_flag = False
        R, T = look_at_view_transform(eye=cam_pos, at=lookat, up=up)
        if torch.isnan(T[0, 0]):
            print('look_at: ', lookat)
            print('campos:', cam_pos)
            print('up:', up)
            skip_flag = True
        # R is (world to camera).T -- > so R is the c2w hhhhh
        return R.squeeze(0), T.squeeze(0), skip_flag

    def _obtain_train_info(self, img_tensor, intrinsic_matrix, camrot):
        '''

        :param img_tensor: (H, W, 3)
        :return:
        '''
        # Random sample pixel for NeRF
        subsamplesize = self.cfg.DATA.RANDOM_SAMPLE_SIZE
        pixelcoords, px, py = self._sample_pixel(subsamplesize, self.width, self.height)  # H x W x 2
        px = torch.LongTensor(px)
        py = torch.LongTensor(py)
        gt = img_tensor[py, px, :]
        gt = gt.reshape(-1, 3)
        # print(pixelcoords)
        # print(intrinsic_matrix)
        raydir = get_dtu_raydir(pixelcoords, intrinsic_matrix, camrot.squeeze(0).numpy(), dir_norm=True)  # H x W x 3
        raydir = np.reshape(raydir, (-1, 3))
        raydir = torch.FloatTensor(raydir)
        return gt, pixelcoords, raydir, (px, py)

    def obtain_test_info(self, intrinsic_matrix, camrot):
        '''

        :param img_tensor: (H, W, 3)
        :return:
        '''
        # Random sample pixel for NeRF
        subsamplesize = self.cfg.DATA.RANDOM_SAMPLE_SIZE
        pixelcoords, px, py = self._sample_pixel(subsamplesize, self.width, self.height)  # H x W x 2
        px = torch.LongTensor(px)
        py = torch.LongTensor(py)
        raydir = get_dtu_raydir(pixelcoords, intrinsic_matrix, camrot.squeeze(0).numpy(), dir_norm=True)  # H x W x 3
        raydir = np.reshape(raydir, (-1, 3))
        raydir = torch.FloatTensor(raydir)
        return None, pixelcoords, raydir, (px, py)

    def _sample_pixel(self, subsamplesize, width, height):
        if self.is_train:
            if self.cfg.DATA.RANDOM_SAMPLE_WAY == "patch":
                indx = self.cfg.indx
                indy = self.cfg.indy
                max_x = min(indx + subsamplesize, 640)
                max_y = min(indy + subsamplesize, 480)
                beginx = max_x - subsamplesize
                beginy = max_y - subsamplesize
                px, py = np.meshgrid(
                    np.arange(beginx, max_x).astype(np.float32),
                    np.arange(beginy, max_y).astype(np.float32))
            elif self.cfg.DATA.RANDOM_SAMPLE_WAY == "random":
                px = np.random.randint(6,
                                       width - 6,
                                       size=(subsamplesize,
                                             subsamplesize)).astype(np.float32)
                py = np.random.randint(6,
                                       height - 6,
                                       size=(subsamplesize,
                                             subsamplesize)).astype(np.float32)
            elif self.cfg.DATA.RANDOM_SAMPLE_WAY == "random2":
                px = np.random.uniform(0,
                                       width - 1e-5,
                                       size=(subsamplesize,
                                             subsamplesize)).astype(np.float32)
                py = np.random.uniform(0,
                                       height - 1e-5,
                                       size=(subsamplesize,
                                             subsamplesize)).astype(np.float32)
            elif self.cfg.DATA.RANDOM_SAMPLE_WAY == "proportional_random":
                raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        return pixelcoords, px, py

    def load_c2w_mat(self, img_id, scnen_name):
        txt_path = os.path.join(self.data_dir, scnen_name, 'pose', "{}.txt".format(img_id))
        c2w_mat = read_pose_file(txt_path)
        w2c_mat = np.linalg.inv(c2w_mat)
        return c2w_mat, w2c_mat


@DATASET_REGISTRY.register()
class Scannet_S1_multi_scenes_valid_index(Scannet_S1_multi_scenes):
    def __init__(self, config, **kwargs):
        super(Scannet_S1_multi_scenes_valid_index, self).__init__(config, **kwargs)

    def load_meta_data(self):
        self.scene_name_list = []
        self.scene_num_list = []
        self.intrinsic_dict = OrderedDict()
        self.fov_dict = OrderedDict()
        scenes_list = os.listdir(self.scannet_pkl_path)
        scenes_list.sort()
        scene_name_dict = OrderedDict()
        scene_kf_dict = OrderedDict()
        scene_non_kf_dict = OrderedDict()
        for scene_name in scenes_list:
            scene_len = len(list(os.listdir(os.path.join(self.data_dir, scene_name,  'color'))))
            key_f_id_list = [i for i in range(scene_len) if i % 10 == 0]
            nonkey_f_id_list = [i for i in range(scene_len) if i % 10 != 0]
            name_list = [i for i in range(scene_len)]
            #
            self.scene_num_list.append(scene_len)
            self.scene_name_list.append(scene_name)
            scene_kf_dict[scene_name] = key_f_id_list
            scene_non_kf_dict[scene_name] = nonkey_f_id_list
            scene_name_dict[scene_name] = name_list
            self.intrinsic_dict[scene_name] = self._load_rgb_intrinsics(scene_name)
            self.fov_dict[scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[scene_name][1, 1])
        self.scene_kf_dict = scene_kf_dict
        # self.scene_non_kf_dict = scene_non_kf_dict
        self.scene_name_dict = scene_name_dict
        self.num_key_list = [len(scene_kf_dict[sc]) - 1 for sc in scene_kf_dict.keys()]
        # New
        with open(self.cfg.DATA.SCANNET_VALID_INPUT_LIST, 'rb') as f:
            self.valid_input_list = pickle.load(f)

    def __len__(self):
        return len(self.valid_input_list) - 1

    def obtain_key_id(self, train_id, scene_name):
        #
        skip_flag = False
        if train_id % 10 == 0:
            key_id_2 = train_id
            if key_id_2 > 10:
                key_id_1 = train_id - 10
            else:
                key_id_1 = train_id + 10
        else:
            num_key_step = train_id // 10
            key_id_1 = 10 * num_key_step
            key_id_2 = 10 * (num_key_step + 1)
            if key_id_2 > self.scene_kf_dict[scene_name][-1]:
                key_id_2 = key_id_1
                skip_flag = True
        return key_id_1, key_id_2, skip_flag

    def _get_RT_lookat(self, lookat, cam_pos, up, **kwargs):
        skip_flag = False
        string_list = kwargs.get('input', '')
        R, T = look_at_view_transform(eye=cam_pos, at=lookat, up=up)
        if torch.isnan(T[0, 0]):
            print('look_at: ', lookat)
            print('campos:', cam_pos)
            print('up:', up)
            skip_flag = True
            print("need to be skipped! Details: {}".format(input),)
        # R is (world to camera).T -- > so R is the c2w hhhhh
        return R.squeeze(0), T.squeeze(0), skip_flag

    def __getitem__(self, id):
        if self.get_item_type == 1:
            raise NotImplementedError()
        elif self.get_item_type == 2:
            raise NotImplementedError()
        elif self.get_item_type == 3:
            valid_input = self.valid_input_list[id]
            words = valid_input.split('__')
            scene_name = words[0]
            scene_img_name = scene_name
            gt_frame_id = int(words[1])
            key_frame_id1, key_frame_id2, skip_flag = self.obtain_key_id(train_id=gt_frame_id, scene_name=scene_name)
            #
            data_k1 = self.get_item_data(key_frame_id1, scene_name)
            data_k2 = self.get_item_data(key_frame_id2, scene_name)
            data_gt = self.get_train_data(gt_frame_id, scene_name)
            # key frame 1
            img_path_k1 = os.path.join(self.rgb_image_path, scene_img_name, 'color', "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, skip_flag1 = self._get_RT_lookat(lookat=data_k1['look_at'], cam_pos=data_k1['campos'],
                                                        up=torch.Tensor(data_k1['up']), input=valid_input)
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1, intrinsic_matrix=data_k1['intrinsic'],
                                                                            camrot=rot_c2w_k1)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1
            }
            # key frame 2
            img_path_k2 = os.path.join(self.rgb_image_path, scene_img_name, 'color', "{}.jpg".format(data_k2['rgb_img_name'].split('.')[0]))
            input_img_k2, img_k2 = self.read_rgb_image(img_path_k2)
            rot_c2w_k2, campos_k2, skip_flag2 = self._get_RT_lookat(lookat=data_k2['look_at'], cam_pos=data_k2['campos'],
                                                        up=torch.Tensor(data_k2['up']), input=valid_input)
            gt_k2, pixelcoords_k2, raydir_k2, pxpy_k2 = self._obtain_train_info(img_k2,
                                                                                intrinsic_matrix=data_k2['intrinsic'],
                                                                                camrot=rot_c2w_k2)
            cam_para_k2 = {'R': rot_c2w_k2, 'T': campos_k2, 'fov': data_k2['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_k2 = {
                'data': data_k2, 'gt': gt_k2, 'rgb': input_img_k2, 'sampled_pidx': pxpy_k2, 'sampled_ray': raydir_k2,
                'camera': cam_para_k2
            }
            # train frame
            img_path_item = os.path.join(self.rgb_image_path, scene_img_name, 'color',
                                         "{}.jpg".format(data_gt['rgb_img_name'].split('.')[0]))
            input_img_item, img_item = self.read_rgb_image(img_path_item)
            rot_c2w_item, campos_item, skip_flag_gt = self._get_RT_lookat(lookat=data_gt['look_at'], cam_pos=data_gt['campos'],
                                                        up=torch.Tensor(data_gt['up']), input=valid_input)
            gt_item, pixelcoords_item, raydir_item, pxpy_item = self._obtain_train_info(img_item,
                                                                                intrinsic_matrix=data_gt['intrinsic'],
                                                                                camrot=rot_c2w_item)
            cam_para_item = {'R': rot_c2w_item, 'T': campos_item, 'fov': data_gt['fov'], 'aspect_ratio': 4.0/3.0}
            output_data_item = {
                'gt': gt_item, 'rgb': input_img_item, 'sampled_pidx': pxpy_item,
                'sampled_ray': raydir_item,
                'camera': cam_para_item
            }
            skip_flag_f = skip_flag or skip_flag_gt or skip_flag1 or skip_flag2
            if skip_flag_f:
                print(valid_input)
            output = {0: output_data_k1, 1: output_data_k2, 2: output_data_item, 'skip_flag': skip_flag_f}
            return output
        else:
            raise RuntimeError("Get_item_type is wrong.")


@DATASET_REGISTRY.register()
class Scannet_S1_multi_scenes_valid_index_scene_based(Scannet_S1_multi_scenes):
    '''
    This class provide sequential data which contains a key-frame and a random gt frame.
    '''
    def __init__(self, config, **kwargs):
        self.min_id_scene=0
        super(Scannet_S1_multi_scenes_valid_index_scene_based, self).__init__(config, **kwargs)

    def load_meta_data(self):
        # self.scene_name_list = []
        self.intrinsic_dict = OrderedDict()
        self.intrinsic_depth_dict = OrderedDict()
        self.fov_dict = OrderedDict()
        #
        with open(self.cfg.DATA.SCANNET_VALID_INPUT_LIST, 'rb') as f:
            self.valid_input_list = pickle.load(f)
        #
        self.kf_order_list = []
        self.scene_kf_dict = OrderedDict()
        self.scene_non_kf_dict = OrderedDict()
        self.hard_new_scene = []
        # parsing the valid_input_list
        for index in range(len(self.valid_input_list)):
            line = self.valid_input_list[index]
            words = line.split('__')
            scene_name = words[0]
            valid_id = int(words[1])
            # init the dict:
            if scene_name not in self.scene_kf_dict:
                self.hard_new_scene.append(True)
                self.kf_order_list.append(line)
                # self.scene_num_dict[scene_name] = 0
                assert valid_id == 0
                self.scene_kf_dict[scene_name] = [0]
                self.scene_non_kf_dict[scene_name] = dict()
                self.scene_non_kf_dict[scene_name][0] = [0]
                key_frame_id_right_now = 0
                temp_non_key = [0]
                # init corresponding dict
                self.intrinsic_dict[scene_name] = self._load_rgb_intrinsics(scene_name)
                self.intrinsic_depth_dict[scene_name] = self._load_depth_intrinsics(scene_name)
                self.fov_dict[scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[scene_name][1, 1])
                # self.scene_non_kf_dict[scene_name][key_frame_id_right_now] = []
            else:
                self.hard_new_scene.append(False)
            # filter index
            if valid_id % 10 == 0 and valid_id > 0:
                self.kf_order_list.append(line)
                # temp_non_key.append(valid_id)
                if len(temp_non_key) == 0:
                    raise RuntimeError("!!!")
                self.scene_non_kf_dict[scene_name][valid_id] = temp_non_key
                self.scene_kf_dict[scene_name].append(valid_id)
                key_frame_id_right_now = valid_id
                temp_non_key = [valid_id]
            else:
                temp_non_key.append(valid_id)
        #
         
        total_num = sum([len(self.scene_kf_dict[k]) for k in self.scene_kf_dict.keys()])
        assert total_num == len(self.kf_order_list)
        print("Total num of training: ", total_num)
        # stage store scene name
        self.scene_name_now = None

    def __len__(self):
        if self.cfg.DEBUG:
            return 500
        return len(self.kf_order_list)

    def __getitem__(self, id):
        if self.get_item_type == 1:
            raise NotImplementedError()
        elif self.get_item_type == 2:

            new_scene_flag = False
            #
            valid_line = self.kf_order_list[id]
            words = valid_line.split('__')
            scene_name = words[0]
            kf_id = int(words[1])
            # whethe this is a new scene
            if self.scene_name_now is None:
                self.scene_name_now = scene_name
                new_scene_flag = True
                self.min_id_scene = kf_id
            else:
                if self.scene_name_now != scene_name:
                    new_scene_flag = True
                    self.scene_name_now = scene_name
                if self.hard_new_scene[id]:
                    new_scene_flag = True
             
            non_key_id_list = self.scene_non_kf_dict[scene_name][kf_id]
            non_kf_id = random.choice(non_key_id_list,)

            data_k1 = self.get_item_data(kf_id, scene_name)
            data_gt = self.get_train_data(non_kf_id, scene_name)
            # key frame
            intrinsic_depth = torch.FloatTensor(self.intrinsic_depth_dict[scene_name])
            img_path_k1 = os.path.join(self.rgb_image_path, scene_name, 'color',
                                       "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, skip_flag1 = self._get_RT_lookat(lookat=data_k1['look_at'],
                                                                    cam_pos=data_k1['campos'],
                                                                    up=torch.Tensor(data_k1['up']), input=valid_line)
            reprojected_c2w_k1, _ = self.load_c2w_mat(img_id=kf_id, scnen_name=scene_name)
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1,
                                                                                intrinsic_matrix=data_k1['intrinsic'],
                                                                                camrot=rot_c2w_k1)
            # camera_1 = FoVPerspectiveCameras(R=rot_c2w_1, T=campos_1, fov=90.0, znear=self.cfg.DATA.NEAR_PLANE, zfar=self.cfg.DATA.FAR_PLANE)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0 / 3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1, 'scene': scene_name, 'c2w': reprojected_c2w_k1, 'intrinsic_depth':intrinsic_depth,
            }
            # train frame
            img_path_item = os.path.join(self.rgb_image_path, scene_name, 'color',
                                         "{}.jpg".format(data_gt['rgb_img_name'].split('.')[0]))
            input_img_item, img_item = self.read_rgb_image(img_path_item)
            rot_c2w_item, campos_item, skip_flag_gt = self._get_RT_lookat(lookat=data_gt['look_at'],
                                                                          cam_pos=data_gt['campos'],
                                                                          up=torch.Tensor(data_gt['up']),
                                                                          input=valid_line)
            reprojected_c2w_k2, _ = self.load_c2w_mat(img_id=non_kf_id, scnen_name=scene_name)
            gt_item, pixelcoords_item, raydir_item, pxpy_item = self._obtain_train_info(img_item,
                                                                                        intrinsic_matrix=data_gt[
                                                                                            'intrinsic'],
                                                                                        camrot=rot_c2w_item)
            cam_para_item = {'R': rot_c2w_item, 'T': campos_item, 'fov': data_gt['fov'], 'aspect_ratio': 4.0 / 3.0}
            output_data_item = {
                'gt': gt_item, 'rgb': input_img_item, 'sampled_pidx': pxpy_item,
                'sampled_ray': raydir_item,
                'camera': cam_para_item,
                'rgb_img_name': data_gt['rgb_img_name'],
                'c2w': reprojected_c2w_k2,
                'intrinsic_depth': intrinsic_depth,
            }
            skip_flag_f = skip_flag_gt or skip_flag1
            if skip_flag_f:
                print(valid_line)
             
            if new_scene_flag:
                return {0: output_data_k1, 1: dict(), 2: output_data_item, 'skip_flag': skip_flag_f,
                        'need_point_recon': new_scene_flag, 'new_scene_flag':self.hard_new_scene[id]}
            else:
                return {0: dict(), 1: output_data_k1, 2: output_data_item, 'skip_flag': skip_flag_f,
                        'need_point_recon': new_scene_flag, 'new_scene_flag':self.hard_new_scene[id]}
        elif self.get_item_type == 3:
            raise NotImplementedError()

        else:
            raise RuntimeError("Get_item_type is wrong.")

    def get_item_data(self, id, scene_name=None):
        assert scene_name is not None
        pkl_path = os.path.join(self.scannet_pkl_path,
                                scene_name,
                                "{}.pkl".format(str(id).zfill(4)))
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        data['look_at'] = np.array(data['look_at'])
        data['look_at'] = data['look_at'][None, ]
        data['campos'] = np.array(data['campos'])
        data['campos'] = data['campos'][None,]
        data['up'] = np.array(data['up'])[None, ]
        data['intrinsic'] = self.intrinsic_dict[scene_name]
        data['weights'] = data['weights'].reshape(-1, 1)
        data['id'] = id
        data['scene'] = scene_name
        data['fov'] = self.fov_dict[scene_name]
        return data

@DATASET_REGISTRY.register()
class Scannet_S1_multi_scenes_valid_index_scene_based_subset(Scannet_S1_multi_scenes_valid_index_scene_based):
    def __init__(self, config, **kwargs):
        self.max_subset_num = config.DATA.MAX_SUBSET_NUM
        super(Scannet_S1_multi_scenes_valid_index_scene_based, self).__init__(config, **kwargs)

    def load_meta_data(self):
        # self.scene_name_list = []
        self.intrinsic_dict = OrderedDict()
        self.intrinsic_depth_dict = OrderedDict()
        self.fov_dict = OrderedDict()
        #
        with open(self.cfg.DATA.SCANNET_VALID_INPUT_LIST, 'rb') as f:
            self.valid_input_list = pickle.load(f)
        #
        self.kf_order_list = []
        self.scene_kf_dict = OrderedDict()
        self.scene_non_kf_dict = OrderedDict()
        #
        scene_sub_dict = dict()  # dict: {name: []}
        name_list = []
         
        # parsing the valid_input_list
        for index in range(len(self.valid_input_list)):
            line = self.valid_input_list[index]
            words = line.split('__')
            scene_name = words[0]
            valid_id = int(words[1])
            # init the dict:
            if scene_name not in scene_sub_dict:
                scene_sub_dict[scene_name] = [[],]
                self.scene_non_kf_dict[scene_name] = dict()
                name_list.append("{}__{}".format(scene_name, 0))
                temp_non_key = []
                # init corresponding dict
                self.intrinsic_dict[scene_name] = self._load_rgb_intrinsics(scene_name)
                self.intrinsic_depth_dict[scene_name] = self._load_depth_intrinsics(scene_name)
                self.fov_dict[scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[scene_name][1, 1])
            # insert the key frame
            # temp_non_key.append(valid_id)
            # whether key frame
            if valid_id % 10 == 0:
                # whether to create the new subset
                 

                if len(scene_sub_dict[scene_name][-1]) > self.max_subset_num:
                    # new subset
                    scene_sub_dict[scene_name].append([valid_id])
                    temp_non_key = [valid_id]
                    name_list.append("{}__{}".format(scene_name, len(scene_sub_dict[scene_name]) - 1))

                elif len(scene_sub_dict[scene_name][-1]) >= 1 and (valid_id - scene_sub_dict[scene_name][-1][-1]) > 50:
                    if len(scene_sub_dict[scene_name][-1]) == 1:
                        scene_sub_dict[scene_name][-1].append(scene_sub_dict[scene_name][-1][0])
                    scene_sub_dict[scene_name].append([valid_id])
                    temp_non_key = [valid_id]
                    name_list.append("{}__{}".format(scene_name, len(scene_sub_dict[scene_name]) - 1))
                else:
                    # append to the current subset
                    temp_non_key.append(valid_id)
                    scene_sub_dict[scene_name][-1].append(valid_id)
                 
                self.scene_non_kf_dict[scene_name][valid_id] = copy.deepcopy(temp_non_key)
                # temp_non_key = [valid_id]
            else:
                temp_non_key.append(valid_id)
         
        # shuffle the kf_order_list
        random.Random(3).shuffle(name_list)
        self.hard_new_scene = []
        for name_item in name_list:
            words = name_item.split("__")
            scene_name_item, subset_index = words[0], int(words[1])
            for index, key_id in enumerate(scene_sub_dict[scene_name_item][subset_index]):
                self.kf_order_list.append("{}__{}".format(scene_name_item, key_id))
                if index == 0:
                    self.hard_new_scene.append(True)
                else:
                    self.hard_new_scene.append(False)

         
        total_num = len(self.kf_order_list)
        # assert total_num == len(self.kf_order_list)
        print("Total num of training: ", total_num)
        # stage store scene name
        self.scene_name_now = None
        #

@DATASET_REGISTRY.register()
class Scannet_S1_multi_scenes_valid_index_scene_based_subset_v1(Scannet_S1_multi_scenes_valid_index_scene_based):
    def __init__(self, config, **kwargs):
        self.max_subset_num = config.DATA.MAX_SUBSET_NUM
        super(Scannet_S1_multi_scenes_valid_index_scene_based, self).__init__(config, **kwargs)

    def load_meta_data(self):
        self.intrinsic_dict = OrderedDict()
        self.intrinsic_depth_dict = OrderedDict()
        self.fov_dict = OrderedDict()
        #
        with open(self.cfg.DATA.SCANNET_VALID_INPUT_LIST, 'rb') as f:
            self.valid_input_list = pickle.load(f)
        #
        self.kf_order_list = []
        self.scene_kf_dict = OrderedDict()
        self.scene_non_kf_dict = OrderedDict()
        #
        scene_sub_dict = dict()  # dict: {name: []}
        name_list = []
         
        # parsing the valid_input_list
        for index in range(len(self.valid_input_list)):
            line = self.valid_input_list[index]
             
            words = line.split('__')
            scene_name = words[0]
            valid_id = int(words[1])
            # init the dict:
            if scene_name not in scene_sub_dict:
                scene_sub_dict[scene_name] = [[],]
                self.scene_non_kf_dict[scene_name] = dict()
                name_list.append("{}__{}".format(scene_name, 0))
                temp_non_key = []
                # init corresponding dict
                self.intrinsic_dict[scene_name] = self._load_rgb_intrinsics(scene_name)
                self.intrinsic_depth_dict[scene_name] = self._load_depth_intrinsics(scene_name)
                self.fov_dict[scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[scene_name][1, 1])
            # insert the key frame
            # temp_non_key.append(valid_id)
            # whether key frame
            if index % 2 == 0:
                # whether to create the new subset

                if len(scene_sub_dict[scene_name][-1]) > self.max_subset_num:
                    # new subset
                    scene_sub_dict[scene_name].append([valid_id])
                    temp_non_key = [valid_id]
                    name_list.append("{}__{}".format(scene_name, len(scene_sub_dict[scene_name]) - 1))

                elif len(scene_sub_dict[scene_name][-1]) >= 1 and (valid_id - scene_sub_dict[scene_name][-1][-1]) > 50:
                    if len(scene_sub_dict[scene_name][-1]) == 1:
                        scene_sub_dict[scene_name][-1].append(scene_sub_dict[scene_name][-1][0])
                    scene_sub_dict[scene_name].append([valid_id])
                    temp_non_key = [valid_id]
                    name_list.append("{}__{}".format(scene_name, len(scene_sub_dict[scene_name]) - 1))
                else:
                    # append to the current subset
                    temp_non_key.append(valid_id)
                    scene_sub_dict[scene_name][-1].append(valid_id)
                 
                self.scene_non_kf_dict[scene_name][valid_id] = copy.deepcopy(temp_non_key)
                # temp_non_key = [valid_id]
            else:
                temp_non_key.append(valid_id)
         
        # shuffle the kf_order_list
        random.Random(3).shuffle(name_list)
        self.hard_new_scene = []
        for name_item in name_list:
            words = name_item.split("__")
            scene_name_item, subset_index = words[0], int(words[1])
            for index, key_id in enumerate(scene_sub_dict[scene_name_item][subset_index]):
                self.kf_order_list.append("{}__{}".format(scene_name_item, key_id))
                if index == 0:
                    self.hard_new_scene.append(True)
                else:
                    self.hard_new_scene.append(False)

         
        total_num = len(self.kf_order_list)
        # assert total_num == len(self.kf_order_list)
        print("Total num of training: ", total_num)
        # stage store scene name
        self.scene_name_now = None
        #


@DATASET_REGISTRY.register()
class Scannet_S1_scenes_test(Scannet_S1_multi_scenes):
    def __init__(self, config, **kwargs):
        self.cfg = config
        self.scene_name = kwargs.get('scene_name', self.cfg.DATA.SCENE_NAME)
        super(Scannet_S1_scenes_test, self).__init__(config, **kwargs)
        if self.get_item_type == 1:
            print("Only return training set!")
        elif self.get_item_type == 2:
            print("Only return test set!")
        elif self.get_item_type == 3:
            print("Return all set!")

    def load_meta_data(self):
        train_img_name, test_img_name = self._load_txt_(self.scene_name)
        self.train_img_name = train_img_name
        self.test_img_name = test_img_name
        print("Test img name list: ", self.test_img_name)

        self.intrinsic_dict = OrderedDict()
        self.intrinsic_dict[self.scene_name] = self._load_rgb_intrinsics(self.scene_name)
        self.fov_dict = OrderedDict()
        self.fov_dict[self.scene_name] = self._get_fov_h(self.img_wh[1], self.intrinsic_dict[self.scene_name][1, 1])

    def _load_txt_(self, scene_name):
        with open(os.path.join(self.cfg.DATA.SCANNET_TXT_PATH, scene_name, 'train.txt')) as f:
            train_lines = f.readlines()
        with open(os.path.join(self.cfg.DATA.SCANNET_TXT_PATH, scene_name, 'test.txt')) as f:
            test_lines = f.readlines()
        return train_lines, test_lines

    def __getitem__(self, id):
        # id = self.cfg.given_id
        img_name, train_flag = self._parsing_id(id)
        # print("Image Name: ", img_name)
        #
        valid_input = img_name
        #
        frame_id = int(img_name.split('.')[0])
        if self.get_item_type == 1:
            # For train data
            data_k1 = self.get_item_data(frame_id, self.scene_name)
            #
            img_path_k1 = os.path.join(self.rgb_image_path, self.scene_name, 'color',
                                       "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, skip_flag1 = self._get_RT_lookat(lookat=data_k1['look_at'], cam_pos=data_k1['campos'],
                                                                    up=torch.Tensor(data_k1['up']), input=valid_input)
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(self.scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1,
                                                                                intrinsic_matrix=data_k1['intrinsic'],
                                                                                camrot=rot_c2w_k1)
            # camera_1 = FoVPerspectiveCameras(R=rot_c2w_1, T=campos_1, fov=90.0, znear=self.cfg.DATA.NEAR_PLANE, zfar=self.cfg.DATA.FAR_PLANE)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0 / 3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1,
                # New for test
                'train_set_flag': train_flag, 'gt_rgb_img': img_k1, 'skip_flag': skip_flag1
            }
            output = {0: output_data_k1}
        elif self.get_item_type == 2:

            # For train data
            data_k1 = self.get_train_data(frame_id, self.scene_name)
            #
            img_path_k1 = os.path.join(self.rgb_image_path, self.scene_name, 'color',
                                       "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, skip_flag1 = self._get_RT_lookat(lookat=data_k1['look_at'],
                                                                    cam_pos=data_k1['campos'],
                                                                    up=torch.Tensor(data_k1['up']), input=valid_input)
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(self.scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1,
                                                                                intrinsic_matrix=data_k1['intrinsic'],
                                                                                camrot=rot_c2w_k1)
            # camera_1 = FoVPerspectiveCameras(R=rot_c2w_1, T=campos_1, fov=90.0, znear=self.cfg.DATA.NEAR_PLANE, zfar=self.cfg.DATA.FAR_PLANE)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0 / 3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1,
                # New for test
                'train_set_flag': train_flag, 'gt_rgb_img': img_k1, 'skip_flag': skip_flag1
            }
            output = {0: output_data_k1}

        else:
            # For train data
            data_k1 = self.get_train_data(frame_id, self.scene_name)
            #
            img_path_k1 = os.path.join(self.rgb_image_path, self.scene_name, 'color',
                                       "{}.jpg".format(data_k1['rgb_img_name'].split('.')[0]))
            input_img_k1, img_k1 = self.read_rgb_image(img_path_k1)

            rot_c2w_k1, campos_k1, skip_flag1 = self._get_RT_lookat(lookat=data_k1['look_at'],
                                                                    cam_pos=data_k1['campos'],
                                                                    up=torch.Tensor(data_k1['up']), input=valid_input)
            if torch.isnan(campos_k1[0]):
                print("scene: {}; id: {}".format(self.scene_name, data_k1['id']))
            gt_k1, pixelcoords_k1, raydir_k1, pxpy_k1 = self._obtain_train_info(img_k1,
                                                                                intrinsic_matrix=data_k1['intrinsic'],
                                                                                camrot=rot_c2w_k1)
            # camera_1 = FoVPerspectiveCameras(R=rot_c2w_1, T=campos_1, fov=90.0, znear=self.cfg.DATA.NEAR_PLANE, zfar=self.cfg.DATA.FAR_PLANE)
            cam_para_k1 = {'R': rot_c2w_k1, 'T': campos_k1, 'fov': data_k1['fov'], 'aspect_ratio': 4.0 / 3.0}
            output_data_k1 = {
                'data': data_k1, 'gt': gt_k1, 'rgb': input_img_k1, 'sampled_pidx': pxpy_k1, 'sampled_ray': raydir_k1,
                'camera': cam_para_k1,
                # New for test
                'train_set_flag': train_flag, 'gt_rgb_img': img_k1, 'skip_flag': skip_flag1
            }
            output = {0: output_data_k1}
            # raise NotImplementedError()
        return output

    def __len__(self):
        if self.get_item_type == 1:
            return len(self.train_img_name)
        elif self.get_item_type == 2:
            return len(self.test_img_name)
        num_list = len(self.train_img_name) + len(self.test_img_name)
        return num_list

    def _parsing_id(self, id):
        if self.get_item_type == 2:
            return self.test_img_name[id], False
        if self.get_item_type == 1:
            return self.train_img_name[id], True
        if self.get_item_type == 3:
             
            if id < len(self.train_img_name):
                # load the train data
                img_name = self.train_img_name[id]
                train_flag = True
            else:
                img_name = self.test_img_name[id - len(self.train_img_name)]
                train_flag = False
            return img_name, train_flag

    def get_item_data(self, id, scene_name=None):
        assert scene_name is not None
        pkl_path = os.path.join(self.scannet_pkl_path,
                                scene_name,
                                "{}.pkl".format(str(id).zfill(4)))
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        data['pcd_dirs'] = data['pcd_dirs'] / (np.linalg.norm(data['pcd_dirs'], axis=-1, keepdims=True) + 1e-6)
        data['look_at'] = np.array(data['look_at'])
        data['look_at'] = data['look_at'][None, ]
        data['campos'] = np.array(data['campos'])
        data['campos'] = data['campos'][None,]
        data['up'] = np.array(data['up'])[None, ]
        # intrinsic_list = data['intrinsic']
        data['intrinsic'] = self.intrinsic_dict[scene_name]
        data['weights'] = data['weights'].reshape(-1, 1)
        data['id'] = id
        data['scene'] = scene_name
        data['fov'] = self.fov_dict[scene_name]
        return data




# -----------------------------------------------------------------------------------------------------------
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