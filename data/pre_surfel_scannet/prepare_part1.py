import json
import os
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


"""
This for scannet only !!!
"""

# This file first prepares the input info saved in a json file,
# which provide:
# dict: { image_id: {
#        'rgb_img_name' ; 'depth_img_name' ; 'intrinsic'; 'campos', 'look_at'
#
# }
# }

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/x/",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--output_dict",
        default="configs/scene0000_01.yaml",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--name",
        default="x",
        metavar="FILE",
        help="path to config file")
    # parser.add_argument(
    #     "--depth_trunc",
    #     default=12,
    #     type=float,
    #     help="path to config file")
    # parser.add_argument(
    #     "--height",
    #     default=960,
    #     type=int,
    #     help="path to config file")
    # parser.add_argument(
    #     "--width",
    #     default=960,
    #     type=int,
    #     help="path to config file")
    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    return parser


def obtain_lookat_up_campos(c2w):
    w2c = np.linalg.inv(c2w)
    campos = c2w[:3, 3]  # Not: cam pos is c2w[:3, 3]
    look_at_dir = w2c[2, :3]
    look_at = campos + look_at_dir
    up = w2c[1, :3]
    return campos, look_at, up


def read_pose_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    c2w_list = []
    for line in lines:
        words = line.strip().split(" ")
        row = [float(x) for x in words]
        c2w_list.append(row)
    c2w = np.array(c2w_list)
    #
    campos, look_at_target, up = obtain_lookat_up_campos(c2w)
    return campos, look_at_target, up


def main(args):
    input_path = args.input_dir
    img_id_list = []
    all_output = OrderedDict()
    # parsing files
    # num_img = len(os.listdir(input_path))
    num_total = len(os.listdir(input_path))
    l_name = os.listdir(input_path)
    os.makedirs(args.output_dict, exist_ok=True)
    l_name.sort()
    sid=0
    for sid in tqdm(range(num_total), desc=l_name[sid]):
        scene_name = l_name[sid]
        output = OrderedDict()
        pose_path = os.path.join(input_path, scene_name, 'pose')
        intrinsic_path = os.path.join(input_path, scene_name, 'intrinsic')
        rgb_path = os.path.join(input_path, scene_name, 'color')
        depth_path = os.path.join(input_path, scene_name, 'depth')
        num_img = len(os.listdir(pose_path))
        for img_id in tqdm(range(num_img), desc=scene_name):
            txt_file_path = os.path.join(pose_path, "{}.txt".format(img_id))
            campos, look_at_target, up = read_pose_file(txt_file_path)
            output[img_id] = {
                'rgb_img_name': "{}.jpg".format(img_id),
                'depth_img_name': "{}.png".format(img_id),
                'campos': campos.tolist(), 'look_at': look_at_target.tolist(), 'up': up.tolist(),
            }
        json_object = json.dumps(output)

        # Writing to sample.json
        with open(os.path.join(args.output_dict, "{}.json".format(scene_name)), "w") as outfile:
            outfile.write(json_object)
        # all_output[scene_name] = output

    #
    # Serializing json
    # json_object = json.dumps(all_output)
    #
    # # Writing to sample.json
    # with open(args.output_dict, "w") as outfile:
    #     outfile.write(json_object)





def strlist_to_floatlist(s):
    num_list = s.strip().split(" ")
    # print(num_list)
    float_list = [float(x) for x in num_list]
    return float_list


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    print(args)
    main(args)