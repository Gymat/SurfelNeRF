import json
import os
import pickle
# import cv2
import numpy as np
import pdb
import argparse
import copy
from tqdm import tqdm

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_folder",
        default="/x/",
        metavar="FILE",
        help="path to config file")
    parser.add_argument(
        "--pkl_root_path",
        default="/x/",
        metavar="FILE",
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
    return parser


def main(args):
    print("!!! Make sure to check the direction of <up> !!!")
    json_list = os.listdir(args.json_folder)
    json_list.sort()
    print(json_list)
    #
    #
    for i in tqdm(range(args.begin_id, args.end_id), desc="Totally"):
        json_item_name = json_list[i]
        scene_name = json_item_name.split('.')[0]
        json_path = os.path.join(args.json_folder, json_item_name)
        with open(json_path, 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
        #
        scene_pkl_path = os.path.join(args.pkl_root_path, scene_name)
        pkl_num = len(os.listdir(scene_pkl_path))
        for i in tqdm(range(pkl_num), desc=str(scene_name)):
            idx = 10 * i
            pkl_file_path = os.path.join(scene_pkl_path, "{}.pkl".format(str(idx).zfill(4)))
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f)
            # pdb.set_trace()
            data['up'] = [x*-1 for x in json_object[str(idx)]['up']]
            #
            with open(pkl_file_path, 'wb') as f:
                pickle.dump(data, f)



if __name__ == '__main__':
    print("For ScanNet only!")
    args = setup_argparser().parse_args()
    print(args)
    main(args)