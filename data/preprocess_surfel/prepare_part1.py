import json
import os
import argparse
from collections import OrderedDict


"""
This for replica only !!!
"""

# This file first prepares the input info saved in a json file


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




def parsing_txt(txt_path):
    txt = open(txt_path, 'r')
    lines = txt.readlines()
    output = OrderedDict()
    for index, line in enumerate(lines):
        if index % 10 == 0:
            imgid = int(line)
            output[imgid] = dict()
        if index % 10 == 2:
            intri_data = strlist_to_floatlist(line)
            output[imgid]['intrinsic'] = intri_data
        if index % 10 == 9:
            pos_look = strlist_to_floatlist(line)
            output[imgid]['campos'] = pos_look[:3]
            output[imgid]['look_at'] = pos_look[3:]
    return output


def main(args):
    input_path = args.input_dir
    img_id_list = []
    # parsing files
    for name in os.listdir(input_path):
        # if '.jpeg' in name:
        #     # THis is an image
        #     words = name.split('_')
        #     img_id = int(words[2][3])
        #     img_id_list.append(img_id)
        if '.txt' in name:
            txt_name = name
    #
    # img_id_list = list(set(img_id_list))
    #
    txt_path = os.path.join(input_path, txt_name)
    txt_context = parsing_txt(txt_path)
    #
    output = OrderedDict()
    #
    for oid in txt_context.keys():
        output[oid] = {
            'rgb_img_name': "{}_{}_pos0.jpeg".format(args.name, str(oid).zfill(4)),
            'depth_img_name': "{}_{}_pos5.jpeg".format(args.name, str(oid).zfill(4)),
        }
        for key in txt_context[oid].keys():
            output[oid][key] = txt_context[oid][key]
    #
    # Serializing json
    json_object = json.dumps(output)

    # Writing to sample.json
    with open(args.output_dict, "w") as outfile:
        outfile.write(json_object)





def strlist_to_floatlist(s):
    num_list = s.strip().split(" ")
    # print(num_list)
    float_list = [float(x) for x in num_list]
    return float_list


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    print(args)
    main(args)