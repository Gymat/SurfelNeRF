import os
import argparse

def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/x/",
        metavar="FILE",
        help="path to scannet dataset")

    return parser

def soft_1(args):
    # This func selects the target 100 scenes to target folder
    tar_path = "./datasets/export_Scannet_train"  # target soft-link folder
    os.makedirs(tar_path, exist_ok=True)
    selected_scene_name_lists = ['scene0525_00', 'scene0286_02', 'scene0107_00', 'scene0101_01', 'scene0347_01', 'scene0113_01', 'scene0163_00', 'scene0680_00', 'scene0501_01', 'scene0204_02', 'scene0691_01', 'scene0676_01', 'scene0287_00', 'scene0138_00', 'scene0365_01', 'scene0005_00', 'scene0320_02', 'scene0581_02', 'scene0067_01', 'scene0223_00', 'scene0178_00', 'scene0200_01', 'scene0349_01', 'scene0444_01', 'scene0564_00', 'scene0156_00', 'scene0323_00', 'scene0160_01', 'scene0085_01', 'scene0397_01', 'scene0043_01', 'scene0020_00', 'scene0597_00', 'scene0214_01', 'scene0212_02', 'scene0236_00', 'scene0311_00', 'scene0529_00', 'scene0059_01', 'scene0571_00', 'scene0692_04', 'scene0452_02', 'scene0055_01', 'scene0451_02', 'scene0045_00', 'scene0182_01', 'scene0092_03', 'scene0058_01', 'scene0306_00', 'scene0332_00', 'scene0492_00', 'scene0267_00', 'scene0348_01', 'scene0188_00', 'scene0297_00', 'scene0400_01', 'scene0369_01', 'scene0429_00', 'scene0410_01', 'scene0662_01', 'scene0303_01', 'scene0505_00', 'scene0376_00', 'scene0125_00', 'scene0399_01', 'scene0510_01', 'scene0040_01', 'scene0586_00', 'scene0536_01', 'scene0225_00', 'scene0118_02', 'scene0548_02', 'scene0051_01', 'scene0298_00', 'scene0147_01', 'scene0029_02', 'scene0620_00', 'scene0531_00', 'scene0408_00', 'scene0642_01', 'scene0268_01', 'scene0286_01', 'scene0148_00', 'scene0659_00', 'scene0293_01', 'scene0108_00', 'scene0514_00', 'scene0449_02', 'scene0603_01', 'scene0165_01', 'scene0254_00', 'scene0313_02', 'scene0292_00', 'scene0281_00', 'scene0515_01', 'scene0380_01', 'scene0214_00', 'scene0419_00', 'scene0181_00', 'scene0280_00', 'scene0211_01', 'scene0468_02', 'scene0318_00', 'scene0380_02', 'scene0677_01', 'scene0350_01']
    # refer scenes name
    data_path = args.data_path
    folder_list = ['color', 'depth', 'intrinsic', 'pose']
    for scene_name in selected_scene_name_lists:
        for sub_folder in folder_list:
            src = os.path.join(data_path, scene_name, 'exported', sub_folder)
            os.makedirs(os.path.join(tar_path, scene_name), exist_ok=True)
            dst = os.path.join(tar_path, scene_name, sub_folder)
            cm_line = "ln -s {} {}".format(src, dst)
            os.system(cm_line)


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    soft_1(args)

