import numpy as np
import os
from tqdm import tqdm
import pickle


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



def main():
    root_path = "/group/30042/public_datasets/ScanNet/export_datasets/"
    print("Check the pose to find the <-inf> pose!!!")
    #
    scene_name_list = list(os.listdir(root_path))
    scene_name_list.sort()
    invalid_dict =  dict()
    # print(scene_name_list)
    for scene_name in tqdm(scene_name_list):
        num_txt = len(os.listdir(os.path.join(root_path, scene_name, 'pose')))
        # print(num_txt)
        for id in range(num_txt):
            if id % 1 == 0:
                txt_path = os.path.join(root_path, scene_name, 'pose',"{}.txt".format(id))
                mat = read_pose_file(txt_path)
                mat = mat.reshape(-1)
                if np.isnan(mat[0]) or np.isinf(mat[0]):
                    print("Scene: {}, ID: {}".format(scene_name, id))
                    if scene_name not in invalid_dict:
                        invalid_dict[scene_name] = [id]
                    else:
                        invalid_dict[scene_name].append(id)
            else:
                continue
    print(invalid_dict.keys())
    ####
    # Build valid training frame
    print("*"*50)
    #
    valid_list = []
    for scene_name in tqdm(scene_name_list):
        num_txt = len(os.listdir(os.path.join(root_path, scene_name, 'pose')))
        if scene_name not in invalid_dict:
            for index in range(num_txt):
                valid_list.append("{}__{}".format(scene_name, index))
        elif scene_name in invalid_dict:
            begin_id = 0
            invalid_ids = invalid_dict[scene_name]
            for i in range(num_txt):
                if i in invalid_ids:
                    continue
                else:
                    valid_list.append("{}__{}".format(scene_name, i))
        else:
            raise RuntimeError()
    #
    with open('./valid_index_scannet_fine.pkl', 'wb') as f:
        pickle.dump(valid_list, f)

if __name__ == '__main__':
    main()