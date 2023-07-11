model_weight="./checkpoints/scannet_pretrained_0beginGRU_subset_4gpu_0708_t2/100000__checkpoints.pth"
log_task_name="init_localgru"

cuda_id=0
#extra_name='debug_local_0begin'
config_folder_name="init_localgru"
scene_num="0553"
data_scene_name=scene${scene_num}_00

CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
 --config-file ./yaml_files/${config_folder_name}/scene${scene_num}.yaml \
 --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}


load_init_point_root=./checkpoints/${log_task_name}
#extra_name="debug_local"
#log_task_name="debug_gru_local"
config_root=${config_folder_name}

CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
 --config-file ./yaml_files/${config_root}/scene${scene_num}.yaml \
 --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_${scene_num} MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
 DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
 MODEL.RASTERIZER.load_init_point True TXT_NAME "0begin_"+${scene_num}

#data_scene_name="scene0079_00"
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0079.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
# --config-file ./yaml_files/${config_root}/scene0079.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0079 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------
#data_scene_name="scene0158_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0158.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
# --config-file ./yaml_files/${config_root}/scene0158.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0158 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------------------------------------------------------------
#data_scene_name="scene0316_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0316.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
# --config-file ./yaml_files/${config_root}/scene0316.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0316 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------------------------------------------------------------
#data_scene_name="scene0521_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0521.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
# --config-file ./yaml_files/${config_root}/scene0521.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0521 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------------------------------------------------------------
#data_scene_name="scene0553_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0553.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
#  --config-file ./yaml_files/${config_root}/scene0553.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0553 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------------------------------------------------------------
#data_scene_name="scene0616_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0616.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
# --config-file ./yaml_files/${config_root}/scene0616.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0616 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
#
## ---------------------------------------------------------------------------------------------------
#data_scene_name="scene0653_00"
#CUDA_VISIBLE_DEVICES=${cuda_id} python fuse_all_points.py \
# --config-file ./yaml_files/${config_folder_name}/scene0653.yaml \
# --num-gpus 1 MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name} DATA.SCENE_NAME ${data_scene_name}  depth_merge_thres 0.1 \
#   | tee -a temp_out_txt/init_${extra_name}_s0000.txt
#
#CUDA_VISIBLE_DEVICES=${cuda_id} python rasterize_from_global.py \
#  --config-file ./yaml_files/${config_root}/scene0653.yaml \
# --num-gpus 1  MODEL.WEIGHTS ${model_weight} LOG.TASK_NAME ${log_task_name}_0653 MODEL.RASTERIZER.load_init_point_root ${load_init_point_root} \
# DATA.GET_ITEM_TYPE 2 DATA.SCENE_NAME ${data_scene_name}  \
# MODEL.RASTERIZER.load_init_point True  | tee temp_out_txt/pre_test_${extra_name}_s0000.txt
