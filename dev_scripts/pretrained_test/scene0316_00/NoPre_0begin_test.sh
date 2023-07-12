model_weight="./checkpoints/pretrained__checkpoints.pth"
log_task_name="init_localgru"

cuda_id=0
#extra_name='debug_local_0begin'
config_folder_name="init_localgru"
scene_num="0316"
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


