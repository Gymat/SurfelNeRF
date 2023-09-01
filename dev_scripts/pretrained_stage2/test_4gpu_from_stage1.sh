python d_train_ft.py --config-file ./yaml_files/pretrained_stage2/base_0begin_gru.yaml --num-gpus 4 \
 TRAIN.LEARNING_RATE_FUSION 0.003 TRAIN.LEARNING_RATE 0.001 TRAIN.LEARNING_RATE_POINTS 0.001 LOG.TASK_NAME "scannet_pretrained_0beginGRU_4gpu_from_stage1" \
 MODEL.WEIGHTS {MODEL_PATH_FROM_STAGE1}
