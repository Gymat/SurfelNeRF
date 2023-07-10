from fvcore.common.config import CfgNode

"""
This file defines default options of configurations.
It will be further merged by yaml files and options from
the command-line.
Note that *any* hyper-parameters should be firstly defined
here to enable yaml and command-line configuration.
"""

_C = CfgNode()

_C.DEVICE = "cuda"

_C.DEVICES_ID = [0]

_C.IS_TRAIN = True

_C.DEBUG = False
# For using unreliable GPUs
_C.KEEP_TRAINING = False

_C.depth_merge_thres = 0.1
_C.img_size = 960
_C.img_size_h = 480
_C.img_size_w = 640
_C.pkl_path = "./dataset/example_data/points_info_v2.pkl"
##############
_C.old_load_points = True
_C.TXT_NAME = "hi"

# ---------------------------------------------------------------------------- #
# Common
# ---------------------------------------------------------------------------- #
# Paths for logging and saving
_C.COMMON = CfgNode()
#
_C.COMMON.GPU_MAX_THREADS_PER_BLOCK = 1024
#
_C.COMMON.RESUME_ITER = "best"



# ---------------------------------------------------------------------------- #
# Log Paths
# ---------------------------------------------------------------------------- #
# Paths for logging and saving
_C.LOG = CfgNode()
#
_C.LOG.CHECKPOINT_PATH = 'checkpoints/'
# step for visualizing and saving the images during training.
_C.LOG.VIS_STEP = 2048  #
#
_C.LOG.TASK_NAME = 'exp1'
#
_C.LOG.TENSORBOARD = False
#
_C.LOG.STEP_PRINT_LOSS = 20
#
_C.LOG.STEP_SAVE_MODEL = 10000
# Step for saving points
_C.LOG.SAVE_POINT_EPOCH = 10
#
_C.LOG.SAVE_ITER_STEP = 1000
#
_C.LOG.TIME_LOGGER_PATH = './time_logger.pth'  # note: pcd info also saved in time logger!
# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
# model settings
_C.MODEL = CfgNode()
#
_C.MODEL.NAME = "PointsVolumetricModel"
# Checkpoints
_C.MODEL.WEIGHTS = None
# ------ For the corresponding func --------
#
_C.MODEL.BLEND_FUNC_NAME = "alpha"
#
_C.MODEL.RAY_GEN_NAME = "near_far_linear"
#
_C.MODEL.TONEMAP_FUNC_NAME = "off"
#
_C.MODEL.RENDER_FUNC_NAME = "radiance"

# -------- For the point reconstruction network ----------
_C.MODEL.NEED_POINTCON = True  # only useful during test

# = = = = = = = = = = = = = =
## For the Image encoder backbone
_C.MODEL.BACKBONE = CfgNode()
#
_C.MODEL.BACKBONE.NAME = "FeatureNet"  # ["FeatureNet", "MnasMulti"]
#

# = = = = = = = = = = = = = =
## For the neural points
_C.MODEL.NEURAL_POINTS = CfgNode()
# name
_C.MODEL.NEURAL_POINTS.NAME = "NeuralPoints_Simple"
#
_C.MODEL.NEURAL_POINTS.REQUIRE_GRAD = True
#
_C.MODEL.NEURAL_POINTS.CHANNELS = 32
#
_C.MODEL.NEURAL_POINTS.LOAD_POINTS = False
# load points path
# _C.MODEL.NEURAL_POINTS.LOAD_POINTS_PATH = None
# grad for parameters in neural_points
_C.MODEL.NEURAL_POINTS.POINT_FEAD_GRAD = True
# grad for parameters in neural_points
_C.MODEL.NEURAL_POINTS.POINT_CONF_GRAD = True
# grad for parameters in neural_points
_C.MODEL.NEURAL_POINTS.POINT_DIR_GRAD = True
# grad for parameters in neural_points
_C.MODEL.NEURAL_POINTS.POINT_COLOR_GRAD = True
# whether
_C.MODEL.NEURAL_POINTS.WCOOR_QUERY = True
# ----------------- Magic Number ----------------------
# Magic number
# vox scale is scale that multiply the voxel size to make the voxel larger or smaller
# But we want to change the vox_size only, instead of two parameters.
_C.MODEL.NEURAL_POINTS.VOX_SCALE = "1 1 1"
# scale is the voxel size (m)
_C.MODEL.NEURAL_POINTS.VOX_SIZE = "0.008 0.008 0.004"
#
_C.MODEL.NEURAL_POINTS.KERNEL_SIZE = "3 3 3"
# 'max neural points each group'
_C.MODEL.NEURAL_POINTS.K = 8
# max shading points number each ray
_C.MODEL.NEURAL_POINTS.SR = 24
# max neural points stored each block
_C.MODEL.NEURAL_POINTS.P = 26
#
_C.MODEL.NEURAL_POINTS.MAX_O = 1000000
#
_C.MODEL.NEURAL_POINTS.Z_DEPTH_INTERVAL = 400
#
_C.MODEL.NEURAL_POINTS.RADIUS_LIMIT_SCALE = 4
#
_C.MODEL.NEURAL_POINTS.VOXEL_RES = 800
# Top-k raseterized surfels when comparing surfels
_C.MODEL.NEURAL_POINTS.TOPK_RASTERIZED_SURFELS = 1

# = = = = = = = = = = = = = =
## For the point aggregator
_C.MODEL.AGGREGATOR = CfgNode()
#
_C.MODEL.AGGREGATOR.NAME = "PointAggregator"
#
_C.MODEL.AGGREGATOR.ACT_TYPE = "ReLU"
# ----------------- Magic Number ----------------------
#
_C.MODEL.AGGREGATOR.DIST_DIM = 6
#
_C.MODEL.AGGREGATOR.AGG_AXIS_WEIGHT = "1 1 1"
# color channel num
_C.MODEL.AGGREGATOR.NUM_FEAT_FREQS = 3
# agg intrp order
# interpolate first and feature mlp 0 | feature mlp then interpolate 1 | feature mlp color then interpolate 2
_C.MODEL.AGGREGATOR.AGG_INTRP_ORDER = 2
#
_C.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER1 = 2
#
_C.MODEL.AGGREGATOR.SHADING_OUT_CHANNEL = 256
#
_C.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER2 = 0
#
_C.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER3 = 2
#
_C.MODEL.AGGREGATOR.SHADING_FEATURE_MLP_LAYER3 = 2
#
_C.MODEL.AGGREGATOR.SHADING_ALPHA_MLP_LAYER = 1
#
_C.MODEL.AGGREGATOR.SHADING_COLOR_MLP_LAYER = 4
#
_C.MODEL.AGGREGATOR.POINT_COLOR_MODE = "1"
#
_C.MODEL.AGGREGATOR.POINT_DIR_MODE = "1"
# number of frequency for position encoding if using nerf or mixed mlp decoders
_C.MODEL.AGGREGATOR.NUM_POS_FREQS = 10
# number of frequency for view direction encoding if using nerf decoders
_C.MODEL.AGGREGATOR.NUM_VIEWDIR_FREQS = 4
#
_C.MODEL.AGGREGATOR.APPLY_PNT_MASK = 1
#
_C.MODEL.AGGREGATOR.DIST_XYZ_FREQ = 5
#
_C.MODEL.AGGREGATOR.SPARSE_LOSS_WEIGHT = 0
#
_C.MODEL.AGGREGATOR.SURFEL_POS_ENC_FREQ = 10

# = = = = = = = = = = = = = =
## For the image encoder
_C.MODEL.IMG_ENC = CfgNode()
#
_C.MODEL.IMG_ENC.NORM_ACT = 'InPlaceABN'  # ['GN', 'InPlaceABN']

# ==========================
#
_C.MODEL.RASTERIZER = CfgNode()
#
_C.MODEL.RASTERIZER.NAME = ""
#
_C.MODEL.RASTERIZER.DEPTH_MERGE_THRES = 0.1
#
_C.MODEL.RASTERIZER.POINTS_PER_PIXEL = 8
#
_C.MODEL.RASTERIZER.BIN_SIZE = 0
#
_C.MODEL.RASTERIZER.points_per_local = 10
# num of SR
_C.MODEL.RASTERIZER.num_shading_points = 12
#
_C.MODEL.RASTERIZER.load_init_point = False
#
_C.MODEL.RASTERIZER.load_init_point_path = ""
#
_C.MODEL.RASTERIZER.load_init_point_root = ""
#
_C.MODEL.RASTERIZER.load_init_point_name = "iter{}_{}_points.pth"
#
_C.MODEL.RASTERIZER.SAVE_SURFEL_NAME = "EXP"
#
_C.MODEL.RASTERIZER.load_init_point_iter = "0"
#
_C.MODEL.RASTERIZER.FILTER_NEAR_POINTS = True


# ----------------- For GRU Fusion ----------------------
_C.MODEL.FUSION = CfgNode()
#
_C.MODEL.FUSION.NAME = "GRU2D"
#
_C.MODEL.FUSION.BUILD_FUSION = False
#
_C.MODEL.FUSION.INPUT_DIM = 32
#
_C.MODEL.FUSION.HIDDEN_DIM = 32
#
_C.MODEL.FUSION.WEIGHTS_DIM = 5
#
_C.MODEL.FUSION.USE_GRU_DDP = True

# ----------------- For Depth Estimator ----------------------
_C.MODEL.DEPTH = CfgNode()
#
_C.MODEL.DEPTH.NAME = "FWD_v1"
#
_C.MODEL.DEPTH.BUILD_DEPTH = False
#
_C.MODEL.DEPTH.WEIGHTS = None
#
_C.MODEL.DEPTH.FIX_BN = False

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
# training parameters settings
_C.TRAIN = CfgNode()
#
_C.TRAIN.BATCH_SIZE = 1
# Learning rate for network
_C.TRAIN.LEARNING_RATE = 0.0002  # 0.0005 #0.00015
# Learning rate for points
_C.TRAIN.LEARNING_RATE_POINTS = 0.0002  # Learning rate for point aggregator.
#
_C.TRAIN.LEARNING_RATE_FUSION = 0.0002  # 0.0005 #0.00015
#
_C.TRAIN.LEARNING_RATE_DEPTH = 0.0001  # 0.0005 #0.00015
# Shuffle the dataloader
_C.TRAIN.SHUFFLE_DATA = False
#
_C.TRAIN.PIN_MEMORY = True
#
_C.TRAIN.NUM_WORKERS = 8
# max iteration for one epoch
_C.TRAIN.MAX_ITER_PER_EPOCH = 10000
#
# _C.TRAIN.USE_DATASET_LEN = False
#
_C.TRAIN.OPTIMIZER_IMG_ENCODER = True
# lr schedule type
_C.TRAIN.LR_POLICY = 'iter_exponential_decay'
# lr_decay_exp
_C.TRAIN.LR_DECAY_EXP = 0.1
# lr_decay_iters
_C.TRAIN.LR_DECAY_ITERS = 1000000
# max training step
_C.TRAIN.MAX_ITER = 200000
# whether to reset optimizer
_C.TRAIN.RESET_OPTIM = False
#
_C.TRAIN.TRAIN_POINTS = False
#
_C.TRAIN.FT_FLAG = False
#


# ---------------------------------------------------------------------------- #
# Input Data options
# ---------------------------------------------------------------------------- #
# Data settings
_C.DATA = CfgNode()
#
_C.DATA.DATASET_NAME = "Replica_P1"
#
_C.DATA.EVAL_DATASET_NAME = "ARKit_V1_local_test"
# dataset root of all data
_C.DATA.DATA_DIR = "./data_src"  #
# dataset (scene) name in the dataset root
_C.DATA.SCENE_NAME = "replica_room_0"
#
_C.DATA.POINT_CLOUD_PATH = "pc.pcd"
# whether to norm the ray dir
_C.DATA.RAY_DIR_NORM = False
#
_C.DATA.IMG_WH = [1200, 680]
#
_C.DATA.RANGES = "-10 -10 -10 10 10 10 "
#
_C.DATA.BG_COLOR = "black"
#
_C.DATA.NEAR_PLANE = 0.0
#
_C.DATA.FAR_PLANE = 6.0
# How to sample the pixel for NeRF
#
_C.DATA.RANDOM_SAMPLE_SIZE = 64
#
_C.DATA.KEY_FRAMES_NUM = 3
#
_C.DATA.RANDOM_SAMPLE_WAY = "random"  # ["Patch"]
# the way of initialing the point cloud
_C.DATA.LOAD_POINTS = "depth"  # {"depth", "points"}
#
_C.DATA.INCLUDE_KEY_FRAMES = True
#
_C.DATA.NEED_FIRST_KEY_FRAME = True
# whether need to reset points
_C.DATA.NEED_RESET_POINTS = True
# whether need to reset points
_C.DATA.NEED_REGISTER_POINTS = True
# Fragments
_C.DATA.MAX_SUBSET_NUM = 10

# ---------------------------------- Surfel -------------------------------------------
_C.DATA.RGB_Image_DIR = ""
_C.DATA.Depth_Image_DIR = ""
#
_C.DATA.GET_ITEM_TYPE = 1
#
_C.DATA.KEY_FRAME_RATE = 4
#
_C.DATA.ALL_JSON_PATH = ""
#
_C.DATA.SCANNET_PKL_PATH = ""
#
_C.DATA.SCANNET_VALID_INPUT_LIST = ""
#
_C.DATA.SCANNET_TXT_PATH = ""

# --------------------------------- PolyCam ----------------------------------------
_C.POLYCAM = CfgNode()
# depth confidence thres
_C.POLYCAM.CONFIDENCE_THRES = 2


# =============================
# For the realtime recon P1
# =============================
_C.DATA.PHASE1 = CfgNode()
# The number of training data in the




# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
# LOSS settings
_C.LOSS = CfgNode()
#
_C.LOSS.ZERO_LOSS_ITEMS = "conf_coefficient"
###
## COLOR LOSS items
# whether
_C.LOSS.COLOR_LOSS_ITEMS = "ray_masked_coarse_raycolor"
# weight
_C.LOSS.COLOR_LOSS_WEIGHT = "1.0"
## Depth LOSS items
# whether
_C.LOSS.DEPTH_LOSS_ITEMS = ""
# weight
_C.LOSS.DEPTH_LOSS_WEIGHT = "1.0"
## Sparse zero one LOSS items
# whether
_C.LOSS.ZERO_ONE_LOSS_ITEMS = ""
# weight
_C.LOSS.ZERO_ONE_LOSS_WEIGHT = " 0.0001 "


# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
# LOSS settings
_C.TEST = CfgNode()
# Whether using replica_m_fusion in the test setting. If True, that would replica_m_fusion the local fragments from 0 to local_fragment_id_{now};
# If False, that would use the local_fragment_id right now!
_C.TEST.FUSION = False
#
_C.TEST.TRAJ_FILE = ""
#
_C.TEST.SAVE_INTERMEDIATE_POINTS = False

###########################################################################################
# TEMP
###########################################################################################



def get_config()->CfgNode:
    return _C.clone()
