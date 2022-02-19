# """
# This file contains definitions of useful data stuctures and the paths
# for the datasets and data files necessary to run the code.
# Things you need to change: *_ROOT that indicate the path to each dataset
# """
# from os.path import join

# H36M_ROOT = ''
# LSP_ROOT = ''
# LSP_ORIGINAL_ROOT = '/workspace/shanshanguo/data/slp_raw/'
# LSPET_ROOT = ''
# MPII_ROOT = ''
# COCO_ROOT = ''
# MPI_INF_3DHP_ROOT = ''
# PW3D_ROOT = ''
# UPI_S1H_ROOT = ''
# SLP_ROOT = '/workspace/shanshanguo/temporary/slp_aline_80_train/'


# # Output folder to save test/train npz files
# #DATASET_NPZ_PATH = 'data/dataset_extras/'
# DATASET_NPZ_PATH = 'data/dataset_extras/'

# # Output folder to store the openpose detections
# # This is requires only in case you want to regenerate 
# # the .npz files with the annotations.

# OPENPOSE_PATH = r'/workspace/shanshanguo/project/openpose-master/output_80_slp/'

# # Path to test/train npz files
# DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
#                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
#                    'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
#                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_val.npz'),
#                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
#                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
#                    'slp': join(DATASET_NPZ_PATH,'slp_22_uncover.npz')
#                   },

#                   {#'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
#                    'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
#                    'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
#                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
#                    'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
#                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
#                    'slp': join(DATASET_NPZ_PATH,'slp_80_train.npz')
#                   }
#                 ]

# DATASET_FOLDERS = {'h36m': H36M_ROOT,
#                    'h36m-p1': H36M_ROOT,
#                    'h36m-p2': H36M_ROOT,
#                    'lsp-orig': LSP_ORIGINAL_ROOT,
#                    'lsp': LSP_ROOT,
#                    'lspet': LSPET_ROOT,
#                    'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
#                    'mpii': MPII_ROOT,
#                    'coco': COCO_ROOT,
#                    '3dpw': PW3D_ROOT,
#                    'upi-s1h': UPI_S1H_ROOT,
#                    'slp': SLP_ROOT
#                 }

# OTHER_MODL = '/workspace/shanshanguo/temporary/slp_aline_80_trainother/'

# CUBE_PARTS_FILE = 'data/cube_parts.npy'
# JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
# JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
# VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
# STATIC_FITS_DIR = 'data/static_fits'
# SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
# SMPL_MODEL_DIR = 'data/smpl'
"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = ''
LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
LSPET_ROOT = ''
MPII_ROOT = ''
COCO_ROOT = ''
MPI_INF_3DHP_ROOT = ''
PW3D_ROOT = ''
UPI_S1H_ROOT = ''
SLP_ROOT = '/workspace/shanshanguo/temporary/slp_aline_22uncover/'# /fan/data/slp/slp_aline_80_train/ slp_aline_22uncover slp_aline_22cover1
# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [
                  {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'slp': join(DATASET_NPZ_PATH,'slp_80_uncover.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'slp': join(DATASET_NPZ_PATH,'slp_80_train.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                   'slp': SLP_ROOT,

                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
