# """
# This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
# Example usage:
# ```
# python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
# ```
# Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
# 1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
# 2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
# 3. 3DPW ```--dataset=3dpw```
# 4. LSP ```--dataset=lsp```
# 5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
# """

# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# import cv2
# import os
# import argparse
# import json
# from collections import namedtuple
# from tqdm import tqdm
# import torchgeometry as tgm
# import sys

# import config
# import constants
# from models import hmr, SMPL
# from datasets import BaseDataset
# from utils.imutils import uncrop
# from utils.pose_utils import reconstruction_error
# from utils.part_utils import PartRenderer
# from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
# from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

# # Define command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
# parser.add_argument('--dataset', default='slp', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp','coco','slp'], help='Choose evaluation dataset')
# parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
# parser.add_argument('--batch_size', default=32, help='Batch size for testing')
# parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
# parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
# parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')


# def untranskey(kp, center, scale, inverse, r=0, f=0):
#     scaleRGB = 256/1024
#     """'Undo' the image cropping/resizing.
#     This function is used when evaluating mask/part segmentation.
#     """
#     nparts = kp.shape[0]
#     #print(nparts)
#     for i in range(nparts):
 
#         kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
#                                   [constants.IMG_RES, constants.IMG_RES],invert=inverse,rot=r)
#     kp = kp/scaleRGB
#     kp = kp.astype('float32')
#     return kp



# def run_evaluation(model, dataset_name, dataset, result_file,
#                    batch_size=1, img_res=224, 
#                    num_workers=32, shuffle=False, log_freq=50):
#     """Run evaluation on the datasets and metrics we report in the paper. """

#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#     # Transfer model to the GPU
#     model.to(device)

#     # Load SMPL model
#     smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
#                         create_transl=False).to(device)
#     smpl_male = SMPL(config.SMPL_MODEL_DIR,
#                      gender='male',
#                      create_transl=False).to(device)
#     smpl_female = SMPL(config.SMPL_MODEL_DIR,
#                        gender='female',
#                        create_transl=False).to(device)
    
#     renderer = PartRenderer()
    
#     # Regressor for H36m joints
#     #J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
#     save_results = result_file is not None
#     # Disable shuffling if you want to save the results
#     if save_results:
#         shuffle=False
#     # Create dataloader for the dataset
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    
#     # Pose metrics
#     # MPJPE and Reconstruction error for the non-parametric and parametric shapes
#     mpjpe = np.zeros(len(dataset))
#     print('dataset length:',len(dataset))
#     recon_err = np.zeros(len(dataset))
#     mpjpe_smpl = np.zeros(len(dataset))
#     recon_err_smpl = np.zeros(len(dataset))

#     # Shape metrics
#     # Mean per-vertex error
#     shape_err = np.zeros(len(dataset))
#     shape_err_smpl = np.zeros(len(dataset))

#     # Mask and part metrics
#     # Accuracy
#     accuracy = 0.
#     parts_accuracy = 0.
#     # True positive, false positive and false negative
#     tp = np.zeros((2,1))
#     fp = np.zeros((2,1))
#     fn = np.zeros((2,1))
#     parts_tp = np.zeros((7,1))
#     parts_fp = np.zeros((7,1))
#     parts_fn = np.zeros((7,1))
#     # Pixel count accumulators
#     pixel_count = 0
#     parts_pixel_count = 0

#     # Store SMPL parameters
#     smpl_pose = np.zeros((len(dataset), 72))
#     smpl_betas = np.zeros((len(dataset), 10))
#     smpl_camera = np.zeros((len(dataset), 3))
#     pred_joints = np.zeros((len(dataset), 17, 3))

#     eval_pose = False
#     eval_masks = False
#     eval_parts = False
#     # Choose appropriate evaluation for each dataset
#     if dataset_name == 'coco' or dataset_name == 'slp':
#         eval_pose = True
#     # Iterate over the entire dataset
#     for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
#         # Get ground truth annotations from the batch
#         gt_pose = batch['pose'].to(device)
#         gt_betas = batch['betas'].to(device)
#         gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
#         images = batch['img'].to(device)
#         #images_depth = batch['img_depth'].to(device)
#         # images_ir = batch['img_ir'].to(device)
#         # images_pm = batch['img_pm'].to(device)
#         gender = batch['gender'].to(device)
#         center = batch['center'].to(device)
#         scale = batch['scale'].to(device)
#         curr_batch_size = images.shape[0]
        
#         with torch.no_grad():
#             pred_rotmat, pred_betas, pred_camera = model([images])
#             pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
#             pred_vertices = pred_output.vertices

#         if save_results:
#             rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
#             rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
#             pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
#             smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
#             smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
#             smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
#         # 3D pose evaluation
#         if eval_pose:
#             pred_joints = pred_output.joints
#             pred_cam_t = torch.stack([pred_camera[:,1],
#                                   pred_camera[:,2],
#                                   2*5000./(224 * pred_camera[:,0] +1e-9)],dim=-1)
#             camera_center = torch.zeros(batch_size, 2, device=torch.device('cuda'))

            
#             gt_keypoints_2d = batch['keypoints'][:,:,:2].cuda()
#             pred_keypoints_2d = perspective_projection(pred_joints,
#                                                    rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(0).expand(batch_size, -1, -1),
#                                                    translation=pred_cam_t,
#                                                    focal_length=5000.,
#                                                    camera_center=camera_center)
           
#             #pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)  
#             center = center.cpu().numpy()
#             scale = scale.cpu().numpy()

#             gt_keypoints_2d = 112.*gt_keypoints_2d
#             gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
#             gt_keypoints = gt_keypoints_2d[:,25:39,:]
#             gt_keypoints_2d = gt_keypoints+112 
#             temp = np.zeros((gt_keypoints_2d.shape[0],14,2))
#             for i in range(gt_keypoints_2d.shape[0]):
#                 temp[i,:,:] = untranskey(gt_keypoints_2d[i,:,:], center[i], scale[i], inverse=1, r=0, f=0)
#             gt_keypoints_2d = torch.tensor(temp)

#             pred_keypoints_2d = pred_keypoints_2d.cpu().numpy()
#             pred_keypoints_2d = pred_keypoints_2d[:,25:39,:]
#             pred_keypoints_2d+=112
#             for i in range(pred_keypoints_2d.shape[0]):
#                 temp[i,:,:] = untranskey(pred_keypoints_2d[i,:,:], center[i], scale[i], inverse=1, r=0, f=0)
#             pred_keypoints_2d = torch.tensor(temp)                                

#             #Absolute error (MPJPE)
#             error = torch.sqrt(((pred_keypoints_2d - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
#             error = torch.sqrt(((pred_keypoints_2d - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
#             mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

#             # Reconstuction_error
#             r_error = reconstruction_error(pred_keypoints_2d.cpu().numpy(), gt_keypoints_2d.cpu().numpy(), reduction=None)
#             recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

#         # Print intermediate results during evaluation
#         #print(step,log_freq)
#         if step % log_freq == log_freq - 1:
#             if eval_pose:
#                 print('MPJPE: ' + str(mpjpe[:step * batch_size].mean()))
#                 print('Reconstruction Error: ' + str(recon_err[:step * batch_size].mean()))
#                 print()

#     # Save reconstructions to a file for further processing
#     if save_results:
#         np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
#     # Print final results during evaluation
#     print('*** Final Results ***')
#     print()
#     if eval_pose:
#         print('MPJPE: ' + str(mpjpe.mean()))
#         print('Reconstruction Error: ' + str(recon_err.mean()))
#         print()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     model = hmr(config.SMPL_MEAN_PARAMS)
#     checkpoint = torch.load(args.checkpoint)
#     model.load_state_dict(checkpoint['model'], strict=False)
#     model.eval()

#     # Setup evaluation dataset
#     dataset = BaseDataset(None, args.dataset, is_train=False)
#     # Run evaluation
#     run_evaluation(model, args.dataset, dataset, args.result_file,
#                    batch_size=args.batch_size,
#                    shuffle=args.shuffle,
#                    log_freq=args.log_freq)
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
# from utils.part_utils import PartRenderer
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['slp','h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=1, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')


def untranskey(kp, center, scale, inverse, r=0, f=0):
    scaleRGB = 256 / 1024
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    nparts = kp.shape[0]
    # print(nparts)
    for i in range(nparts):
        kp[i, 0:2] = transform(kp[i, 0:2] + 1, center, scale,
                               [constants.IMG_RES, constants.IMG_RES], invert=inverse, rot=r)
    kp = kp / scaleRGB
    kp = kp.astype('float32')
    return kp
def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=1, img_res=224,
                   num_workers=1, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # renderer = PartRenderer()

    # Regressor for H36m joints
    # J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        #depths = batch['depth'].to(device)
        gender = batch['gender'].to(device)
        center = batch['center'].to(device)
        scale = batch['scale'].to(device)
        curr_batch_size = images.shape[0]

        with torch.no_grad():
            #pred_rotmat_1, pred_betas_1, pred_camera_1,\
            #pred_rotmat_2, pred_betas_2, pred_camera_2,\
            pred_rotmat, pred_betas, pred_camera = model([images])
            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        if save_results:
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()


        # 2D Absolute error (MPJPE)
        pred_joints = pred_output.joints
        pred_cam_t = torch.stack([pred_camera[:, 1],
                                  pred_camera[:, 2],
                                  2 * 5000. / (224 * pred_camera[:, 0] + 1e-9)], dim=-1)
        camera_center = torch.zeros(batch_size, 2, device=torch.device('cuda'))
        # print('camera_center',camera_center)

        gt_keypoints_2d = batch['keypoints'][:, :, :2].cuda()

        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(
                                                       0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=5000.,
                                                   camera_center=camera_center)

        # pred_keypoints_2d = pred_keypoints_2d / (224 / 2.)
        center = center.cpu().numpy()
        scale = scale.cpu().numpy()

        gt_keypoints_2d = 112. * gt_keypoints_2d
        gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
        gt_keypoints = gt_keypoints_2d[:, 25:39, :]
        gt_keypoints_2d = gt_keypoints + 112
        temp = np.zeros((gt_keypoints_2d.shape[0], 14, 2))
        for i in range(gt_keypoints_2d.shape[0]):
            temp[i, :, :] = untranskey(gt_keypoints_2d[i, :, :], center[i], scale[i], inverse=1, r=0, f=0)
        gt_keypoints_2d = torch.tensor(temp)

        pred_keypoints_2d = pred_keypoints_2d.cpu().numpy()
        pred_keypoints_2d = pred_keypoints_2d[:, 25:39, :]
        pred_keypoints_2d += 112
        for i in range(pred_keypoints_2d.shape[0]):
            temp[i, :, :] = untranskey(pred_keypoints_2d[i, :, :], center[i], scale[i], inverse=1, r=0, f=0)
        pred_keypoints_2d = torch.tensor(temp)

        # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_2d - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        error = torch.sqrt(((pred_keypoints_2d - gt_keypoints_2d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

        # Reconstuction_error
        r_error = reconstruction_error(pred_keypoints_2d.cpu().numpy(), gt_keypoints_2d.cpu().numpy(), reduction=None)
        recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

    print('*** Final Results ***')
    print()
    print('MPJPE: ' + str(mpjpe.mean()))
    print('Reconstruction Error: ' + str(recon_err.mean()))
    print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
