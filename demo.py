# """
# Demo code

# To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

# In summary, we provide 3 different ways to use our demo code and models:
# 1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
# 2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
# 3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

# Example with OpenPose detection .json
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
# ```
# Example with predefined Bounding Box
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
# ```
# Example with cropped and centered image
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
# ```

# Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
# """

# import torch
# from torchvision.transforms import Normalize
# import numpy as np
# import cv2
# import argparse
# import json
# import sys

# from models import hmr, SMPL
# from utils.imutils import crop
# from utils.renderer import Renderer
# from utils.geometry import  perspective_projection
# import config
# import constants
# import os
# from os.path import join

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
# #parser.add_argument('--img', type=str, required=True, help='Path to input image')
# #parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
# #parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
# parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
# skels_name = (
# 	# ('Pelvis', 'Thorax'),
# 	('Thorax', 'Head'),
# 	('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
# 	('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
# 	# ('Pelvis', 'R_Hip'),
# 	('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
# 	# ('Pelvis', 'L_Hip'),
# 	('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
# 	)
# colors = [(204, 204, 0),(51, 153, 51),(51,153,255)]
# joints_name = (
# 	"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
# 	"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
# 	"Neck")  # max std joints, first joint_num_ori will be true labeled
# def bbox_from_gt(npz_file, img_id):
#     part = np.load(npz_file,allow_pickle=True)
#     center = part['center'][img_id]
#     scale= part['scale'][img_id]
#     return center,scale 

# def nameToIdx(name_tuple, joints_name):  # test, tp,
# 	'''
# 	from reference joints_name, change current name list into index form
# 	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
# 	:param joints_name:
# 	:return:
# 	'''
# 	jtNm = joints_name
# 	if type(name_tuple[0]) == tuple:
# 		# Transer name_tuple to idx
# 		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
# 	else:
# 		# direct transfer
# 		return tuple(jtNm.index(tpl) for tpl in name_tuple)
# def vis_keypoints(img_path,keypoints,kps_lines, kp_thresh=0.4, alpha=1):
    
# 	# Perform the drawing on a copy of the image, to allow for blending.
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     kp_mask = np.copy(img)
#     kps = keypoints.T
# 	# Draw the keypoints.
#     for l in range(len(kps_lines)):
#         i1 = kps_lines[l][0]
#         i2 = kps_lines[l][1]
#         p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
#         p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
#         cv2.line(
# 				kp_mask, p1, p2,
# 				color=colors[0], thickness=2, lineType=cv2.LINE_AA)
#         cv2.circle(
# 				kp_mask, p1,
# 				radius=3, color=colors[1], thickness=-1, lineType=cv2.LINE_AA)
#         cv2.circle(
# 				kp_mask, p2,
# 				radius=3, color=colors[2], thickness=-1, lineType=cv2.LINE_AA)

# 	# Blend the keypoints.
#     return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

# def process_image(img_file, bbox_file, npz_file, img_id, input_res=224):
#     """Read image, do preprocessing and possibly crop it according to the bounding box.
#     If there are bounding box annotations, use them to crop the image.
#     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
#     """
#     rescale =1.2
#     normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
#     img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
#     if bbox_file is None:
#         # Assume that the person is centerered in the image
#         center,scale = bbox_from_gt(npz_file, img_id)
#     else:
#         height = img.shape[0]
#         width = img.shape[1]
#         center = np.array([width // 2, height // 2])
#         scale = max(height, width) / 200
#     #scale=rescale*scale
#     imgs = crop(img, center, scale, (input_res, input_res))
#     img = imgs.astype(np.float32) / 255.
#     img = torch.from_numpy(img).permute(2,0,1)
#     norm_img = normalize_img(img.clone())[None]
#     return imgs, img, norm_img

# if __name__ == '__main__':
#     args = parser.parse_args()
    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
#     # Load pretrained model
#     model = hmr(config.SMPL_MEAN_PARAMS).to(device)
#     checkpoint = torch.load(args.checkpoint,map_location='cpu')
#     model.load_state_dict(checkpoint['model'], strict=False)

#     # Load SMPL model
#     smpl = SMPL(config.SMPL_MODEL_DIR,
#                 batch_size=1,
#                 create_transl=False).to(device)
#     model.eval()

#     # Setup renderer for visualization
#     renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

#     # Preprocess input image and generate predictions
#     image_path = '/workspace/shanshanguo/temporary/slp_aline_22test/'
#     image_other = '/workspace/shanshanguo/temporary/slp_aline_22testother/'
#     image_name = os.listdir(image_path)
#     npz_file = '/workspace/shanshanguo/project/SPIN-master_single/data/dataset_extras/slp_test.npz'
#     outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
#     #openpose_path = r'/workspace/shanshanguo/project/openpose-master/out_slp_rgb_test/'
#     #box_path = r'/workspace/shanshanguo/temporary/slp_rgb_demo_box/image_000004_00049.json'
#     image_name.sort()
#     #框还是没有对上 这里的sort顺序不对
#     #print(image_name[:200])
#     #sys.exit()
#     part = np.load(npz_file,allow_pickle=True)
#     imagename = part['imgname']
#     #print(type(imagename))
#     # print(imgname)
#     # sys.exit()
#     for imgname in image_name:
#         index = np.where(imagename==imgname)
#         index = index[0][0]
#         #imgname='80_44_0_RGB.png'
#         imgpath = os.path.join(image_path,imgname)
#         depth = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_depth.png')
#         ir = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_ir.png')
#         pm = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_pm.png')
#         #path1 = imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+'0_RGB_keypoints.json'
#         box = None
#         imgs, img, norm_img = process_image(imgpath, box, npz_file, index , input_res=constants.IMG_RES)
#         cv2.imwrite(outfile + imgname, imgs)
#         imgs1,img_depth, norm_imgdepth = process_image(depth, box, npz_file, index, input_res=constants.IMG_RES)
#         imgs2,img_ir, norm_imgir = process_image(ir, box, npz_file, index, input_res=constants.IMG_RES)
#         imgs3,img_pm, norm_imgpm = process_image(pm, box, npz_file, index, input_res=constants.IMG_RES)
#         #print(outfile)
        

#         with torch.no_grad():
#             pred_rotmat_rgb, pred_shape_rgb, pred_cam_rgb,pred_rotmat_depth, pred_shape_depth, pred_cam_depth,pred_rotmat_ir, \
#             pred_shape_ir, pred_cam_ir,pred_rotmat_pm, pred_shape_pm, pred_cam_pm,\
#                 pred_rotmat, pred_betas, pred_camera  = model(norm_img.to(device),norm_imgdepth.to(device),norm_imgir.to(device),norm_imgpm.to(device))
#             #print(pred_rotmat.shape, pred_betas.shape,pred_camera.shape)
#             pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
#             pred_vertices = pred_output.vertices
#             pred_joints = pred_output.joints
    
#         pred_cam_t = torch.stack([pred_camera[:,1],
#                                 pred_camera[:,2],
#                                 2*5000./(224 * pred_camera[:,0] +1e-9)],dim=-1)
    
#         camera_center = torch.zeros(1, 2, device=torch.device('cuda'))
    

#         pred_keypoints_2d = perspective_projection(pred_joints,
#                                                rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(0).expand(1, -1, -1),
#                                                translation=pred_cam_t,
#                                                focal_length=5000.,
#                                                camera_center=camera_center)
#         keypoints = pred_keypoints_2d.squeeze()
#         keypoints = keypoints.cpu().numpy()
#         keypoints = keypoints[25:40,]
#         keypoints+=112.
#         skels_idx = nameToIdx(skels_name, joints_name=joints_name)
#         img_key = vis_keypoints(outfile + imgname,keypoints,skels_idx) #2.png is background pic
#         cv2.imwrite(os.path.join(outfile,imgname.split('.')[0]+'_key.png'),img_key) # predict groudtruth draw
        
#         # Calculate camera parameters for rendering
#         camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
#         camera_translation = camera_translation[0].cpu().numpy()
#         pred_vertices = pred_vertices[0].cpu().numpy()
#         img = img.permute(1,2,0).cpu().numpy()

    
#         # Render parametric shape
#         img_shape = renderer(pred_vertices, camera_translation, img)
#         # Render side views
#         aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
#         center = pred_vertices.mean(axis=0)
#         rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        
#         # Render non-parametric shape
#         img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

#         # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

#         # Save reconstructions
#         cv2.imwrite(outfile + imgname.split('.')[0] + '_shape.png', 255 * img_shape[:,:,::-1])
#         cv2.imwrite(outfile + imgname.split('.')[0] + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
# """
# Demo code

# To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

# In summary, we provide 3 different ways to use our demo code and models:
# 1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
# 2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
# 3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

# Example with OpenPose detection .json
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
# ```
# Example with predefined Bounding Box
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
# ```
# Example with cropped and centered image
# ```
# python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
# ```

# Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
# """

# import torch
# from torchvision.transforms import Normalize
# import numpy as np
# import cv2
# import argparse
# import json
# import sys
# import os
# from os.path import join


# from models import hmr, SMPL
# from utils.imutils import crop
# from utils.renderer import Renderer
# from utils.geometry import  perspective_projection
# import config
# import constants
# import os

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
# #parser.add_argument('--img', type=str, required=True, help='Path to input image')
# #parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
# #parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
# parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
# skels_name = (
# 	# ('Pelvis', 'Thorax'),
# 	('Thorax', 'Head'),
# 	('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
# 	('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
# 	# ('Pelvis', 'R_Hip'),
# 	('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
# 	# ('Pelvis', 'L_Hip'),
# 	('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
# 	)
# colors = [(204, 204, 0),(51, 153, 51),(51,153,255)]
# joints_name = (
# 	"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
# 	"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
# 	"Neck")  # max std joints, first joint_num_ori will be true labeled
# def bbox_from_gt(npz_file, img_id):
#     part = np.load(npz_file,allow_pickle=True)
#     center = part['center'][img_id]
#     scale= part['scale'][img_id]
#     return center,scale 

# def nameToIdx(name_tuple, joints_name):  # test, tp,
# 	'''
# 	from reference joints_name, change current name list into index form
# 	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
# 	:param joints_name:
# 	:return:
# 	'''
# 	jtNm = joints_name
# 	if type(name_tuple[0]) == tuple:
# 		# Transer name_tuple to idx
# 		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
# 	else:
# 		# direct transfer
# 		return tuple(jtNm.index(tpl) for tpl in name_tuple)
# def vis_keypoints(img_path,keypoints,kps_lines, kp_thresh=0.4, alpha=1):
    
# 	# Perform the drawing on a copy of the image, to allow for blending.
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     kp_mask = np.copy(img)
#     kps = keypoints.T
# 	# Draw the keypoints.
#     for l in range(len(kps_lines)):
#         i1 = kps_lines[l][0]
#         i2 = kps_lines[l][1]
#         p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
#         p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
#         cv2.line(
# 				kp_mask, p1, p2,
# 				color=colors[0], thickness=2, lineType=cv2.LINE_AA)
#         cv2.circle(
# 				kp_mask, p1,
# 				radius=3, color=colors[1], thickness=-1, lineType=cv2.LINE_AA)
#         cv2.circle(
# 				kp_mask, p2,
# 				radius=3, color=colors[2], thickness=-1, lineType=cv2.LINE_AA)

# 	# Blend the keypoints.
#     return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

# def process_image(img_file, bbox_file, npz_file, img_id, input_res=224):
#     """Read image, do preprocessing and possibly crop it according to the bounding box.
#     If there are bounding box annotations, use them to crop the image.
#     If no bounding box is specified but openpose detections are available, use them to get the bounding box.
#     """
#     rescale =1.2
#     normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
#     img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
#     if bbox_file is None:
#         # Assume that the person is centerered in the image
#         center,scale = bbox_from_gt(npz_file, img_id)
#     else:
#         height = img.shape[0]
#         width = img.shape[1]
#         center = np.array([width // 2, height // 2])
#         scale = max(height, width) / 200
#     scale=rescale*scale
#     imgs = crop(img, center, scale, (input_res, input_res))
#     img = imgs.astype(np.float32) / 255.
#     img = torch.from_numpy(img).permute(2,0,1)
#     norm_img = normalize_img(img.clone())[None]
#     return imgs, img, norm_img

# if __name__ == '__main__':
#     args = parser.parse_args()
    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
#     # Load pretrained model
#     model = hmr(config.SMPL_MEAN_PARAMS).to(device)
#     checkpoint = torch.load(args.checkpoint,map_location='cpu')
#     model.load_state_dict(checkpoint['model'], strict=False)

#     # Load SMPL model
#     smpl = SMPL(config.SMPL_MODEL_DIR,
#                 batch_size=1,
#                 create_transl=False).to(device)
#     model.eval()

#     # Setup renderer for visualization
#     renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

#     # Preprocess input image and generate predictions
#     image_path = '/workspace/shanshanguo/temporary/slp_aline_22test/'
#     image_other = '/workspace/shanshanguo/temporary/slp_aline_22testother/'
#     image_name = os.listdir(image_path)
#     npz_file = '/workspace/shanshanguo/project/SPIN-master_single/data/dataset_extras/slp_test.npz'
#     outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
#     #openpose_path = r'/workspace/shanshanguo/project/openpose-master/out_slp_rgb_test/'
#     #box_path = r'/workspace/shanshanguo/temporary/slp_rgb_demo_box/image_000004_00049.json'
#     image_name.sort()
#     #框还是没有对上 这里的sort顺序不对
#     # print(image_name)
#     # sys.exit()
#     for index, imgname in enumerate(image_name):
#         #imgname='80_44_0_RGB.png'
#         imgpath = os.path.join(image_path,imgname)
#         depth = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_depth.png')
#         ir = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_ir.png')
#         pm = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_pm.png')
#         #path1 = imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+'0_RGB_keypoints.json'
#         box = None
#         imgs, img, norm_img = process_image(imgpath, box, npz_file, index , input_res=constants.IMG_RES)
#         cv2.imwrite(outfile + imgname, imgs)
#         imgs1,img_depth, norm_imgdepth = process_image(depth, box, npz_file, index , input_res=constants.IMG_RES)
#         imgs2,img_ir, norm_imgir = process_image(ir, box, npz_file, index , input_res=constants.IMG_RES)
#         imgs3,img_pm, norm_imgpm = process_image(pm, box, npz_file, index , input_res=constants.IMG_RES)

#         with torch.no_grad():
#             pred_rotmat, pred_betas, pred_camera = model(torch.cat([norm_img,norm_imgdepth,norm_imgir,norm_imgpm],1).to(device))
#             #print(pred_rotmat.shape, pred_betas.shape,pred_camera.shape)
#             pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
#             pred_vertices = pred_output.vertices
#             pred_joints = pred_output.joints
    
#         pred_cam_t = torch.stack([pred_camera[:,1],
#                                 pred_camera[:,2],
#                                 2*5000./(224 * pred_camera[:,0] +1e-9)],dim=-1)
    
#         camera_center = torch.zeros(1, 2, device=torch.device('cuda'))
    

#         pred_keypoints_2d = perspective_projection(pred_joints,
#                                                rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(0).expand(1, -1, -1),
#                                                translation=pred_cam_t,
#                                                focal_length=5000.,
#                                                camera_center=camera_center)
#         keypoints = pred_keypoints_2d.squeeze()
#         keypoints = keypoints.cpu().numpy()
#         keypoints = keypoints[25:40,]
#         keypoints+=112.
#         skels_idx = nameToIdx(skels_name, joints_name=joints_name)
#         img_key = vis_keypoints(outfile + imgname,keypoints,skels_idx) #2.png is background pic
#         cv2.imwrite(os.path.join(outfile,imgname.split('.')[0]+'_key.png'),img_key) # predict groudtruth draw
        
#         # Calculate camera parameters for rendering
#         camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
#         camera_translation = camera_translation[0].cpu().numpy()
#         pred_vertices = pred_vertices[0].cpu().numpy()
#         img = img.permute(1,2,0).cpu().numpy()

    
#         # Render parametric shape
#         img_shape = renderer(pred_vertices, camera_translation, img)
#         # Render side views
#         aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
#         center = pred_vertices.mean(axis=0)
#         rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        
#         # Render non-parametric shape
#         img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

#         # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

#         # Save reconstructions
#         cv2.imwrite(outfile + imgname.split('.')[0] + '_shape.png', 255 * img_shape[:,:,::-1])
#         cv2.imwrite(outfile + imgname.split('.')[0] + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import sys

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
from utils.geometry import  perspective_projection
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import config
import constants
import os
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
#parser.add_argument('--img', type=str, required=True, help='Path to input image')
#parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
#parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
skels_name = (
	# ('Pelvis', 'Thorax'),
	('Thorax', 'Head'),
	('Thorax', 'R_Shoulder'), ('R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'),
	('Thorax', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'),
	# ('Pelvis', 'R_Hip'),
	('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'),
	# ('Pelvis', 'L_Hip'),
	('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'),
	)
colors = [(204, 204, 0),(51, 153, 51),(51,153,255)]
colors_org = [(153,50,204),(0,0,255),(255,255,0)]
joints_name = (
	"R_Ankle", "R_Knee", "R_Hip", "L_Hip", "L_Knee", "L_Ankle", "R_Wrist", "R_Elbow", "R_Shoulder", "L_Shoulder",
	"L_Elbow", "L_Wrist", "Thorax", "Head", "Pelvis", "Torso",
	"Neck")  # max std joints, first joint_num_ori will be true labeled
def bbox_from_gt(npz_file, img_id):
    part = np.load(npz_file,allow_pickle=True)
    center = part['center'][img_id]
    scale= part['scale'][img_id]
    return center,scale 

def nameToIdx(name_tuple, joints_name):  # test, tp,
	'''
	from reference joints_name, change current name list into index form
	:param name_tuple:  The query tuple like tuple(int,) or tuple(tuple(int, ...), )
	:param joints_name:
	:return:
	'''
	jtNm = joints_name
	if type(name_tuple[0]) == tuple:
		# Transer name_tuple to idx
		return tuple(tuple([jtNm.index(tpl[0]), jtNm.index(tpl[1])]) for tpl in name_tuple)
	else:
		# direct transfer
		return tuple(jtNm.index(tpl) for tpl in name_tuple)
def vis_keypoints(img_path,keypoints,kps_lines, kp_thresh=0.4, alpha=1,flag=0):
    
	# Perform the drawing on a copy of the image, to allow for blending.
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kp_mask = np.copy(img)
    kps = keypoints.T
	# Draw the keypoints.
    if flag==0:
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            cv2.line(
                    kp_mask, p1, p2,
                    color=colors[0], thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[1], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[2], thickness=-1, lineType=cv2.LINE_AA)
    else:
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            cv2.line(
                    kp_mask, p1, p2,
                    color=colors_org[0], thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors_org[1], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors_org[2], thickness=-1, lineType=cv2.LINE_AA)

	# Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def process_image(img_file, bbox_file, npz_file, img_id, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    rescale =1.2
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None:
        # Assume that the person is centerered in the image
        center,scale = bbox_from_gt(npz_file, img_id)
    else:
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    #scale=rescale*scale
    imgs = crop(img, center, scale, (input_res, input_res))
    img = imgs.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return imgs, img, norm_img
def untranskey(kp, center, scale, inverse, r=0, f=0):
    scaleRGB = 256/1024
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    nparts = kp.shape[0]
    #print(nparts)
    for i in range(nparts):
 
        kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES],invert=inverse,rot=r)
    kp = kp/scaleRGB
    kp = kp.astype('float32')
    return kp
if __name__ == '__main__':
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint,map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)

    # Preprocess input image and generate predictions
    image_path = '/workspace/shanshanguo/temporary/slp_aline_22test/'
    image_other = '/workspace/shanshanguo/temporary/slp_aline_22testother/'
    image_name = os.listdir(image_path)
    image_org = '/workspace/shanshanguo/temporary/slp_orign_22rgb/'
    npz_file = '/workspace/shanshanguo/project/SPIN-master_single/data/dataset_extras/slp_test.npz'
    outfile = args.img.split('.')[0] if args.outfile is None else args.outfile
    #openpose_path = r'/workspace/shanshanguo/project/openpose-master/out_slp_rgb_test/'
    #box_path = r'/workspace/shanshanguo/temporary/slp_rgb_demo_box/image_000004_00049.json'
    image_name.sort()
    #框还是没有对上 这里的sort顺序不对
    #print(image_name[:200])
    #sys.exit()
    part = np.load(npz_file,allow_pickle=True)
    imagename = part['imgname']
    orgkey = part['part'][:,:14,:2]
    # print(orgkey)
    # print(orgkey.shape)
    # print(type(orgkey))
    #sys.exit()
    #print(type(imagename))
    # print(imgname)
    # sys.exit()
    for imgname in image_name:
        index = np.where(imagename==imgname)
        index = index[0][0]
        #imgname='80_44_0_RGB.png'
        imgpath = os.path.join(image_path,imgname)
        depth = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_depth.png')
        ir = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_ir.png')
        pm = join(image_other,imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+imgname.split('_')[2]+'_pm.png')
        #path1 = imgname.split('_')[0]+'_'+imgname.split('_')[1]+'_'+'0_RGB_keypoints.json'
        box = None
        imgs, img, norm_img = process_image(imgpath, box, npz_file, index , input_res=constants.IMG_RES)
        center,scale = bbox_from_gt(npz_file, index)
        #print(outfile)
        cv2.imwrite(outfile + imgname, imgs)
        imgs1,img_depth, norm_imgdepth = process_image(depth, box, npz_file, index , input_res=constants.IMG_RES)
        imgs2,img_ir, norm_imgir = process_image(ir, box, npz_file, index , input_res=constants.IMG_RES)
        imgs3,img_pm, norm_imgpm = process_image(pm, box, npz_file, index , input_res=constants.IMG_RES)

        with torch.no_grad():
            pred_rotmat_rgb, pred_shape_rgb, pred_cam_rgb,pred_rotmat_depth, pred_shape_depth, pred_cam_depth,pred_rotmat_ir, \
            pred_shape_ir, pred_cam_ir,pred_rotmat_pm, pred_shape_pm, pred_cam_pm,\
                pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device),norm_imgdepth.to(device),norm_imgir.to(device),norm_imgpm.to(device))
            #print(pred_rotmat.shape, pred_betas.shape,pred_camera.shape)
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
    
        pred_cam_t = torch.stack([pred_camera[:,1],
                                pred_camera[:,2],
                                2*5000./(224 * pred_camera[:,0] +1e-9)],dim=-1)
    
        camera_center = torch.zeros(1, 2, device=torch.device('cuda'))
    

        pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(0).expand(1, -1, -1),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
        keypoints = pred_keypoints_2d.squeeze()
        keypoints = keypoints.cpu().numpy()
        keypoints = keypoints[25:40,]
        keypoints+=112.
        skels_idx = nameToIdx(skels_name, joints_name=joints_name)
        img_key = vis_keypoints(outfile + imgname,keypoints,skels_idx,flag=0) #2.png is background pic
        cv2.imwrite(os.path.join(outfile,imgname.split('.')[0]+'_key.png'),img_key) # predict groudtruth draw
        
        temp = np.zeros((14,2))
        temp = untranskey(keypoints, center, scale, inverse=1, r=0, f=0)
        img_org_key = vis_keypoints(image_org + imgname,temp,skels_idx,flag=0)        
        cv2.imwrite(os.path.join(outfile,imgname.split('.')[0]+'_org_key.png'),img_org_key) # predict groudtruth draw                      

        keyorg = orgkey[index,:,:]
        keyorg = keyorg/(256/1024)
        img_org_keys = vis_keypoints(os.path.join(outfile,imgname.split('.')[0]+'_org_key.png'),keyorg,skels_idx,flag=1)        
        cv2.imwrite(os.path.join(outfile,imgname.split('.')[0]+'_org_key_final.png'),img_org_keys) # predict groudtruth draw     


        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()

    
        # Render parametric shape
        img_shape = renderer(pred_vertices, camera_translation, img)
        # Render side views
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
        center = pred_vertices.mean(axis=0)
        rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        
        # Render non-parametric shape
        img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

        # outfile = args.img.split('.')[0] if args.outfile is None else args.outfile

        # Save reconstructions
        cv2.imwrite(outfile + imgname.split('.')[0] + '_shape.png', 255 * img_shape[:,:,::-1])
        cv2.imwrite(outfile + imgname.split('.')[0] + '_shape_side.png', 255 * img_shape_side[:,:,::-1])

