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

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
import os

from os.path import join
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='Path to pretrained checkpoint')
#parser.add_argument('--img', type=str, required=True, help='Path to input image')
#parser.add_argument('--bbox', type=str, default=None, help='Path to .json file containing bounding box coordinates')
#parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')
# BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
# BODY_PARTS= [
#     (16, 14),
#     (14, 12),
#     (17, 15),
#     (15, 13),
#     (12, 13),
#     (6, 12),
#     (7, 13),
#     (6, 7),
#     (6, 8),
#     (7, 9),
#     (8, 10),
#     (9, 11),
#     (2, 3),
#     (1, 2),
#     (1, 3),
#     (2, 4),
#     (3, 5),
#     (4, 6),
#     (5, 7)
# ]
BODY_PARTS= [
    (16, 1),
    (2, 12),
    (1, 15),
    (16, 0),
    (2, 6),
    (9, 2),
    (7, 8),#
    (2, 3),
    (15, 17),
    (9, 10),
    (11, 10), #
    (12, 13),
    (1, 2),
    (4, 3),#
    (5, 4),#
    (7, 6),#
    (13, 14)
]
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
              ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
              ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
              ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

# def detect_key_point(openpose_file, image_path, out_dir):
#     with open(openpose_file, 'r') as f:
#         keypoints = json.load(f)['people'][0]['pose_keypoints']#改了pose_keypoints_2d
#     keypoints = np.reshape(np.array(keypoints), (-1,3))
#     frame = cv2.imread(image_path)
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
#     scalefactor = 2.0
#     points = []
#     for idx, keypoint in enumerate(keypoints):
#             # draw keypoint
#             x, y, prob = keypoint
#             points.append((int(x), int(y)) if prob > 0.05 else None)
        
#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert (partFrom in BODY_PARTS)
#         assert (partTo in BODY_PARTS)
 
#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]
 
#         if points[idFrom] and points[idTo]:
#             cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
#             cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
#     text = keypoint
#     cv2.putText(frame, text,(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     cv2.imwrite(os.path.join(out_dir), frame)
#     cv2.imshow('OpenPose using OpenCV', frame)

def showAnns(img_path, openpose_file, BODY_PARTS,out_dir):
    colors = [(204, 204, 0),(51, 153, 51),(51,153,255)] # 标注关节点的颜色
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints']#改了pose_keypoints_2d
    keypoints = np.reshape(np.array(keypoints), (-1,3)) 
    print(keypoints.shape)  
    img = cv2.imread(img_path)
    img = img.copy()
    print(keypoints)
    for i in range(len(keypoints)):
        kpt = np.array(keypoints[i])
        x = kpt[0]
        y = kpt[1]
        prob = kpt[2]
        if prob > 0.05:
            cv2.circle(img, (int(x), int(y)), 1, colors[2], -1) # 画点
        
        for part in BODY_PARTS:
            # 通过body_part_m来连线
            # 部位为part[0]的节点坐标，这里需要减去1.是因为得到的结果是以0开头的，而提供的是以1开头
            keypoint_1 =keypoints[part[0] - 1] 
            x_1 = int(keypoint_1[0]) # 单个部位坐标x
            y_1 = int(keypoint_1[1])
            keypoint_2 = keypoints[part[1] - 1]
            x_2 = int(keypoint_2[0])
            y_2 = int(keypoint_2[1])
            if keypoint_1[2] > 0 and keypoint_2[2] > 0:
                # 画线  参数--# img:图像，起点坐标，终点坐标，颜色，线的宽度
                # opencv读取图片通道为BGR
                cv2.line(img, (x_1, y_1), (x_2, y_2), colors[1], 1)
    cv2.imwrite(os.path.join(out_dir), img)
    #cv2.imshow('keypoints', img)
    #cv2.waitKey(20000)

def showAnnskey(img_path, keypoints, openpose_file, BODY_PARTS,out_dir):
    colors = [(204, 204, 0),(51, 153, 51),(51,153,255)] # 标注关节点
    img = cv2.imread(img_path)
    img = img.copy()
    for i in range(len(keypoints)):
        kpt = np.array(keypoints[i])
        x = kpt[0]
        y = kpt[1]
        prob = kpt[2]
        cv2.circle(img, (int(x), int(y)), 3, colors[1], -1) # 画点
        
        for part in BODY_PARTS:
            # 通过body_part_m来连线
            # 部位为part[0]的节点坐标，这里需要减去1.是因为得到的结果是以0开头的，而提供的是以1开头
            keypoint_1 =keypoints[part[0] - 1] 
            x_1 = int(keypoint_1[0]) # 单个部位坐标x
            y_1 = int(keypoint_1[1])
            keypoint_2 = keypoints[part[1] - 1]
            x_2 = int(keypoint_2[0])
            y_2 = int(keypoint_2[1])
            if keypoint_1[2] > 0 and keypoint_2[2] > 0:
                # 画线  参数--# img:图像，起点坐标，终点坐标，颜色，线的宽度
                # opencv读取图片通道为BGR
                cv2.line(img, (x_1, y_1), (x_2, y_2), colors[1], 1)
    cv2.imwrite(os.path.join(out_dir), img)
    #cv2.imshow('keypoints', img)
    #cv2.waitKey(20000)

def bbox_from_openpose(openpose_file, rescale=1.6, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    #print('2')
    with open(openpose_file, 'r') as f:
        #print('1')
        keypoints = json.load(f)['people'][0]['pose_keypoints']#改了pose_keypoints_2d
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    print(valid_keypoints.shape)
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale

    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    #print(img_file)
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        print('here')
        center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    cv2.imwrite('/workspace/shanshanguo/temporary/test_key/2.png', img)
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img

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
    image_name = "100_0_0_"
    image_other = '/workspace/shanshanguo/temporary/slp_aline_22testother/'
    openpose = '/workspace/shanshanguo/project/openpose-master/output_22_slp/'
    #box_path = r'/workspace/shanshanguo/temporary/slp_rgb_demo_box/image_000004_00049.json'
    out_dit = '/workspace/shanshanguo/temporary/test_key/'
    depth = join(image_other,image_name+'depth.png')
    print(depth)
    ir = join(image_other,image_name+'ir.png')
    pm = join(image_other,image_name+'pm.png')
    path1 = image_name+'RGB_keypoints.json'
    openpose_path = os.path.join(openpose,path1)
    print(openpose_path)
    #box = os.path.join(box_path,path2)
    box = None
    out_dir = '/workspace/shanshanguo/temporary/test_key/1.png'
    out_dir2 = '/workspace/shanshanguo/temporary/test_key/2.png'
    out_dir3 = '/workspace/shanshanguo/temporary/test_key/3.png'
    #detect_key_point(openpose_path, os.path.join(image_path,image_name+'RGB.png'), out_dir) 
    showAnns(os.path.join(image_path,image_name+'RGB.png'), openpose_path, BODY_PARTS,out_dir)
    img, norm_img = process_image(os.path.join(image_path,image_name+'RGB.png'), box, openpose_path, input_res=constants.IMG_RES)
    print(type(norm_img))
    #cv2.imwrite(os.path.join(out_dir2), norm_img)

    img_depth, norm_imgdepth = process_image(depth, box, openpose_path, input_res=constants.IMG_RES)
    img_ir, norm_imgir = process_image(ir, box, openpose_path, input_res=constants.IMG_RES)
    img_pm, norm_imgpm = process_image(pm, box, openpose_path, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(torch.cat([norm_img,norm_imgdepth,norm_imgir,norm_imgpm],1).to(device))
        #print(pred_rotmat.shape, pred_betas.shape,pred_camera.shape)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()
    print(img.shape)
    pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*5000./(224 * pred_camera[:,0] +1e-9)],dim=-1)
    camera_center = torch.zeros(1, 2, device=torch.device('cuda'))
    pred_keypoints_3d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3, device=torch.device('cuda')).unsqueeze(0).expand(1, -1, -1),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    print(pred_keypoints_3d.shape)
    print(type(pred_keypoints_3d))
    print(pred_keypoints_3d.cpu().numpy())
    keypoints = pred_keypoints_3d.cpu().numpy()[25:]
    showAnnskey(os.path.join(image_path,image_name+'RGB.png'),keypoints, openpose_path, BODY_PARTS,out_dir3)                                       

    
        # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)
        
        # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
    center = pred_vertices.mean(axis=0)
    rot_vertices = np.dot((pred_vertices - center), aroundy) + center
        
        # Render non-parametric shape
    img_shape_side = renderer(rot_vertices, camera_translation, np.ones_like(img))

    outfile = image_name if args.outfile is None else args.outfile
    #print(outfile)

        # Save reconstructions
    cv2.imwrite(outfile + image_name.split('.')[0] + '_shape.png', 255 * img_shape[:,:,::-1])
    cv2.imwrite(outfile + image_name.split('.')[0] + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
