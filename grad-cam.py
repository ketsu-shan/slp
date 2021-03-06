import torch
import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
sys.path.append('..')
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import argparse
import os
import torch.nn as nn
from models import hmr
import config

data_path = '../' + config.SMPL_MEAN_PARAMS
mean_params = np.load(data_path)
init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
print(init_pose.size())
i=0##testing in what
hmr_model = hmr(data_path, pretrained=False)
checkpoint = torch.load('../model_checkpoint.pt', map_location='cpu')
hmr_model.load_state_dict(checkpoint['model'], strict=False)
hmr_model.eval()
#for name in hmr_model.state_dict():
#	print(name)
#	print(hmr_model.state_dict()[name])
image = []
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():##resnet50没有.feature这个特征，直接删除用就可以。
            x = module(x)
            # print('name=',name)
            # print('x.size()=',x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		# 用去掉回归层的网络提取给定层的特征
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		# 返回特征和最后一层输出
		target_activations, output  = self.feature_extractor(x)
		# 拉成一维
		output = output.view(output.size(0), -1)
		#print('classfier=',output.size())
		if self.cuda:
			output = output.cpu()
			#output = hmr_model.regressor(output).cuda()##这里就是为什么我们多加载一个resnet模型进来的原因，因为后面我们命名的model不包含fc层，但是这里又偏偏要使用。#
			pred_rotmat, pred_shape, pred_cam = hmr_model.regressor(output, init_pose, init_shape, init_cam, n_iter=3)
		else:
			#output = hmr_model.regressor(output)##这里对应use-cuda上更正一些bug,不然用use-cuda的时候会导致类型对不上,这样保证既可以在cpu上运行,gpu上运行也不会出问题.
			pred_rotmat, pred_shape, pred_cam = hmr_model.regressor(output, init_pose, init_shape, init_cam, n_iter=3)
			print(pred_rotmat.shape)
			pred_rotmat = pred_rotmat.view(pred_rotmat.size(0), -1)
			output = pred_shape
			print(output.size())
		return target_activations, output

def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input

def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	
	cam = cam / np.max(cam)
	cv2.imwrite("cam/cam_{}.jpg".format(name), np.uint8(255 * cam))
	#cv2.imwrite("cam/cam_{}.jpg".format(name), np.uint8(255 * cam))
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			# 特征图和返回最后一层结果
			features, output = self.extractor(input)

		if index == None:
			# 选择最大的数值索引
			index = np.argmax(output.cpu().data.numpy())
			
		print('********')
		print('index=',index)
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()##features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
		#self.model.zero_grad()
		one_hot.backward(retain_graph=True)##这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		#print('grads_val',grads_val.shape)
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		#print('weights',weights.shape)
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		#print('cam',cam.shape)
		#print('features',features[-1].shape)
		#print('target',target.shape)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam
class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model#这里同理，要的是一个完整的网络，不然最后维度会不匹配。
		self.model.eval()
		self.cuda = use_cuda
		k = 1
		if self.cuda:
			self.model = model.cuda()
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return (torch.clamp(grad_in[0], min=0.0),)
	def forward(self, input, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
		input = Variable(torch.unsqueeze(input, dim=0).float(), requires_grad=False)
		return self.model(input, init_pose, init_shape, init_cam, n_iter)

	def __call__(self, input, index = None):
		if self.cuda:
			pred_rotmat, pred_shape, pred_cam = self.forward(input.cuda())
		else:
			pred_rotmat, pred_shape, pred_cam = self.forward(input, init_pose, init_shape, init_cam, n_iter=3)
			pred_rotmat = pred_rotmat.view(pred_rotmat.size(0), -1)
		output = pred_shape
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		#print(input.grad)
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)
		#self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)
		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")
	return args

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args() 

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	model = hmr(data_path, pretrained=False)

	checkpoint = torch.load('../model_checkpoint.pt', map_location='cpu')
	model.load_state_dict(checkpoint['model'], strict=False)

	model.eval()
	del model.regressor
	#modules = list(resnet.children())[:-1]
	#model = torch.nn.Sequential(*modules)

	#print(model)
	grad_cam = GradCam(model , \
					target_layer_names = ["layer4"], use_cuda=args.use_cuda)##这里改成layer4也很简单，我把每层name和size都打印出来了，想看哪层自己直接嵌套就可以了。（最后你会在终端看得到name的）
	x=os.walk(args.image_path)
	for root,dirs,filename in x:
	#print(type(grad_cam))
		print(filename)
	for s in filename:
    		image.append(cv2.imread(args.image_path+s,1))
		#img = cv2.imread(filename, 1)
	for img in image:
		img = np.float32(cv2.resize(img, (224, 224))) / 255
		input = preprocess_image(img)
		input.required_grad = True
	# If None, returns the map for the highest scoring category.
	# Otherwise, targets the requested index.
		# shape
		
		mask_sum = np.zeros((224,224))
		for target_index in range(0, init_shape.size()[1]):
			i=i+1
			mask = grad_cam(input, target_index)
			show_cam_on_image(img, mask,i)
			cv2.imwrite("cam/cam_{}.jpg".format(i+10), np.uint8(255 * mask))
			mask_sum += mask
			cv2.imwrite("cam/cam_{}.jpg".format(i+100), np.uint8(255 * mask_sum))
			mask_sum1 = mask_sum / np.max(mask_sum)
			cv2.imwrite("cam/cam_{}.jpg".format(i+1000), np.uint8(255 * mask_sum1))

		cv2.imwrite("cam/cam_{}.jpg".format(i+10000), np.uint8(25.5 * mask_sum))
		