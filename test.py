import os
import torch
import torch.optim as optim
import numpy as np
import cv2
from model import Params, JointRecNet_v3, JointRecNet_v2, JointRecNet, JointRecNet_batch, JointRecNet_v2_big, JointRecNet_big
from visualization import draw_skeleton_3d, draw_skeleton_2d, draw_skeleton_3d_dynamic, generate_fibonacci_sphere, transform_sphere, draw_skeleton_3d_with_sphere, draw_skeleton_3d_dynamic_with_object, generate_cube, draw_cube_2d
from data import extract_pts2d, normalize_pts_to_frame, load_data_batch_v1, load_data_batch_v2, load_data_batch_v3
import copy

# create_model = JointRecNet_v3.create_model
# create_model = JointRecNet_v2.create_model
create_model = JointRecNet_v2_big.create_model
# create_model = JointRecNet.create_model
# create_model = JointRecNet_batch.create_model
# create_model = JointRecNet_big.create_model


def test_3d_dynamic(model_path, X, start_idx=0, stop_idx=1000, result_dir=None, elev=90, azim=50, axis=[-0.25, 0.25]):
	#CUDA:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)
	print ('Training data:',X.shape)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	# Test:
	joint3d_list = []
	for sample_idx in range(start_idx, stop_idx):
		# Predict:
		joint3d = model.predict(X[sample_idx]).view(17,3).cpu().numpy()
		joint3d_list.append(joint3d)

		print('{}/{}      '.format(sample_idx+1, stop_idx), end='\r')

	#Visualize:
	draw_skeleton_3d_dynamic(joint3d_list, result_dir, elev=elev, azim=azim, axis_lim=axis)


def test_range(model_path, X, result_dir, start_idx=0, stop_idx=1000, num_cameras=6):
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)
	print ('Training data:',X.shape)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	# Test:
	for sample_idx in range(start_idx, stop_idx):
		# Predict:
		X_pred = model.forward(X[sample_idx]).detach().view(-1).cpu().numpy()

		# Draw:
		row_imgs = []
		for j in range(num_cameras):
			pts2d_gt = normalize_pts_to_frame(extract_pts2d(X[sample_idx].cpu().numpy(),j))
			skeleton_gt = draw_skeleton_2d(pts2d_gt, name='gt-'+str(j), show=False)
			skeleton_pd = normalize_pts_to_frame(extract_pts2d(X_pred,j))
			skeleton_pd = draw_skeleton_2d(skeleton_pd, name='pred-'+str(j), show=False, background=(128,128,128))

			#Resize and concatenate:
			target_size = (640,360)
			skeleton_gt = cv2.resize(skeleton_gt,target_size)
			skeleton_pd = cv2.resize(skeleton_pd,target_size)
			result = np.concatenate([skeleton_gt,skeleton_pd],axis=1)
			row_imgs.append(result)

		result = np.concatenate(row_imgs,axis=0)
		result_path = os.path.join(result_dir, str(sample_idx).zfill(5) + '.jpeg')
		cv2.imwrite(result_path, result)

		print('{}/{}      '.format(sample_idx+1, stop_idx), end='\r')



def test(model_path, X, result_dir, sample_idx=0):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)
	print ('Training data:',X.shape)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	#3D:
	X_pred = model.forward(X[sample_idx]).detach().view(-1).cpu().numpy()
	joint3d = model.predict(X[sample_idx]).view(17,3).cpu().numpy()
	draw_skeleton_3d(joint3d)

	#2D:
	row_imgs = []
	for j in range(6):
		pts2d_gt = normalize_pts_to_frame(extract_pts2d(X[sample_idx].cpu().numpy(),j))
		skeleton_gt = draw_skeleton_2d(pts2d_gt, name='gt-'+str(j), show=False)
		skeleton_pd = normalize_pts_to_frame(extract_pts2d(X_pred,j))
		skeleton_pd = draw_skeleton_2d(skeleton_pd, name='pred-'+str(j), show=False, background=(128,128,128))

		#Resize and concatenate:
		target_size = (640,360)
		skeleton_gt = cv2.resize(skeleton_gt,target_size)
		skeleton_pd = cv2.resize(skeleton_pd,target_size)
		result = np.concatenate([skeleton_gt,skeleton_pd],axis=1)
		row_imgs.append(result)

	result = np.concatenate(row_imgs,axis=0)

	result_path = os.path.join(result_dir, 'result_idx' + str(sample_idx) + '.png')
	cv2.imwrite(result_path, result)

	cv2.imshow('result',result)
	cv2.waitKey()


def test_shpere(model_path, X, sample_idx=0, elev=-90, azim=-90, axis=[-0.35, 0.35]):
	# CUDA:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)
	print ('data:',X.shape)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	#3D:
	joint3d = model.predict(X[sample_idx]).view(17,3).cpu().numpy()

	# Hip point:
	hip = (joint3d[11]+joint3d[12])*0.5

	# Shpere:
	sphere = generate_fibonacci_sphere()
	sphere = transform_sphere(sphere, offset=hip, scale=0.3)

	draw_skeleton_3d_with_sphere(joint3d, sphere, elev=elev, azim=azim, axis_lim=axis)


def test_3d_dynamic_with_object(model_path, X, start_idx=0, stop_idx=1000, result_dir=None, elev=-90, azim=-90, axis=[-0.35, 0.35], object_type='sphere'):
	#CUDA:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)
	print ('Training data:',X.shape)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	if object_type == 'sphere':
		bject_ones = generate_fibonacci_sphere(num_pts=100)
	elif object_type == 'cube':
		object_ones = generate_cube()
	else:
		assert False

	# Test:
	joint3d_list = []
	for sample_idx in range(start_idx, stop_idx):
		# Predict:
		joint3d = model.predict(X[sample_idx]).view(17,3).cpu().numpy()

		# Hip point:
		hip = (joint3d[11]+joint3d[12])*0.5

		# Shpere:
		object_pts = copy.deepcopy(object_ones)
		object_pts = transform_sphere(object_pts, offset=hip, scale=0.025)

		joint3d_list.append((joint3d,object_pts))

		print('{}/{}      '.format(sample_idx+1, stop_idx), end='\r')

	#Visualize:
	draw_skeleton_3d_dynamic_with_object(joint3d_list, result_dir, elev=elev, azim=azim, axis_lim=axis, object_type=object_type)


def test_projections(model_path, X, result_dir, start_idx=0, stop_idx=1000, num_cameras=6, object_type='cube'):
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Data:
	X = X.to(device)

	# Load model:
	model = create_model(Params())
	model.load_state_dict(torch.load(model_path))
	model.to(device)
	print ('Model has been loaded from ', model_path)

	# Shpere:
	if object_type == 'sphere':
		bject_ones = generate_fibonacci_sphere(num_pts=1000)
	elif object_type == 'cube':
		object_ones = generate_cube()
	else:
		assert False

	# Test:
	for sample_idx in range(start_idx, stop_idx):
		# Predict:
		joint3d = model.predict(X[sample_idx])
		X_pred = model.reproject(joint3d).detach().view(-1).cpu().numpy()
		joint3d_cpu = joint3d.view(17,3).cpu().numpy()

		# Hip point:
		hip = (joint3d_cpu[11]+joint3d_cpu[12])*0.5

		# Shpere:
		obj3d = copy.deepcopy(object_ones)
		obj3d = transform_sphere(obj3d, offset=hip, scale=0.025) #0.04
		obj3d = torch.from_numpy(obj3d).to(device)
		obj3d = obj3d.view(1,-1,3)

		# Project shpere on planes:
		obj2d_pred = model.reproject(obj3d).detach().view(6,-1,2).cpu().numpy()

		# Draw:
		row_imgs = []
		for j in range(num_cameras):
			pts2d_gt = normalize_pts_to_frame(extract_pts2d(X[sample_idx].cpu().numpy(),j))
			skeleton_gt = draw_skeleton_2d(pts2d_gt, name='gt-'+str(j), show=False)
			skeleton_pd = normalize_pts_to_frame(extract_pts2d(X_pred,j))
			skeleton_pd = draw_skeleton_2d(skeleton_pd, name='pred-'+str(j), show=False, background=(128,128,128))

			# Draw shpere:
			obj2d_norm = normalize_pts_to_frame(obj2d_pred[j])
			skeleton_pd = draw_cube_2d(obj2d_norm,skeleton_pd)

			#Resize and concatenate:
			target_size = (640,360)
			skeleton_gt = cv2.resize(skeleton_gt,target_size)
			skeleton_pd = cv2.resize(skeleton_pd,target_size)
			result = np.concatenate([skeleton_gt,skeleton_pd],axis=1)
			row_imgs.append(result)

		result = np.concatenate(row_imgs,axis=0)
		result_path = os.path.join(result_dir, str(sample_idx).zfill(5) + '.jpeg')
		cv2.imwrite(result_path, result)

		print('{}/{}      '.format(sample_idx+1, stop_idx), end='\r')


if __name__ == '__main__':
	#Load data:
	# X = load_data_batch_v2(shuffle=False)
	X = load_data_batch_v3(shuffle=False, target='person_8/light-100_temp-5600/garments_1/freestyle')

	model_path = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/models/v2-big-norm-weighted-mse0.0008937817183323205.pth'
	result_dir = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/test/v2-big-norm-weighted-mse_cube_2/'


	# test(model_path, X, result_dir, sample_idx=1780)
	# test_range(model_path, X, result_dir, start_idx=0, stop_idx=1000, num_cameras=6)

	# # cam1-loss:
	# elev, azim = -90, -90
	# axis = [-0.35, 0.35]

	elev, azim = -90, -90
	axis = [-0.025, 0.025]
	# test_3d_dynamic(model_path, X, start_idx=0, stop_idx=X.shape[0], result_dir=None, elev=elev, azim=azim, axis=axis)


	elev, azim = -80, -21
	axis = [-0.025, 0.025]
	# test_shpere(model_path, X, sample_idx=111)
	test_3d_dynamic_with_object(model_path, X, start_idx=0, stop_idx=X.shape[0], result_dir=None, elev=elev, azim=azim, axis=axis, object_type='cube')


	# test_projections(model_path, X, result_dir, start_idx=0, stop_idx=1000, object_type='cube')

	
