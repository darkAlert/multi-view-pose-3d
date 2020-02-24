import os
import numpy as np
import torch
import sys
sys.path.append('/home/darkalert/KazendiJob/Data/HoloVideo/scripts/')
from data_struct import DataStruct


def get_npz_names(path_to_npz):
	npz_names = []
	for dirpath, dirnames, filenames in os.walk(path_to_npz):
		for filename in [f for f in filenames if f.endswith('.npz') or f.endswith('.NPZ')]:
			name = filename.split('.')[0]
			npz_names.append(name)
	npz_names.sort()

	return npz_names

def load_npz(path_to_npz):
	data = np.load(path_to_npz, encoding='latin1', allow_pickle=True)
	names = data['names']
	keypoints = data['keypoints']
	boxes = data['boxes']

	return names, keypoints, boxes

def get_data(data_dirs, norm_size=(1920/2,1080/2)):
	if not isinstance(data_dirs, list):
		data_dirs = [data_dirs]

	# Load data:
	kpts_samples = []
	for data_dir in data_dirs:
		kpts_list = []
		cam_names = get_npz_names(data_dir)
		num_cam = len(cam_names)
		for cam_name in cam_names:
			_, kp, _ = load_npz(os.path.join(data_dir,cam_name + '.npz'))
			kpts_list.append(np.array(kp[:,:,:2])) #.flatten()

		# Trim extra frames:
		num_frames = min([kpts.shape[0] for kpts in kpts_list])
		kpts_list = [kpts[:num_frames] for kpts in kpts_list]

		# Stack the keypoints (M,N,17,2) -> (N,Mx17,2):
		kpts = np.stack(kpts_list, axis=1).reshape((-1,17*num_cam,2))

		# Normalize points:
		if norm_size is not None:
			n = kpts.shape[0]
			for i in range(n):
				kpts[i,:,:] /= norm_size
				kpts[i,:,:] -= (1.0,1.0)

		kpts_samples.append(kpts.reshape((-1,17*num_cam*2)))

	kpts = np.concatenate(kpts_samples, axis=0)

	return kpts


def load_data_batch_v1(shuffle=True):
	data_dirs = []
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/andrey/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/andrey_hololens/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_2_freestyle/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_2_front_position/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_2_rotation/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_3_freestyle/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_4/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_7_front_position/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_8_front_position/')
	data_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/test/keypoints_and_bboxes/person_9_rotation/')
	
	X = get_data(data_dirs)

	if shuffle:
		np.random.shuffle(X)

	return torch.from_numpy(X).float()


def extract_pts2d(X, pos=0, step=17*2):
	return X[pos*step:(pos+1)*step].reshape(-1,2)

def normalize_pts_to_plane(pts2d, plane_size=(1920,1080)):
	''' depricated '''
	n = pts2d.shape[0]
	for i in range(n):
		pts2d[i,:] += (1.0,1.0)
		pts2d[i,:] *= plane_size

	return pts2d

def normalize_pts_to_frame(pts2d, frame_size=(1920,1080)):
	dy = (frame_size[0]-frame_size[1])*0.5
	size = (frame_size[0]*0.5,frame_size[0]*0.5)
	n = pts2d.shape[0]
	for i in range(n):
		pts2d[i,:] += (1.0,1.0)
		pts2d[i,:] *= size
		pts2d[i,1] -= dy

	return pts2d

def load_data_batch_v2(shuffle=True, norm_size=(1920,1080)):
	root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/poses'
	data = DataStruct().parse(root_dir, levels='subject/light/garment/scene/cam', ext='npz')
	samples = []

	# Iterate over scenes:
	for node,path in data.nodes('scene'):
		kpts_list = []
		for cam in data.items(node):
			_, kp, _ = load_npz(cam.abs_path)
			kpts_list.append(np.array(kp[:,:,:2]))
		num_cam = len(kpts_list)

		# Trim extra frames:
		num_frames = min([kpts.shape[0] for kpts in kpts_list])
		kpts_list = [kpts[:num_frames] for kpts in kpts_list]

		# Stack the keypoints (M,N,17,2) -> (N,Mx17,2):
		kpts = np.stack(kpts_list, axis=1).reshape((-1,17*num_cam,2))

		# Normalize points:
		if norm_size is not None:
			dy = (norm_size[0]-norm_size[1])*0.5
			size = (norm_size[0]*0.5,norm_size[0]*0.5)
			n = kpts.shape[0]
			for i in range(n):
				kpts[i,:,1] += dy
				kpts[i,:,:] /= size
				kpts[i,:,:] -= (1.0,1.0)

		samples.append(kpts.reshape((-1,17*num_cam*2)))

	X = np.concatenate(samples, axis=0)

	if shuffle:
		np.random.shuffle(X)

	return torch.from_numpy(X).float()

def load_data_batch_v2_weights(shuffle=True, norm_size=(1920,1080)):
	root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/poses'
	data = DataStruct().parse(root_dir, levels='subject/light/garment/scene/cam', ext='npz')
	samples = []

	# Iterate over scenes:
	for node,path in data.nodes('scene'):
		kpts_list = []
		for cam in data.items(node):
			_, kp, _ = load_npz(cam.abs_path)
			kpts_list.append(np.array(kp[:,:,:]))
		num_cam = len(kpts_list)

		# Trim extra frames:
		num_frames = min([kpts.shape[0] for kpts in kpts_list])
		kpts_list = [kpts[:num_frames] for kpts in kpts_list]

		# Stack the keypoints (M,N,17,2) -> (N,Mx17,2):
		kpts = np.stack(kpts_list, axis=1).reshape((-1,17*num_cam,3))

		# Normalize points:
		if norm_size is not None:
			dy = (norm_size[0]-norm_size[1])*0.5
			size = (norm_size[0]*0.5,norm_size[0]*0.5)
			n = kpts.shape[0]
			for i in range(n):
				kpts[i,:,1] += dy
				kpts[i,:,:2] /= size
				kpts[i,:,:2] -= (1.0,1.0)

		samples.append(kpts.reshape((-1,17*num_cam*3)))

	X = np.concatenate(samples, axis=0)

	if shuffle:
		np.random.shuffle(X)

	return torch.from_numpy(X).float()


def load_data_batch_v3(shuffle=True, norm_size=(1920,1080), target=None):
	root_dir = '/home/darkalert/KazendiJob/Data/HoloVideo/Data/poses'
	data = DataStruct().parse(root_dir, levels='subject/light/garment/scene/cam', ext='npz')
	samples = []

	# Iterate over scenes:
	for node,path in data.nodes('scene'):
		# Find target path:
		if target is not None:
			if target != path:
				continue

		kpts_list = []
		for cam in data.items(node):
			_, kp, _ = load_npz(cam.abs_path)
			kpts_list.append(np.array(kp[:,:,:2]))
		num_cam = len(kpts_list)

		# Trim extra frames:
		num_frames = min([kpts.shape[0] for kpts in kpts_list])
		kpts_list = [kpts[:num_frames] for kpts in kpts_list]

		# Stack the keypoints (M,N,17,2) -> (N,Mx17,2):
		kpts = np.stack(kpts_list, axis=1).reshape((-1,17*num_cam,2))

		# Normalize points:
		if norm_size is not None:
			dy = (norm_size[0]-norm_size[1])*0.5
			size = (norm_size[0]*0.5,norm_size[0]*0.5)
			n = kpts.shape[0]
			for i in range(n):
				kpts[i,:,1] += dy
				kpts[i,:,:] /= size
				kpts[i,:,:] -= (1.0,1.0)

		samples.append(kpts.reshape((-1,17*num_cam*2)))

		if target is not None:
			break

	X = np.concatenate(samples, axis=0)

	if shuffle:
		np.random.shuffle(X)

	return torch.from_numpy(X).float()


def shuffle_tensor(X):
	return X[torch.randperm(X.size()[0])]

