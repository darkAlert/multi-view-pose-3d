import os
import torch
import torch.optim as optim
import numpy as np
import time
from data import load_data_batch_v1, load_data_batch_v2, load_data_batch_v2_weights, shuffle_tensor
from model import Params, JointRecNet_v3, JointRecNet_v2, JointRecNet, JointRecNet_batch, JointRecNet_v2_big, JointRecNet_big
from visualization import draw_skeleton_3d

# create_model = JointRecNet_v3.create_model
# create_model = JointRecNet_v2.create_model
create_model = JointRecNet_v2_big.create_model
# create_model = JointRecNet.create_model
# create_model = JointRecNet_batch.create_model
# create_model = JointRecNet_big.create_model


def train():
	# Cuda or CPU:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Set params:
	params = Params()
	params.num_epoch = 1
	params.lr = 1e-4

	# Load data:
	X = load_data_batch_v1().to(device)
	print ('Training data:',X.shape)

	# Init model:
	model = create_model(params)
	model.to(device)
	print (model)


	# # Optimizer params:
	# optimizer_params = []
	# for key,module in dict(model.named_parameters()).items():
	# 	if key == 'reproject.weight':
	# 		optimizer_params.append({'params': module,'weight_decay':0.0,'lr':params.lr})
	# 	else:
	# 		optimizer_params.append({'params': module})


	# Training:
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
	# optimizer = torch.optim.Adam(optimizer_params, lr=params.lr, weight_decay=params.weight_decay)
	model.train()

	for ei in range(params.num_epoch):
		for i in range(X.shape[0]):

			# Forward pass:
			optimizer.zero_grad()
			X_pred = model.forward(X[i])

			# Compute the loss:
			loss = criterion(X_pred, X[i])
			if i%10 == 0:
				print("[ITER]: %i, [LOSS]: %.6f" % (i, loss.item()))

			# Backward pass:
			loss.backward()

			# for i,p in enumerate(model.parameters()):
				# print (i,p.grad.shape)

			# Update params:
			optimizer.step()

		print("[EPOCH]: %i, [LOSS]: %.6f" % (ei, loss.item()))

	# Save model:
	model_path = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/models/last_model_v2_' + str(loss.item())+ '.pth'
	torch.save(model.state_dict(), model_path)
	print ('Model has been saved to ', model_path)


	# Test:
	joint3d = model.predict(X[0]).view(17,3).cpu().numpy()
	print ('Xi:',X[i])
	print (joint3d)
	draw_skeleton_3d(joint3d)

def train_batch(batch_size=100):
	# Cuda or CPU:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Set params:
	params = Params()
	params.num_epoch = 35
	params.lr = 1e-3

	# Load data:
	X = load_data_batch_v2().to(device)
	print ('Training data:',X.shape)

	# Init model:
	model = create_model(params)
	model.to(device)
	print (model)

	#Optimizer params:
	optimizer_params = []
	for key,module in dict(model.named_parameters()).items():
		if key == 'reproject.weight':
			optimizer_params.append({'params': module,'weight_decay':0.0,'lr':0.01})
		else:
			optimizer_params.append({'params': module})

	# Loss and optimizer:
	criterion = torch.nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
	optimizer = torch.optim.Adam(optimizer_params, lr=params.lr, weight_decay=params.weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
	
	# Training
	model.train()
	n = X.shape[0]
	for ei in range(params.num_epoch):
		start = time.time()
		for i in range(0,n,batch_size):
			count = min(batch_size,n-i)
			X_batch = X[i:i+count]

			# Forward pass:
			optimizer.zero_grad()
			# X_pred, joints3d_pred = model.forward(X_batch)
			X_pred = model.forward(X_batch)

			# Compute the loss:
			loss = criterion(X_pred, X_batch)

			# Backward pass:
			loss.backward()

			# Update params:
			optimizer.step()

		# Log:
		elapsed_time = time.time() - start
		lr_log = ["%.5f" % g['lr'] for g in optimizer.param_groups]
		print("[EPOCH]: %i, [LOSS]: %.6f, [elapsed]: %.4fs, [lr]: %s" % (ei, loss.item(), elapsed_time, lr_log))
		
		# Scheduler:
		scheduler.step()

	# Save model:
	model_path = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/models/v2-big-norm-weighted-mse' + str(loss.item())+ '.pth'
	torch.save(model.state_dict(), model_path)
	print ('Model has been saved to ', model_path)

	# Test:
	joint3d = model.predict(X[0]).view(17,3).cpu().numpy()
	print ('Xi:',X[100])
	print (joint3d)
	draw_skeleton_3d(joint3d)


def weighted_mse_loss(input, target, weight):
	X = (input - target) ** 2
	X[:,:,0] = X[:,:,0]*weight
	X[:,:,1] = X[:,:,1]*weight
	return torch.mean(X)

def mse_loss(input, target):
	return torch.mean((input - target) ** 2)

def train_weighted(batch_size=100):
	# Cuda or CPU:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Set params:
	params = Params()
	params.num_epoch = 35
	params.lr = 1e-3

	# Load data:
	X = load_data_batch_v2_weights().to(device)
	print ('Training data:',X.shape)

	# Init model:
	model = create_model(params)
	model.to(device)
	print (model)

	#Optimizer params:
	optimizer_params = []
	for key,module in dict(model.named_parameters()).items():
		if key == 'reproject.weight':
			optimizer_params.append({'params': module,'weight_decay':0.0,'lr':0.01})
		else:
			optimizer_params.append({'params': module})

	# Loss and optimizer:
	criterion = torch.nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
	optimizer = torch.optim.Adam(optimizer_params, lr=params.lr, weight_decay=params.weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
	
	# Training
	model.train()
	n = X.shape[0]
	for ei in range(params.num_epoch):
		X = shuffle_tensor(X)
		start = time.time()
		for i in range(0,n,batch_size):
			count = min(batch_size,n-i)
			sample_batch = X[i:i+count].view(count,-1,3)
			X_batch = sample_batch[:,:,:2].reshape(count,-1)
			W_batch = sample_batch[:,:,2]

			# Forward pass:
			optimizer.zero_grad()
			# X_pred, joints3d_pred = model.forward(X_batch)
			X_pred = model.forward(X_batch)

			# Compute the loss:
			loss = weighted_mse_loss(X_pred.view(count,-1,2), X_batch.view(count,-1,2), W_batch)
			
			# # Additional loss for cam1:
			# joints2d_pred = joints3d_pred[:,:,:2]
			# joints2d_cam1_gt = X_batch[:,:params.num_pts*2].view(-1,params.num_pts,2)
			# loss += criterion(joints2d_pred, joints2d_cam1_gt)

			# Backward pass:
			loss.backward()

			# Update params:
			optimizer.step()

		# Log:
		elapsed_time = time.time() - start
		lr_log = ["%.5f" % g['lr'] for g in optimizer.param_groups]
		print("[EPOCH]: %i, [LOSS]: %.6f, [elapsed]: %.4fs, [lr]: %s" % (ei, loss.item(), elapsed_time, lr_log))
		
		# Scheduler:
		scheduler.step()

	# Save model:
	model_path = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/models/v2-big-norm-weighted-mse' + str(loss.item())+ '.pth'
	torch.save(model.state_dict(), model_path)
	print ('Model has been saved to ', model_path)

	# Test:
	joint3d = model.predict(X[0].view(-1,3)[:,:2]).view(17,3).cpu().numpy()
	print (joint3d)
	draw_skeleton_3d(joint3d)



if __name__ == '__main__':
	# train()
	# train_batch()
	train_weighted()