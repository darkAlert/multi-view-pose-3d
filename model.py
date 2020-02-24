import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Params():
	def __init__(self):
		self.num_cams = 6         # number of cameras
		self.num_pts = 17         # number of points (body joints)
		self.num_epoch = 10
		self.lr = 1e-3            # learning rate
		self.weight_decay = 1e-5  # weight decay


class ReprojectLayer(nn.Module):
	def __init__(self, N, M):
		super(ReprojectLayer, self).__init__()
		self.bias = None
		self.weight = None
		self.N = N
		self.M = M

	def forward(self, pts3d_flat, projmats_flat):
		'''Projects 3D points to an plane using projection matrices
		Args:
			pts3d_flat:   flat tensor containing N 3d points, 
						  where N is the number of points
			projmat_flat: flat tensor containing M projection matrices shaped (3x4), 
						  where M is the number of cameras
		Returns:
			NxMx2 projected 2d points with a flattened shape
		'''
		pts3d = pts3d_flat.view(-1,3)           # (N, 3)
		projmats = projmats_flat.view(-1,3,4)   # (M, 3, 4)

		# Euclidean points to homogeneous:
		pts3d_hom = torch.cat([pts3d, torch.ones((pts3d.shape[0], 1), dtype=pts3d.dtype, device=pts3d.device)], dim=1)


		# Project 3d points to a plane:
		projections = []
		n = projmats.shape[0]
		for i in range(n):
			pts2d_hom = pts3d_hom @ projmats[i].t()
			# Homogeneous points to euclidean:
			pts2d = (pts2d_hom.transpose(1, 0)[:-1] / pts2d_hom.transpose(1, 0)[-1]).transpose(1, 0) 
			projections.append(pts2d)

		pts2d_flat = torch.cat(projections).flatten()
		return pts2d_flat


class JointRecNet(nn.Module):
	def __init__(self, num_pts, num_cams, hidden_sizes = (1000,500)):
		super(JointRecNet, self).__init__()

		input_size = num_cams * num_pts * 2          # input 2d body joints
		joint3d_size = num_pts * 3                   # output 3d body joints
		projmat_size = num_cams * 3 * 4              # 3x4 projection matrices

		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.projmat = nn.Linear(hidden_sizes[-1], projmat_size)
		self.reproject = ReprojectLayer(num_pts,num_cams) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		# joint3d = torch.tanh(joint3d)
		projmat = self.projmat(x)
		x = self.reproject(joint3d,projmat)
		return x

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			# x = torch.tanh(x)
			x = self.fc2(x)
			x = F.relu(x)
			# x = torch.tanh(x)
			joint3d = self.joint3d(x)
			# joint3d = torch.tanh(joint3d)
			projmat = self.projmat(x)
			return joint3d.view(-1,3)#, projmat.view(-1,3,4)

	@staticmethod
	def create_model(params = Params()):
		model = JointRecNet(params.num_pts, params.num_cams)

		return model





class ReprojectLayer_batch(nn.Module):
	def __init__(self, N, M):
		super(ReprojectLayer_batch, self).__init__()
		self.bias = None
		self.weight = None
		self.N = N
		self.M = M

	def forward(self, pts3d_flat, projmats_flat):
		'''Projects 3D points to an plane using projection matrices
		Args:
			pts3d_flat:   flat tensor containing N 3d points, 
						  where N is the number of points
			projmat_flat: flat tensor containing M projection matrices shaped (3x4), 
						  where M is the number of cameras
		Returns:
			NxMx2 projected 2d points with a flattened shape
		'''
		pts3d = pts3d_flat.view(-1, self.N, 3)            # (batch_size, N, 3)
		projmats = projmats_flat.view(-1, self.M, 3, 4)   # (batch_size, M, 3, 4)
		batch_size = projmats.shape[0]
		assert batch_size == pts3d.shape[0]

		# Euclidean points to homogeneous:
		pts3d_hom = torch.empty((batch_size, self.N, 4),dtype=pts3d.dtype, device=pts3d.device)
		for i in range(batch_size):
			pts3d_hom[i] = torch.cat([pts3d[i], torch.ones((pts3d[i].shape[0], 1), dtype=pts3d[i].dtype, device=pts3d[i].device)], dim=1)

		# Project 3d points to a plane:
		batch = []
		for i in range(batch_size):
			projections = []
			for j in range(self.M):
				pts2d_hom = pts3d_hom[i,:] @ projmats[i,j].t()
				# Homogeneous points to euclidean:
				pts2d = (pts2d_hom.transpose(1, 0)[:-1] / pts2d_hom.transpose(1, 0)[-1]).transpose(1, 0) 
				projections.append(pts2d.view(-1))
			batch.append(torch.cat(projections, dim=0).view(-1,self.M*self.N*2))

		return torch.cat(batch, dim=0)


class JointRecNet_batch(nn.Module):
	# hidden_sizes = (3000,2500)
	def __init__(self, num_pts, num_cams, hidden_sizes = (1000,500)):
		super(JointRecNet_batch, self).__init__()
		self.num_pts = num_pts
		input_size = num_cams * num_pts * 2          # input 2d body joints
		joint3d_size = num_pts * 3                   # output 3d body joints
		projmat_size = num_cams * 3 * 4              # 3x4 projection matrices

		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.projmat = nn.Linear(hidden_sizes[-1], projmat_size)
		self.reproject = ReprojectLayer_batch(num_pts,num_cams) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		# joint3d = torch.tanh(joint3d)
		projmat = self.projmat(x)
		x = self.reproject(joint3d,projmat)
		return x

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			x = self.fc2(x)
			x = F.relu(x)
			joint3d = self.joint3d(x)
			return joint3d.view(-1,self.num_pts,3)

	@staticmethod
	def create_model(params = Params()):
		model = JointRecNet_batch(params.num_pts, params.num_cams)

		return model



class JointRecNet_big(nn.Module):
	def __init__(self, num_pts, num_cams, hidden_sizes = (1000,2500,500)):
		super(JointRecNet_big, self).__init__()
		self.num_pts = num_pts
		input_size = num_cams * num_pts * 2          # input 2d body joints
		joint3d_size = num_pts * 3                   # output 3d body joints
		projmat_size = num_cams * 3 * 4              # 3x4 projection matrices

		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.projmat = nn.Linear(hidden_sizes[-1], projmat_size)
		self.reproject = ReprojectLayer_batch(num_pts,num_cams) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		# joint3d = torch.tanh(joint3d)
		projmat = self.projmat(x)
		x = self.reproject(joint3d,projmat)
		return x

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			x = self.fc2(x)
			x = F.relu(x)
			x = self.fc3(x)
			x = F.relu(x)
			joint3d = self.joint3d(x)
			return joint3d.view(-1,self.num_pts,3)

	@staticmethod
	def create_model(params = Params()):
		model = JointRecNet_big(params.num_pts, params.num_cams)

		return model




class ReprojectLayer_v2(nn.Module):
	def __init__(self, N, M):
		super(ReprojectLayer_v2, self).__init__()
		self.N = N
		self.M = M
		self.weight = torch.nn.Parameter(torch.Tensor(M, 3, 4))
		# self.bias = False
		# self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		# self.weight.data.uniform_(-0.1, 0.1)

	def forward(self, pts3d_flat):
		'''Projects 3D points to an plane using projection matrices
		Args:
			pts3d_flat:   flat tensor containing N 3d points, 
						  where N is the number of cameras
		Returns:
			NxMx2 projected 2d points with a flattened shape
		'''
		batch_size = pts3d_flat.shape[0]
		pts3d = pts3d_flat.view(-1, 3)          # (N*batch_size, 3)

		# Euclidean points to homogeneous:
		pts3d_hom = torch.cat([pts3d, torch.ones((pts3d.shape[0], 1), dtype=pts3d.dtype, device=pts3d.device)], dim=1)

		# Project 3d points to a plane:
		projections = []
		m = self.weight.shape[0]
		for i in range(m):
			pts2d_hom = pts3d_hom @ self.weight[i].t()
			# Homogeneous points to euclidean:
			pts2d = (pts2d_hom.transpose(1, 0)[:-1] / pts2d_hom.transpose(1, 0)[-1]).transpose(1, 0)
			# projections.append(pts2d.view(-1,self.N*2))
			projections.append(pts2d.view(batch_size,-1))

		return torch.cat(projections, dim=1)



class JointRecNet_v2(nn.Module):
	def __init__(self, num_pts, num_cams, hidden_sizes = (1000,500)):
		super(JointRecNet_v2, self).__init__()
		self.num_pts = num_pts
		input_size = num_cams * num_pts * 2          # input 2d body joints
		joint3d_size = num_pts * 3                   # output 3d body joints
		projmat_size = num_cams * 3 * 4              # 3x4 projection matrices

		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.reproject = ReprojectLayer_v2(num_pts,num_cams) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		x = self.reproject(joint3d)
		return x, joint3d.view(-1,self.num_pts,3)

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			x = self.fc2(x)
			x = F.relu(x)
			joint3d = self.joint3d(x)
			return joint3d.view(-1,self.num_pts,3)


	@staticmethod
	def create_model(params = Params()):
		model = JointRecNet_v2(params.num_pts, params.num_cams)

		return model


class JointRecNet_v2_big(nn.Module):
	def __init__(self, num_pts, num_cams, hidden_sizes = (1000,2500,500)):
		super(JointRecNet_v2_big, self).__init__()
		self.num_pts = num_pts
		input_size = num_cams * num_pts * 2          # input 2d body joints
		joint3d_size = num_pts * 3                   # output 3d body joints
		projmat_size = num_cams * 3 * 4              # 3x4 projection matrices

		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.reproject = ReprojectLayer_v2(num_pts,num_cams) 

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		x = self.reproject(joint3d)
		return x#, joint3d.view(-1,self.num_pts,3)

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			x = self.fc2(x)
			x = F.relu(x)
			x = self.fc3(x)
			x = F.relu(x)
			joint3d = self.joint3d(x)
			return joint3d.view(-1,self.num_pts,3)


	@staticmethod
	def create_model(params = Params()):
		model = JointRecNet_v2_big(params.num_pts, params.num_cams)

		return model








class ReprojectLayer_v3(nn.Module):
	def __init__(self):
		super(ReprojectLayer_v3, self).__init__()
		self.bias = None
		self.weight = None

	def forward(self, pts3d_flat, projmats):
		'''Projects 3D points to an plane using projection matrices
		Args:
			pts3d_flat:   flat tensor containing N 3d points, 
						  where N is the number of cameras
			projmat: flat tensor containing M projection matrices shaped (3x4), 
						  where M is the number of points
		Returns:
			NxMx2 projected 2d points with a flattened shape
		'''
		pts3d = pts3d_flat.view(-1, 3)          # (N, 3)

		# Euclidean points to homogeneous:
		pts3d_hom = torch.cat([pts3d, torch.ones((pts3d.shape[0], 1), dtype=pts3d.dtype, device=pts3d.device)], dim=1)

		# Project 3d points to a plane:
		projections = []
		n = projmats.shape[0]
		for i in range(n):
			pts2d_hom = pts3d_hom @ projmats[i].t()
			# Homogeneous points to euclidean:
			pts2d = (pts2d_hom.transpose(1, 0)[:-1] / pts2d_hom.transpose(1, 0)[-1]).transpose(1, 0) 
			projections.append(pts2d)

		pts2d_flat = torch.cat(projections).flatten()
		return pts2d_flat


class JointRecNet_v3(nn.Module):
	def __init__(self, input_size, joint3d_size, num_cams, device, hidden_sizes = (1000,500)):
		super(JointRecNet_v3, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_sizes[0])
		self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
		self.joint3d = nn.Linear(hidden_sizes[-1], joint3d_size)
		self.projmats = torch.Tensor(num_cams,3,4).to(device)
		torch.nn.init.kaiming_uniform_(self.projmats, a=math.sqrt(5)) 
		self.projmats.requires_grad = True
		self.reproject = ReprojectLayer_v3()


	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		joint3d = self.joint3d(x)
		x = self.reproject(joint3d,self.projmats)
		# print (self.projmats)
		return x

	def predict(self, x):
		with torch.no_grad():
			x = self.fc1(x)
			x = F.relu(x)
			x = self.fc2(x)
			x = F.relu(x)
			joint3d = self.joint3d(x)
			return joint3d.view(-1,3)#, projmat.view(-1,3,4)

	@staticmethod
	def create_model(params = Params(), device = None):
		input_size = params.num_cams * params.num_pts * 2   # input 2d body joints
		joint3d_size = params.num_pts * 3                   # output 3d body joints

		model = JointRecNet_v3(input_size, joint3d_size, params.num_cams, device)

		return model
