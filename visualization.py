import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import math, random

def draw_skeleton_3d_dynamic(pts3d_list, save_dir=None, elev = -90, azim=-90, axis_lim=[-0.5, 0.5]):
	if save_dir is not None:
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	plt.rcParams['image.interpolation'] = 'nearest'
	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev = elev, azim=azim)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	body_edges = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],[11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])
	colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
	n_pts = pts3d_list[0].shape[0]

	for i,pts3d in enumerate(pts3d_list):
		ax.clear()
		ax.set_xlim(axis_lim)
		ax.set_ylim(axis_lim)
		ax.set_zlim(axis_lim)

		# Draw points:
		ax.plot(*zip(*pts3d), marker='o', color='r', ls='', markersize=5)

		# Draw edges:
		for edge in body_edges:
			if edge[0] <  n_pts and edge[1] < n_pts:
				ax.plot(pts3d[edge,0], pts3d[edge,1], pts3d[edge,2], color=colors[1])

		if save_dir is None:
			plt.pause(0.000001)
		else:
			path = os.path.join(save_dir, str(i).zfill(5) + '.jpeg')
			fig.savefig(path)#, bbox_inches='tight', pad_inches = 0)

		print('{}/{}      '.format(i+1, len(pts3d_list)), end='\r')
		

	plt.show()

def draw_skeleton_3d(pts3d, elev = -90, azim=-90, axis_lim=[-0.5, 0.5]):
	plt.rcParams['image.interpolation'] = 'nearest'
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev = elev, azim=azim)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.axis('equal')
	ax.set_xlim(axis_lim)
	ax.set_ylim(axis_lim)
	ax.set_zlim(axis_lim)

	body_edges = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],[11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])
	colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
	n_pts = pts3d.shape[0]

	# Draw points:
	ax.plot(*zip(*pts3d), marker='o', color='r', ls='', markersize=5)

	for edge in body_edges:
		if edge[0] <  n_pts and edge[1] < n_pts:
			ax.plot(pts3d[edge,0], pts3d[edge,1], pts3d[edge,2], color=colors[1])

	plt.show()


def draw_skeleton_2d(pts2d, canvas_shape=(1080,1920,3), draw_indices=None, name='canvas', show=True, background=(0,0,0)):
	bones = np.asarray([1,3,0,-1,2,7,5,9,6,-1,8,5,6,11,12,13,14])
	# body_edges = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],[11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])
	canvas = np.zeros(canvas_shape, np.uint8)
	canvas[:] = background

	for i in range(len(pts2d)):
		x = int(round(pts2d[i][0]))
		y = int(round(pts2d[i][1]))
		canvas = cv2.circle(canvas, (x,y), 5, (0,0,255), 5)

		if draw_indices is not None:
				font = cv2.FONT_HERSHEY_SIMPLEX
				bottomLeftCornerOfText = (x,y)
				fontScale = 1
				fontColor = (255,255,255)
				lineType = 2
				cv2.putText(canvas,str(i), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

		# for edge in body_edges:
		# 	x2 = int(round(pts2d[parent_i][0]))
		# 	y2 = int(round(pts2d[parent_i][1]))
		# 	if x+y != 0 and x2+y2 != 0:
		# 		canvas = cv2.line(canvas,(x,y),(x2,y2),(255,0,0),4)
		# 	ax.plot(pts3d[edge,0], pts3d[edge,1], pts3d[edge,2], color=colors[1])

		if bones is not None:
			parent_i = bones[i]
			if parent_i >= 0:
				x2 = int(round(pts2d[parent_i][0]))
				y2 = int(round(pts2d[parent_i][1]))
				if x+y != 0 and x2+y2 != 0:
					canvas = cv2.line(canvas,(x,y),(x2,y2),(255,0,0),4)

	if show:
		cv2.imshow(name,canvas)
		cv2.waitKey()

	return canvas


def draw_skeleton_3d_with_sphere(pts3d, sphere, elev = -90, azim=-90, axis_lim=[-0.5, 0.5]):
	plt.rcParams['image.interpolation'] = 'nearest'
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev = elev, azim=azim)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.axis('equal')
	ax.set_xlim(axis_lim)
	ax.set_ylim(axis_lim)
	ax.set_zlim(axis_lim)

	body_edges = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],[11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])
	colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
	n_pts = pts3d.shape[0]

	# Draw points:
	ax.plot(*zip(*pts3d), marker='o', color='r', ls='', markersize=5)

	for edge in body_edges:
		if edge[0] <  n_pts and edge[1] < n_pts:
			ax.plot(pts3d[edge,0], pts3d[edge,1], pts3d[edge,2], color=colors[1])

	# Draw sphere:
	ax.plot(*zip(*sphere), marker='o', color='g', ls='', markersize=3)

	plt.show()


def draw_skeleton_3d_dynamic_with_object(pts3d_list, save_dir=None, elev = -90, azim=-90, axis_lim=[-0.5, 0.5], object_type='sphere'):
	if save_dir is not None:
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	plt.rcParams['image.interpolation'] = 'nearest'
	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(elev = elev, azim=azim)
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	body_edges = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],[11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])
	cube_edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
	colors = plt.cm.hsv(np.linspace(0, 1, 20)).tolist()
	n_pts = pts3d_list[0][0].shape[0]

	for i,(pts3d,object_pts) in enumerate(pts3d_list):
		ax.clear()
		ax.set_xlim(axis_lim)
		ax.set_ylim(axis_lim)
		ax.set_zlim(axis_lim)

		# Draw points:
		ax.plot(*zip(*pts3d), marker='o', color='r', ls='', markersize=5)

		# Draw edges:
		for edge in body_edges:
			if edge[0] <  n_pts and edge[1] < n_pts:
				ax.plot(pts3d[edge,0], pts3d[edge,1], pts3d[edge,2], color=colors[1])

		# Draw object:
		ax.plot(*zip(*object_pts), marker='o', color='g', ls='', markersize=3)

		if object_type == 'cube':
			for j,edge in enumerate(cube_edges):
				ax.plot(object_pts[edge,0], object_pts[edge,1], object_pts[edge,2], color=colors[j])

		if save_dir is None:
			plt.pause(0.000001)
		else:
			path = os.path.join(save_dir, str(i).zfill(5) + '.jpeg')
			fig.savefig(path)#, bbox_inches='tight', pad_inches = 0)

		print('{}/{}      '.format(i+1, len(pts3d_list)), end='\r')
		

	plt.show()

def draw_cube_2d(obj2d_pts, canvas):
	cube_edges = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
	colors = plt.cm.hsv(np.linspace(0, 1, 20)).tolist()

	for pi in range(obj2d_pts.shape[0]):
		x = int(round(obj2d_pts[pi,0]))
		y = int(round(obj2d_pts[pi,1]))
		cv2.circle(canvas, (x,y), 5, (0,255,0), 5)

	for j,edge in enumerate(cube_edges):
		pt1 = (int(round(obj2d_pts[cube_edges[j,0],0])), int(round(obj2d_pts[cube_edges[j,0],1])))
		pt2 = (int(round(obj2d_pts[cube_edges[j,1],0])), int(round(obj2d_pts[cube_edges[j,1],1])))
		cv2.line(canvas, pt1, pt2, color=colors[j], thickness=2)

	return canvas

def generate_fibonacci_sphere(num_pts=100,randomize=True, axis_lim =[-2.0,2.0], draw=False):
	
	# Generate sphere points:
	rnd = 1.
	if randomize:
		rnd = random.random() * num_pts

	points = np.empty((num_pts,3), dtype=np.float32)
	offset = 2./num_pts
	increment = math.pi * (3. - math.sqrt(5.));

	for i in range(num_pts):
		y = ((i * offset) - 1) + (offset / 2);
		r = math.sqrt(1 - pow(y,2))

		phi = ((i + rnd) % num_pts) * increment

		x = math.cos(phi) * r
		z = math.sin(phi) * r

		points[i,:] = (x,y,z)

	# Draw:
	if draw:
		plt.rcParams['image.interpolation'] = 'nearest'
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(elev = -90, azim=-90)
		ax.set_xlim(axis_lim)
		ax.set_ylim(axis_lim)
		ax.set_zlim(axis_lim)

		ax.axis('equal')
		ax.plot(*zip(*points), marker='o', color='r', ls='')
		plt.show()

	return points


def transform_sphere(points, offset=(0.0,0.0,0.0), scale=1.0):
	for i in range(points.shape[0]):
		points[i] = points[i]*scale + offset

	return points

def generate_cube():
	'''
      4_______5     
      /.     /|
   0 /_.___1/ |
	|  .   |  |
	| 7----|-/6
    |______|/
   3       2
	'''
	points = np.empty((8,3), dtype=np.float32)
	step = 0.5
	points[0,:] = (-step, step, step)
	points[1,:] = ( step, step, step)
	points[2,:] = ( step,-step, step)
	points[3,:] = (-step,-step, step)
	points[4,:] = (-step, step,-step)
	points[5,:] = ( step, step,-step)
	points[6,:] = ( step,-step,-step)
	points[7,:] = (-step,-step,-step)

	return points


