import os
import cv2
import numpy as np

def get_names(path_to_npz):
	names = []
	for dirpath, dirnames, filenames in os.walk(path_to_npz):
		for filename in [f for f in filenames if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]:
			names.append(filename)
	names.sort()

	return names


def megre_frames(src_dirs, dst_dir, resolution=(1920,1080)):
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)

	name_list = []
	for src_dir in src_dirs:
		name_list.append(get_names(src_dir))
	n = min([len(names) for names in name_list])
	name_list = [names[:n] for names in name_list]

	for i, names in enumerate(zip(*name_list)):
		images = []
		for j, name in enumerate(names):
			path = os.path.join(src_dirs[j], name)
			img = cv2.imread(path,1)

			if img.shape[1] != resolution[0]:
				img = cv2.resize(img, (0,0),fx=3, fy=3)

			images.append(img)

		result = np.concatenate(images, axis=0)
		
		dst_path = os.path.join(dst_dir, str(i).zfill(5) + '.png')
		cv2.imwrite(dst_path, result)

		print('{}/{}      '.format(i+1, n), end='\r')

	print ('All done!')



if __name__ == '__main__':
	src_dirs = []
	src_dirs.append('/home/darkalert/KazendiJob/Data/HoloVideo/Data/frames/person_8/freestyle/cam1')
	src_dirs.append('/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/test/person8_freestyle_3d')
	# src_dirs.append('/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/test/v2-batch-big_3d')

	dst_dir = '/home/darkalert/KazendiJob/DLab/net-reconstruction-3d/test/merged_person8_freestyle_3d'

	megre_frames(src_dirs, dst_dir)
