import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import pickle


def load_and_process_images(file_path):
	# convert path from string to path object
	file_path = Path(file_path)
	print(file_path)
	img_files = sorted(file_path.glob('*.jpg'))
	num_images = len(img_files)
	if num_images < 1:
		raise RuntimeError('No images found')

	image_idx2id = {}
	images = []
	img_idx = 0

	for img_path in img_files:
		img_id = str(img_path).split('c')[1][1: -4]
		image_idx2id[img_idx] = img_id
		if img_idx % 10 == 0:
			print(img_idx)
		img_idx += 1
		with Image.open(img_path) as img_opener:
			img = img_opener
			if img.mode != 'RGB':
				img = img.convert('RGB')
			img = img.resize((64, 64), resample=5, reducing_gap=2.0)
			np_img = np.array(img)
			np_img = (np_img / np.float32(255)) * 2 - 1
			# np_img = np.pad(np_img, ((2, 2), (2, 2), (0, 0)), mode='constant')
			images.append(np_img)

	return images, image_idx2id


def store_data_to_hdf5(file_path, images, image_idx2id, hdf5_name=None):
	num_images = len(images)
	# convert path from string to path object
	hdf5_dir_path = Path(file_path + 'hdf5_files')
	# make new dir for hdf5 files
	hdf5_dir_path.mkdir(parents=True, exist_ok=True)
	# create new hdf5 file
	if hdf5_name is not None:
		name = f"{hdf5_name}"
	else:
		name = f"{num_images}_images"
	h5_file = h5py.File(hdf5_dir_path / f"{name}.h5", "w")
	# Create a dataset in the file
	h5_file.create_dataset("images", np.shape(images), np.float32, data=images)
	# Store data (serialize)
	with open(hdf5_dir_path / "image_idx2id.pkl", 'wb') as handle:
		pickle.dump(image_idx2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
	h5_file.close()
	return hdf5_dir_path


def main():
	file_path_str = './images/cropped'
	images, image_idx2id = load_and_process_images(file_path_str)
	hdf5_dir_path = store_data_to_hdf5(file_path_str, images, image_idx2id)
	num_images = len(images)

	# Open the HDF5 file
	h5_file = h5py.File(hdf5_dir_path / f"{num_images}_images.h5", "r+")
	images_after = np.array(h5_file["/images"]).astype(np.float32)

	with open(hdf5_dir_path / "image_idx2id.pkl", 'rb') as handle:
		unserialized_data = pickle.load(handle)

	print('end')
	h5_file.close()


if __name__ == '__main__':
	main()


