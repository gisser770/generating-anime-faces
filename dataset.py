import torch
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Tuple, Dict, List
import h5py
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time


transform = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


class MyDataset(Dataset):
	def __init__(self, path, number_of_samples, is_train=True) -> None:
		print('dataset init start')
		# if this is train dataset
		self.is_train = is_train

		# Set variables
		self.path = {'images': Path(path)}
		self.requested_data_size = number_of_samples
		# Load features
		self.features = {}
		self._get_features()

		# Create list of entries
		# self.entries = self._get_entries()
		print('dataset init end')

	def __getitem__(self, index: int) -> Tuple:
		# return self.entries[index]['x'], self.entries[index]['y']
		# img, img_id = self._get_entry(index)
		# return transform(img), img_id
		# return img, img_id
		return self._get_entry(index)

	def __len__(self) -> int:
		"""
		:return: the length of the dataset (number of sample).
		"""
		# return len(self.features['images'])
		return self.images.shape[0]

	def _get_features(self) -> Any:
		"""
		Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
		into the memory.
		:return:
		:rtype:
		"""
		# open hd5f files containing all of the images in np array
		h5_path_list = sorted(self.path['images'].glob('*.h5'))
		num_h5_files = len(h5_path_list)
		if num_h5_files < 1:
			raise RuntimeError('No h5 files found')
		if num_h5_files != 1:
			raise RuntimeError('Too many h5 files found')
		h5_file = h5py.File(h5_path_list[0], "r")
		print('start loading data')
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		images_np = np.array(h5_file["/images"]).astype(np.float32)[:self.requested_data_size, ]
		self.images = torch.cat([torch.unsqueeze(transform(x).to(device), 0) for x in images_np], 0)
		print('end loading data')
		h5_file.close()

		# open pkl file containing dict with image id to image idx on the hd5f file
		image_id2idx_path = sorted(self.path['images'].glob("*.pkl"))
		with open(image_id2idx_path[0], 'rb') as handle:
			self.features['image_idx2id'] = pickle.load(handle)

	def _get_entries(self) -> List:
		"""
		This function create a list of all the entries. We will use it later in __getitem__
		:return: list of samples
		"""
		entries = []

		for idx, item in self.features.items():
			entries.append(self._get_entry(item))

		return entries

	def _get_entry(self, index: int) -> tuple:
		ret = self.images[index, :, :, :]
		return ret


def main():
	# paths = {}
	# paths['images'] = Path("./data/100_first_pic/hdf5_files/")
	# paths['other'] = Path("./data/cache/")
	# # paths['images'] = Path("/home/student/Benny/HW2/hdf5_results")
	# # paths['other'] = Path("/home/student/Benny/HW2/cache")
	# my_data = MyDataset("./data/100_first_pic/hdf5_files/")
	# train_loader = torch.utils.data.DataLoader(dataset=my_data, batch_size=64, shuffle=True)


	dataset = MyDataset('./data/images/croppedhdf5_files', 32000)
	train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

	get_batch_time = 0
	time_ = time.time()

	for i, (x, y) in enumerate(train_loader):
		get_batch_time += time.time() - time_
		print(f'{i} batch time = {get_batch_time}')
		time_ = time.time()


if __name__ == '__main__':
	main()

