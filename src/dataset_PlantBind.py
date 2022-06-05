import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from torch.utils.data.distributed import DistributedSampler

from utils import data_format

def Onehot(input_data):
	seq = list(input_data.iloc[:,0])
	# seq = seq.cpu().numpy()
	bicoding_dict={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0],
	               'K':[0,0,0,0],'W':[0,0,0,0],'Y':[0,0,0,0],'R':[0,0,0,0],'M':[0,0,0,0],'S':[0,0,0,0]}
	final_bicoding=[]
	for each_seq in seq:
		feat_bicoding=[]
		for each_nt in each_seq:
			feat_bicoding += bicoding_dict[str(each_nt)]
		final_bicoding.append(feat_bicoding)
	
	return final_bicoding

class PlantBindDataset(Dataset): #分别导入一个数据，可选择 train / test / valid
	def __init__(self, data_dir_path, mode):
		"""
		Inputs:
			mode: train, valid, test
		"""
		self.data_dir = data_dir_path 
		self.mode = mode
		
		if self.mode == 'train':
			# 读取序列信息		
			self.train_seq = pd.read_table(os.path.join(self.data_dir,"train_sequence.table"),header=None)
			self.train_seq_oht = np.array(Onehot(self.train_seq)) #(seqnum, 404)
			# 读取序列结构信息
			self.train_DNA_shape = np.load(os.path.join(self.data_dir,"train_DNA_shape.npy")) #(seqnum, 97, 14)
			# 读取样本标签
			self.train_label = pd.read_table(os.path.join(self.data_dir,"train_label.txt"), header=0)
			self.class_name = list(self.train_label.columns)
			self.train_label = self.train_label.to_numpy()  #(seqnum, 315)
		elif self.mode == 'valid':
			# 读取序列信息		
			self.valid_seq = pd.read_table(os.path.join(self.data_dir,"valid_sequence.table"),header=None)
			self.valid_seq_oht = np.array(Onehot(self.valid_seq))
			# 读取序列结构信息
			self.valid_DNA_shape = np.load(os.path.join(self.data_dir,"valid_DNA_shape.npy"))
			# 读取样本标签
			self.valid_label = pd.read_table(os.path.join(self.data_dir,"valid_label.txt"), header=0)
			self.class_name = list(self.valid_label.columns)
			self.valid_label = self.valid_label.to_numpy()			
		else:			
			# 读取序列信息		
			self.test_seq = pd.read_table(os.path.join(self.data_dir,"test_sequence.table"),header=None)
			self.test_seq_oht = np.array(Onehot(self.test_seq))
			# 读取序列结构信息
			self.test_DNA_shape = np.load(os.path.join(self.data_dir,"test_DNA_shape.npy"))
			# 读取样本标签
			self.test_label = pd.read_table(os.path.join(self.data_dir,"test_label.txt"), header=0)
			self.class_name = list(self.test_label.columns)
			self.test_label = self.test_label.to_numpy()
	
	def __getitem__(self, index):
		if self.mode == 'train':
			x = self.train_seq_oht[index,...]
			y = self.train_label[index,...]
			z = self.train_DNA_shape[index,...]
		elif self.mode == 'valid':
			x = self.valid_seq_oht[index,...]
			y = self.valid_label[index,...]
			z = self.valid_DNA_shape[index,...]
		elif self.mode == 'test':
			x = self.test_seq_oht[index,...]
			y = self.test_label[index,...]
			z = self.test_DNA_shape[index,...]
	
		x = torch.from_numpy(x)
		y = torch.from_numpy(y)
		z = torch.from_numpy(z)
	
		x = x.type('torch.cuda.FloatTensor')
		y = y.type('torch.cuda.FloatTensor')
		z = z.type('torch.cuda.FloatTensor')

		return (x, y, z)
	
	def __len__(self):
		if self.mode == 'train':
			return self.train_label.shape[0]
		elif self.mode == 'valid':
			return self.valid_label.shape[0]
		elif self.mode == 'test':
			return self.test_label.shape[0]

def load_PlantBind_data(data_dir_path, batch_size, mode_):
	
	print('Message: loading %s data!' % mode_)
	dataset = PlantBindDataset(data_dir_path, mode = mode_)
	
	if mode_ == 'train':
		# 使用DistributedSampler
		data_sampler = DistributedSampler(dataset)
		data_loader = DataLoader(dataset=dataset, shuffle = True if not data_sampler else False,
								 sampler = data_sampler,
								 batch_size = batch_size)
		#data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	else:
		data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	
	return data_loader

if __name__ == "__main__":
	test_data = PlantBindDataset('./data', mode='test')
	x_test, y_test, threeD = test_data[:]
	#x_test_, threeD_ = data_format(x_test, threeD)
	#print(x_test.size(), y_test.size(), threeD.size())
	#print(x_test_.size(), threeD_.size())

	test_loader = load_PlantBind_data(data_dir_path='./data', batch_size=128, mode_='test')
	a,b,c = test_loader.dataset[:]
	print(len(test_loader.dataset))
	for batch in test_loader:
		x, y, z = batch
		#x, z = data_format(x, z)
		print('x size', x.size())
		print(x[1,:])
		print('*'*88)
		print('y size', y.size())
		print(y[1,:])
		print('*'*88)
		print('z size', z.size())
		print(z[1,:])
		break

