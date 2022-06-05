import os
import argparse

import torch
from torch import nn
import torch.distributed as dist

from model import PlantBind
from dataset_PlantBind import PlantBindDataset, load_PlantBind_data
from trainer import train
from tester import test

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
	# setting the hyper parameters
	parser = argparse.ArgumentParser(description="PlantBind Network for predicting TFBS.")
	parser.add_argument('--inputs',default='./data',type=str) #  <<-------
	parser.add_argument('--epochs', default=100, type=int)  #50
	parser.add_argument('--multi_GPU',default=False, type=str2bool, nargs='?', help='multi-gpu')
	parser.add_argument('--batch_size', default=1024, type=int) #128
	parser.add_argument('--length', default=101,type=int)#  <<---------------------------
	parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
	parser.add_argument('--lrf', type=float, default=0.01)
	parser.add_argument('--save_dir', default='./output')#  <<-------
	parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
	#parser.add_argument('-device', type=str, default='cuda:0', help='device id')
	
	parser.add_argument('--num_task',default=315, type=int)  #  <<---------------------------

	parser.add_argument('--mode',default='train',help="Set the model to train or not") #  <<------------------
	
	parser.add_argument('--OHEM',default=False,type=str2bool,nargs='?')       
	parser.add_argument('--focal_loss',default=False,type=str2bool,nargs='?')

	parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')

	args = parser.parse_args()

	if args.multi_GPU:
		#! GPU并行
		local_rank = args.local_rank
		if local_rank != -1:
			dist_backend = 'nccl'
			dist.init_process_group(backend=dist_backend) #  初始化进程组，同时初始化 distributed 包
		device = local_rank if local_rank != -1 else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
		torch.cuda.set_device(local_rank)  # 配置每个进程的gpu
		
		# Seq_CNN & 3D_CNN -> LSTM + Attention -> FC -> output
		model = PlantBind(seq_len=args.length, num_task=args.num_task)
		model.to(device) #  封装之前要把模型移到对应的gpu
		if torch.cuda.device_count() > 1: 
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
		
		# print(model)
		print('-'*30 + 'Program starts running!' + '-'*30)
		if local_rank == 0:
			# 创建保存结果的文件夹	
			if os.path.exists(args.save_dir):
				print('Message: the save dir is existed!')
			else:
				os.makedirs(args.save_dir)
				print('Message: success create save dir!')

	else:
		# 创建保存结果的文件夹
		print('-'*30 + 'Program starts running!' + '-'*30)
		if os.path.exists(args.save_dir):
			print('Message: the save dir is existed!')
		else:
			os.makedirs(args.save_dir)
			print('Message: success create save dir!')

		model = PlantBind(seq_len=args.length, num_task=args.num_task)
		model.cuda()

	# train or test
	if args.mode=='train':
		# load data # train and valid
		print('Message: Loading training data!')  #训练数据
		train_loader = load_PlantBind_data(args.inputs, batch_size=args.batch_size, mode_='train')

		print('Message: Loading vaildating data!')  #验证数据
		valid_loader = load_PlantBind_data(args.inputs, batch_size=args.batch_size, mode_='valid')

		print('Message: Complete data loading!')
		train(model, train_loader, valid_loader, args)	
	elif args.mode=='test':
		print('Message: Loading testing data!') #测试数据
		test_data = PlantBindDataset(args.inputs, mode=args.mode)
		torch.cuda.empty_cache()
		print('Message: Loading saved model!') #加载模型
		model.load_state_dict(torch.load(args.save_dir + '/trained_model_{}_seqs.pkl'.format(args.length))) #读取最好的模型
		test(model=model, test_data=test_data, args=args)
	else:
		print('Message: You should select the "train" or "test" mode!')

if __name__ == "__main__":
	main()