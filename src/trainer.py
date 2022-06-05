import os
import csv
import math
import  numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from torch.optim import Adam, lr_scheduler

from utils import cal_loss_weight, loss_function, roc_auc_score, data_format

np.seterr(invalid='ignore')

def Performance_Monitor(y_pred, y_true):
	acc, auc, ap = 0, 0, 0
	num_task = len(y_pred)
	j = 0
	for i in range(num_task):
		y_true_ = y_true.cpu().numpy()[:,i]
		y_pred_ = y_pred[i].cpu().detach().numpy()
		y_pred_binary = np.array([0 if instance < 0.5 else 1 for instance in y_pred_])
		acc += np.mean(y_pred_binary == y_true_)
		try:
			auc += roc_auc_score(y_true_, y_pred_)
		except ValueError:
			pass
		try:
			if sum(y_true_) > 0:
				ap += average_precision_score(y_true_, y_pred_)
				j += 1
			else:
				pass
		except ValueError:
			pass
	return acc/num_task, auc/num_task, ap/(j+0.001)

def valid(model, valid_loader, loss_weight, args):
	with torch.no_grad():
		model.eval()
		valid_loss = 0
		metrics_dict = {"acc":0, "auc":0, "ap":0}

		for DNA_seq_valid, DNA_label_valid, DNA_shape_valid in valid_loader:
			x, y_true, z = DNA_seq_valid.cuda(), DNA_label_valid.cuda(), DNA_shape_valid.cuda()
			x, z = data_format(x, z)
			
			y_pred, attention = model(x, z)

			valid_loss += loss_function(y_pred=y_pred, y_true=y_true, loss_weight=loss_weight, ohem=args.OHEM, focal=args.focal_loss)
			
			acc, auc, ap = Performance_Monitor(y_pred=y_pred, y_true=y_true)
			metrics_dict['acc'] += acc
			metrics_dict["auc"] += auc
			metrics_dict["ap"] += ap

		num_samples = len(valid_loader.dataset)
		num_batches = len(valid_loader)

		valid_loss /= num_samples
		metrics_dict['acc'] /= num_batches
		metrics_dict['auc'] /= num_batches
		metrics_dict['ap'] /= num_batches

	return valid_loss, metrics_dict

def train(model, train_loader, valid_loader, args):

	if not os.path.exists(args.save_dir):
		print('Message: %s does not exist, create it now'%(args.save_dir))
		os.mkdir(args.save_dir)
	logfile = open(args.save_dir + '/log.csv', 'w')
	
	logwriter = csv.DictWriter(logfile, fieldnames=['epoch','train_loss','val_loss',
													'val_acc', 'val_auc', 'val_ap'])
	logwriter.writeheader()

	t0 = time()
	
	optimizer = Adam(model.parameters(), lr=args.lr)
	lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
	lr_decay  = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

	best_val_acc = 0
	best_val_loss = 50
	# 计算损失函数权重
	print('Message: weigh the loss of each class at the end of the network.')
	loss_weight_ = cal_loss_weight(train_loader.dataset) # dictionary

	# parepare weights parameters
	loss_weight = []
	for i in range(args.num_task):
		# initialize weights
		#loss_weight.append(torch.tensor(loss_weight_[i], requires_grad=True, device="cuda"))
		loss_weight.append(loss_weight_[i].clone().detach().requires_grad_(True).cuda())
	optimizer.zero_grad()
	print('Message: Begin Training!')
	for epoch in range(1,args.epochs+1):
		model.train()
		ti = time()
		training_loss = 0.0

		for i, (DNA_seq, DNA_label, DNA_shape) in enumerate(train_loader): ###
			x, y_true, z = DNA_seq.cuda(), DNA_label.cuda(), DNA_shape.cuda()
			x, z = data_format(x, z)

			y_pred, attention = model(x, z)
			#loss = loss_function(y_pred=y_pred, y_true=y_true, loss_weight=loss_weight, ohem=args.OHEM, focal=args.focal_loss)	
			loss = loss_function(y_pred=y_pred, y_true=y_true, loss_weight=loss_weight, ohem=args.OHEM, focal=args.focal_loss)	
			optimizer.zero_grad()

			# gradient clipping 通过Gradient Clipping，将梯度约束在一个范围内，这样不会使得梯度过大。
			clip_value = 1
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
			
			loss.backward()
			optimizer.step()
			training_loss += loss.item()

			if i == 1:
				#sanity check y_pred
				print("Message: Sanity Checking, at epoch%02d, iter%02d, y_pred is"%(epoch, i),
						torch.stack([y_pred[j][1].cpu().detach() for j in range(3)]).tolist())
				print("Message: Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'])
		
		training_loss_avg = training_loss / len(train_loader.dataset)
		lr_decay.step()
		
		#######compute validation loss and acc##########
		################################################
		val_loss, metrics_dict = valid(model, valid_loader, loss_weight, args)
		logwriter.writerow(dict(epoch=epoch, train_loss=training_loss_avg,
								val_loss=val_loss.detach().cpu().numpy(),
								val_acc=metrics_dict['acc'],
								val_auc=metrics_dict["auc"],
								val_ap=metrics_dict['ap']))
								
		print("===>Epoch %02d: train_loss=%.5f, val_loss=%.4f, val_acc=%.4f, val_auc=%.4f, val_ap=%.4f, time=%ds"
			  %(epoch,training_loss_avg,val_loss,metrics_dict["acc"],
			    metrics_dict["auc"],metrics_dict["ap"],time()-ti))
				  
		#! save model #根据验证集合保存一个最好的模型
		if metrics_dict['acc'] > best_val_acc and val_loss < best_val_loss:  #update best validation acc and save model
			best_val_acc = metrics_dict["acc"]
			best_val_loss = val_loss
			torch.save(model.module.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
			print("Message: best val_acc increased to %.4f" % best_val_acc)
	
	logfile.close()
	
	#训练完最后一个epoch 最后保存一个模型
	torch.save(model.module.state_dict(), args.save_dir + '/trained_model_%d_seqs.pkl' % args.length)
	print('Message: Trained model saved to \'%s/trained_model.pkl\'' % (args.save_dir))
	print('Message: Total time = %ds' % (time() - t0))
	print('-'*30 + 'End Training' + '-'*30)

	return model
