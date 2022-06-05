import torch
from torch import nn
import numpy as np

class BahdanauAttention(nn.Module):
	"""
	input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
									h_n: (num_directions, batch_size, units)
	return: (batch_size, num_task, units)
	"""
	def __init__(self,in_features, hidden_units,num_task):
		super(BahdanauAttention,self).__init__()
		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

	def forward(self, hidden_states, values):
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)

		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
		#print(attention_weights.shape,values.shape)
		context_vector = torch.matmul(values,attention_weights)
		context_vector = torch.transpose(context_vector,1,2)
		return context_vector, attention_weights

class PlantBind(nn.Module):
	"""
		Seq_CNN & 3D_CNN -> LSTM + Attention -> FC -> output
	"""
	def __init__(self, seq_len, num_task=None):
		self.num_task = num_task
		self.seq_len = seq_len
		if self.seq_len == 51:
			in_feature = 11
		elif self.seq_len == 101:
			in_feature = 23
		elif self.seq_len == 201:
			in_feature = 48
		else:
			print('Message: seq len must be 51/101/201')
		super(PlantBind, self).__init__()
		self.CNN1 = nn.Sequential(
						nn.Conv1d(in_channels=4,out_channels=64,kernel_size=13,stride=2,padding=0),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.MaxPool1d(kernel_size=2,padding=1),
						nn.Linear(in_features=in_feature,out_features=self.seq_len))   # [b, 128, 101] 如果conv1 kernel_size=7，in_features=25

		self.CNN2 = nn.Sequential(
						nn.Conv1d(in_channels=14,out_channels=64,kernel_size=13,stride=2,padding=0),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
						nn.ReLU(),
						nn.Dropout(p=0.2),
						nn.MaxPool1d(kernel_size=2,padding=1),
						nn.Linear(in_features=in_feature,out_features=self.seq_len))   # [b, 128, 101]

		self.BiLSTM = nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=True)
		self.Attention = BahdanauAttention(in_features=256,hidden_units=10,num_task=num_task)
		for i in range(num_task):
			setattr(self, "FC%d" %i, nn.Sequential(
									   nn.Linear(in_features=256,out_features=64),
									   nn.ReLU(),
									   nn.Dropout(),
									   nn.Linear(in_features=64,out_features=1),
									   nn.Sigmoid()))
	
	def forward(self,x_onehot, x_3D):
		
		x1 = self.CNN1(x_onehot) #[64, 128, 12]
		x2 = self.CNN2(x_3D)	  #[64, 128, 12]
		
		batch_size1, features1, seq_len1 = x1.size()  #[64, 128, 101]
		x1 = x1.view(batch_size1,seq_len1, features1) #[64, 101, 128]
		
		batch_size2, features2, seq_len2 = x2.size()  #[64, 128, 101]
		x2 = x2.view(batch_size2,seq_len2, features2) #[64, 101, 128]

		x = x1 + x2  #[64,101,128]
		
		output, (h_n, c_n) = self.BiLSTM(x)
		
		h_n = h_n.view(batch_size1, output.size()[-1]) # pareprae input for Attention
		context_vector, attention_weights = self.Attention(h_n,output) # Attention (batch_size, num_task, unit)
		outs = []
		for i in range(self.num_task):
			FClayer = getattr(self, "FC%d"%i)
			y = FClayer(context_vector[:,i,:])
			y = torch.squeeze(y, dim=-1)
			outs.append(y)
		return outs, attention_weights

if __name__ == '__main__':
	model = PlantBind(seq_len=51,num_task=315)
	x_onehot = torch.randn(size=(8, 4, 51))
	x_3D = torch.randn(size=(8, 14, 51))
	out, attn = model(x_onehot, x_3D)
	print(out[0])
	print(attn.size())
	y_pred_ = [] #保存所有的y_pred
	for i in range(len(out)):
		y_pred_.append(list(out[i].cpu().detach().numpy()))
	print(np.array(y_pred_).shape)
	y_pred_T= np.array(y_pred_).T
	print(y_pred_T[:,0])
