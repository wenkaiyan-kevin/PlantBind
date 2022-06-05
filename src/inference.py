import os
import torch
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve


from model import PlantBind
from utils import data_format

## 读取数据
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
    def __init__(self, data_dir_path):
        self.data_dir = data_dir_path 
        
        # 读取序列信息		
        self.seq = pd.read_table(os.path.join(self.data_dir,"sequence.table"),header=None)
        self.seq_oht = np.array(Onehot(self.seq)) #(seqnum, 404)
        # 读取序列结构信息
        self.DNA_shape = np.load(os.path.join(self.data_dir,"DNA_shape.npy")) #(seqnum, 97, 14)

    def __getitem__(self, index):

        x = self.seq_oht[index,...]
        z = self.DNA_shape[index,...]

        x = torch.from_numpy(x)
        z = torch.from_numpy(z)

        x = x.type('torch.cuda.FloatTensor')
        z = z.type('torch.cuda.FloatTensor')

        return (x, z)

    def __len__(self):
        return self.seq_oht.shape[0]


# setting the parameters
length = 101
num_task = 315
data_path = '/home/wkyan/ywk_lab/03_PlantBind/02-model-evaluate-data/02-test/14'
model_weights = '/home/wkyan/ywk_lab/03_PlantBind/04-training/05-Kfold/10-Kfold-10/output/trained_model_101_seqs.pkl'
#model_weights = '/home/wkyan/ywk_lab/03_PlantBind/04-training/02-seq-101/01-bs-1024-lr-0.01/output/trained_model_101_seqs.pkl'
data_type = 'test'
TF_index = 203-1

torch.cuda.empty_cache()

print('Message: Loading testing data!') #测试数据
test_data = PlantBindDataset(data_path)

# ##开始程序##
model = PlantBind(seq_len=length, num_task=num_task)
model.cuda()

print('Message: Loading saved model!') #加载模型
model.load_state_dict(torch.load(model_weights)) #读取最好的模型
#model.eval()

x, z = test_data[:]
X, z = data_format(DNA_Seq=x, DNA_Shape=z)
print(X.size(), z.size())

#pos_num = X.size(0)//2
#neg_num = X.size(0) - pos_num
#y = np.concatenate((np.ones(pos_num),np.zeros(neg_num)))

pos_num = X.size(0)
y = np.ones(pos_num)
#print(y.shape)
#print(y)
#y = pd.read_table(data_path+"/label.txt", header=0)
#y = y.to_numpy()[:,0]  #(seqnum, 315)


try:
    y_pred, attention = model(X.cuda(), z.cuda())
    y_true = y
except RuntimeError: #如果上一行有报错即执行这一行
    print('Catch RuntimeError, parepare to batch the test set')
    batch_size = 1024
    num_iter = X.shape[0] // batch_size
    y_true = y[0:num_iter*batch_size,...]
    
    x_test_tem = X[0:1*batch_size,...]
    z_test_tem = z[0:1*batch_size,...]
    y_pred, attention = model(x_test_tem.cuda(), z_test_tem.cuda())
    for i in range(1, num_iter): # 从1开始
        x_test_tem = X[i*batch_size:(i+1)*batch_size,...]
        z_test_tem = z[i*batch_size:(i+1)*batch_size,...]
        y_pred_tem, attn_tem = model(x_test_tem.cuda(), z_test_tem.cuda())
        attention = torch.cat((attention.cpu().detach(), attn_tem.cpu().detach()),dim=0)
        for j in range(num_task):#0，1，2，3....11
            y_pred[j] = torch.cat((y_pred[j].cpu().detach(),y_pred_tem[j].cpu().detach()),dim=0)

print(attention.size())
# ###
y_pred_ = [] #保存所有的y_pred [315, sample_num]
for i in range(len(y_pred)):
    y_pred_.append(list(y_pred[i].cpu().detach().numpy()))
y_pred_T= np.array(y_pred_).T #[sample_num, 315] # 转换之后的预测结果

print(y_pred_T.shape)

# ###
print('Message: Loading thresholds!')
thresholds = np.loadtxt('/home/wkyan/ywk_lab/03_PlantBind/04-training/05-Kfold/10-Kfold-10/output/Gmean_threshold_b.txt').tolist()
#thresholds = np.loadtxt('/home/wkyan/ywk_lab/03_PlantBind/04-training/02-seq-101/01-bs-1024-lr-0.01/output/Gmean_threshold_m.txt').tolist()
#f = open('output-38.txt','w')
best = 0
for TF_index in range(315):
    best_threshold = thresholds[TF_index]
    #print("best_threshold:",best_threshold)
    y_pred_new = [0 if instance < best_threshold else 1 for instance in list(y_pred_T[:,TF_index])]
    pred = np.sum(y_pred_new)
    #print(TF_index+1,pred,sep='\t',file=f)
    print(TF_index+1,pred)
    if pred > best:
       print("best:",TF_index+1,pred)
       best = pred
    #print(y_pred_T[:,TF_index][0:100])

#for TF_index in range(315):
#    auc_b = roc_auc_score(y_true, y_pred_T[:,TF_index]) 
#    ap_b = average_precision_score(y_true, y_pred_T[:,TF_index])
#    print('auc_b', auc_b)
#    print('ap_b', ap_b)


