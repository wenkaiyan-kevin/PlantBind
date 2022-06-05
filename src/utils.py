import torch

import numpy as np
from math import sqrt
from sklearn.metrics import recall_score,precision_score,roc_auc_score,roc_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
import matplotlib as mpl

# 修改输入数据格式
def data_format(DNA_Seq, DNA_Shape):
	DNA_Seq_ = DNA_Seq.view(DNA_Seq.size(0), -1, 4).transpose(1, 2)
	
	zero_padding = torch.zeros(DNA_Shape.size(0), 14, 2).cuda()
	DNA_Shape = DNA_Shape.transpose(1, 2)
	DNA_Shape_ = torch.cat([zero_padding, DNA_Shape, zero_padding], dim=2)

	return DNA_Seq_.cuda(), DNA_Shape_.cuda()
# 评估模型性能
def precision_multi(y_true,y_pred):
	
	"""
		Input: y_true, y_pred with shape: [n_samples, n_classes]
		Output: example-based precision
	"""
	n_samples = y_true.shape[0]
	result = 0
	for i in range(n_samples):
		if not (y_pred[i] == 0).all():
			true_posi = y_true[i] * y_pred[i]
			n_true_posi = np.sum(true_posi)
			n_pred_posi = np.sum(y_pred[i])
			result += n_true_posi / n_pred_posi
	return result / n_samples
def recall_multi(y_true,y_pred):
	"""
		Input: y_true, y_pred with shape: [n_samples, n_classes]
		Output: example-based recall

	"""
	n_samples = y_true.shape[0]
	result = 0
	for i in range(n_samples):
		if not (y_true[i] == 0).all():
			true_posi = y_true[i] * y_pred[i]
			n_true_posi = np.sum(true_posi)
			n_ground_true = np.sum(y_true[i])
			result += n_true_posi / n_ground_true
	return result / n_samples
def f1_multi(y_true,y_pred):
	"""
		Input: y_true, y_pred with shape: [n_samples, n_classes]
		Output: example-based recall
	"""
	n_samples = y_true.shape[0]
	result = 0
	for i in range(n_samples):
		if not ((y_true[i] == 0).all() and (y_pred[i] == 0).all()):
			true_posi = y_true[i] * y_pred[i]
			n_true_posi = np.sum(true_posi)
			n_ground_true = np.sum(y_true[i])
			n_pred_posi = np.sum(y_pred[i])
			f1 = 2*(n_true_posi) / (n_ground_true+n_pred_posi)
			result += f1
	return result / n_samples
def hamming_loss(y_true,y_pred):
		"""
			Input: y_true, y_pred with shape: [n_samples, n_classes]
			Output: hamming loss
		"""
		n_samples = y_true.shape[0]
		n_classes = y_true.shape[1]
		loss = 0
		for i in range(n_samples):
			xor = np.sum((y_true[i] + y_pred[i]) % 2)
			loss += xor / n_classes
		return loss / n_samples
def cal_metrics(model_out, label, plot, args, plot_name="TFs_315"):
    num_task = model_out.shape[1]
    threshold_list = [0.5 for i in range(num_task)]                              # thresholds standard
    metrics_b = {'SN_b':[],'SP_b':[],'Recall_b':[],'Precision_b':[],'ACC_b':[],'MCC_b':[],'AUC_b':[],'AP_b':[]}
    metrics_m = {'SN_m':[],'SP_m':[],'Recall_m':[],'Precision_m':[],'ACC_m':[],'MCC_m':[],'AUC_m':[],'AP_m':[]}
    # Compute ROC curve and ROC area for each class
    fpr_b,tpr_b = dict(), dict()
    fpr_m,tpr_m = dict(), dict()
    precisions_b, recalls_b = dict(), dict()
    precisions_m, recalls_m = dict(), dict()
    label = label.cpu().numpy()
    Y_pred = np.zeros(label.shape)
    threshold_list_b = []
    threshold_list_m = []
    for i in range(num_task):
        ######################################  二分类  ########################################
        # 先把正负样本找出来做成新的矩阵，用来做二分类任务
        ## 提取数据
        pos_index = [i for i,j in enumerate(label[:,i]) if (j==1)]
        neg_index = list(np.where(np.sum(label, axis=1)==0)[0])
        if len(pos_index) < len(neg_index):
            neg_index = neg_index[0:len(pos_index)]
        index = pos_index + neg_index
        lable_b = label[index,:]
        model_out_b = model_out[index,:]
        #######################
        y_true = lable_b[:,i]
        y_pred = np.clip(model_out_b[:,i],0,1) # 将输出结果限制在0和1之间
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in list(y_pred)])
        y_score = model_out_b[:,i]

        auc_b = roc_auc_score(y_true, y_score) 
        ap_b = average_precision_score(y_true, y_score)

        fpr_b[i], tpr_b[i], thresholds_b = roc_curve(y_true, y_score)
        precisions_b[i], recalls_b[i], _ = precision_recall_curve(y_true, y_score)
        gmeans_b = np.sqrt(tpr_b[i] * (1-fpr_b[i]))

        ix = np.argmax(gmeans_b)
        print('B Task:%f, Best Threshold=%f, G-Mean=%.3f' % (i,thresholds_b[ix], gmeans_b[ix]))

        best_threshold = thresholds_b[ix]
        y_pred_new = np.array([0 if instance < best_threshold else 1 for instance in list(y_score)])
        threshold_list_b.append(best_threshold) #收集新阈值
        # multiclass based confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()  # 扁平化
        pp = tp + fn 
        pn = tn + fp 
        sensitivity = tp / pp
        specificity = tn / pn
        recall = sensitivity
        precision = tp / (tp + fp)
        acc = (tp+tn) / (pp+pn)
        mcc = acc / np.sqrt((1+(fp-fn)/pp)*(1+(fn-fp)/pn))
        # update dictionary
        metrics_b['SN_b'].append(sensitivity)
        metrics_b['SP_b'].append(specificity)
        metrics_b['Recall_b'].append(recall)
        metrics_b['Precision_b'].append(precision)
        metrics_b['ACC_b'].append(acc)
        metrics_b['MCC_b'].append(mcc)
        metrics_b['AUC_b'].append(auc_b)
        metrics_b['AP_b'].append(ap_b)
        ######################################  多分类  ########################################
        y_true = label[:,i] #纵向的11428个
        y_pred = np.clip(model_out[:,i],0,1) # 将输出结果限制在0和1之间 #横向的11428
        y_pred = np.array([0 if instance < threshold_list[i] else 1 for instance in list(y_pred)])
        Y_pred[:,i] = y_pred  #保存第一次处理过的预测结果0/1
        y_score = model_out[:,i] #预测的小数点 #横向的11428个

        auc_m = roc_auc_score(y_true,y_score)
        ap_m = average_precision_score(y_true,y_score)

        fpr_m[i], tpr_m[i], thresholds_m = roc_curve(y_true, y_score)
        precisions_m[i], recalls_m[i], _ = precision_recall_curve(y_true, y_score)

        gmeans_m = np.sqrt(tpr_m[i] * (1-fpr_m[i]))

        # locate the index of the largest g-mean
        ix = np.argmax(gmeans_m)
        print('M Best Threshold=%f, G-Mean=%.3f' % (thresholds_m[ix], gmeans_m[ix]))
        best_threshold = thresholds_m[ix]
        y_pred_new = np.array([0 if instance < best_threshold else 1 for instance in list(y_score)])
        threshold_list_m.append(best_threshold) #收集新阈值
        Y_pred[:,i] = y_pred_new  #保存处理过的预测结果0/1

        # multiclass based confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_new).ravel()
        pp = tp+fn #正样本总数量
        pn = tn+fp #负样本总数量
        sensitivity = tp / pp
        specificity = tn / pn
        recall = sensitivity
        precision = tp / (tp + fp)
        acc_m = (tp+tn) / (pp+pn)
        mcc = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        # update dictionary
        metrics_m['SN_m'].append(sensitivity)
        metrics_m['SP_m'].append(specificity)
        metrics_m['Recall_m'].append(recall)
        metrics_m['Precision_m'].append(precision)
        metrics_m['ACC_m'].append(acc)
        metrics_m['MCC_m'].append(mcc)
        metrics_m['AUC_m'].append(auc_m)
        metrics_m['AP_m'].append(ap_m)
    
    np.savetxt(args.save_dir+'/Gmean_threshold_m.txt', threshold_list_m, fmt='%f')
    np.savetxt(args.save_dir+'/Gmean_threshold_b.txt', threshold_list_b, fmt='%f')

    precision_multi_ = precision_multi(label, Y_pred)
    recall_multi_ = recall_multi(label, Y_pred)
    f1_multi_ = f1_multi(label, Y_pred)
    hamming_loss_ = hamming_loss(label, Y_pred)

    print("precision multi: %f"%(precision_multi_))
    print("recall multi: %f"%(recall_multi_))
    print("f1 multi: %f"%(f1_multi_))
    print("hamming loss: %f"%(hamming_loss_))

    if plot:
        # modifying parameters for plot
        golden_mean = (sqrt(5)-1.0)/2.0 #used for size=
        fig_width = 6 # fig width in inches
        fig_height = fig_width*golden_mean # fig height in inches
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['axes.titlesize'] = 10
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['font.family'] = 'serif'
        lw = 0.5
        # roc curve
        fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))
        # PR curve
        fig_2, axes_2 = plt.subplots(nrows=1,ncols=2,figsize=(fig_width*2+0.7,fig_height+0.1))

        for i in range(num_task):
            axes[0].plot(fpr_b[i], tpr_b[i], color='#FF6A53',lw=lw)
            axes[0].plot([0, 1], [0, 1], 'k--', lw=lw)
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.0])
            axes[0].tick_params(axis='x',which='both',top=False)
            axes[0].tick_params(axis='y',which='both',right=False)
            axes[0].set_aspect('equal', adjustable='box')
            axes[0].set_xlabel('False Positive Rate')
            axes[0].set_ylabel('True Positive Rate')
            axes[0].set_title('ROC curves (binary)')

            axes_2[0].plot(recalls_b[i], precisions_b[i], color='#FF6A53',lw=lw)
            axes_2[0].plot([0, 1], [0.5, 0.5], 'k--', lw=lw)
            axes_2[0].set_xlim([0.0, 1.0])
            axes_2[0].set_ylim([0.45, 1.0])
            axes_2[0].tick_params(axis='x',which='both',top=False)
            axes_2[0].tick_params(axis='y',which='both',right=False)
            xmin, xmax = axes_2[0].get_xlim()
            ymin, ymax = axes_2[0].get_ylim()
            axes_2[0].set_aspect(abs((xmax-xmin)/(ymax-ymin)), adjustable='box')
            axes_2[0].set_xlabel('Recall')
            axes_2[0].set_ylabel('Precision')
            axes_2[0].set_title('PR curves (binary)')

            axes[1].plot(fpr_m[i], tpr_m[i], color='#FFAD13',lw=lw)
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.0])
            axes[1].tick_params(axis='x',which='both',top=False)
            axes[1].tick_params(axis='y',which='both',right=False)
            axes[1].set_aspect('equal', adjustable='box')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC curves (multiple)')

            axes_2[1].plot(recalls_m[i], precisions_m[i], color='#FFAD13', lw=lw)
            axes_2[1].set_xlim([0.0, 1.0])
            axes_2[1].set_ylim([0.0, 1.0])
            axes_2[1].tick_params(axis='x',which='both',top=False)
            axes_2[1].tick_params(axis='y',which='both',right=False)
            xmin, xmax = axes_2[1].get_xlim()
            ymin, ymax = axes_2[1].get_ylim()
            axes_2[1].set_aspect(abs((xmax-xmin)/(ymax-ymin)), adjustable='box')
            axes_2[1].set_xlabel('Recall')
            axes_2[1].set_ylabel('Precision')
            axes_2[1].set_title('PR curves (multiple)')

        axes[1].plot([0, 1], [0, 1], 'k--', lw=lw)
        axes_2[1].plot([0, 1], [0.04, 0.04], 'k--', lw=lw)

        # Put a legend to the right of the current axis
        axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)
        axes_2[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1),borderaxespad=0.,frameon=False)

        fig.tight_layout()
        fig_2.tight_layout()

        fig.savefig(args.save_dir+'/roc_prc_curve_%s.pdf'%(plot_name))
        fig_2.savefig(args.save_dir+'/precision_recall_curve_%s.pdf'%(plot_name))

        print('Successfully save figure to %s/roc_prc_curve_%s.pdf'%(args.save_dir, plot_name))
        print('Successfully save figure to %s/precision_recall_curve_%s.pdf'%(args.save_dir, plot_name))


    return metrics_b,metrics_m
# 计算误差
def cal_loss_weight(dataset, beta=0.99999):
	# weigh the loss of each class at the end of the network
	data, label, threeD = dataset[:]
	total_example = label.shape[0]
	num_task = label.shape[1]
	labels_dict = dict(zip(range(num_task),[torch.sum(label[:,i]) for i in range(num_task)]))
	keys = labels_dict.keys()
	class_weight = dict()

	# Class-Balanced Loss Based on Effective Number of Samples
	for key in keys:
		effective_num = 1.0 - beta**labels_dict[key]
		weights = (1.0 - beta) / effective_num
		class_weight[key] = weights

	weights_sum = sum(class_weight.values())

	# normalizing weights
	for key in keys:
		class_weight[key] = class_weight[key] / weights_sum * num_task

	return class_weight
def loss_function(y_pred, y_true, loss_weight=None, ohem=False, focal=False):
	num_task = y_true.shape[-1]
	num_examples = y_true.shape[0]
	k = 0.7
	def binary_cross_entropy(x, y, focal=False):
		alpha = 0.75
		gamma = 2

		pt = x * y + (1 - x) * (1 - y)
		at = alpha * y + (1 - alpha)* (1 - y)

		# focal loss
		if focal:
			loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
		else:
			loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
		return loss
	# loss = nn.BCELoss(reduction='sum') fail to double backwards
	loss_output = torch.zeros(num_examples).cuda()
	for i in range(num_task):
		if loss_weight:
			out = loss_weight[i]*binary_cross_entropy(y_pred[i], y_true[:,i], focal)
			loss_output += out
		else:
			loss_output += binary_cross_entropy(y_pred[i], y_true[:,i], focal)
	# Online Hard Example Mining
	if ohem:
		val, idx = torch.topk(loss_output,int(k*num_examples)) #topk:取数组的前k个元素进行排序
		loss_output[loss_output<val[-1]] = 0

	loss = torch.sum(loss_output)
	return loss

