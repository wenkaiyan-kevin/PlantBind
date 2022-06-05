import json
import numpy as np
import torch

from utils import cal_metrics, data_format

def test(model, test_data, args):

    DNA_seq_test, DNA_label_test, DNA_shape_test = test_data[:]
    DNA_seq_test, DNA_shape_test = data_format(DNA_Seq=DNA_seq_test, DNA_Shape=DNA_shape_test)
    x, y_true, z = DNA_seq_test, DNA_label_test, DNA_shape_test

    model.eval()
    try:
        y_pred, attention = model(x.cuda(), z.cuda())
    except RuntimeError: #如果上一行有报错即执行这一行
        print('Catch RuntimeError, parepare to batch the test set'+ '-'* 50)
        batch_size = 1024
        num_iter = x.shape[0] // batch_size
        y_true = y_true[0:num_iter*batch_size,...]
        x_test_tem = x[0:1*batch_size,...]
        z_test_tem = z[0:1*batch_size,...]
        y_pred, attention = model(x_test_tem.cuda(), z_test_tem.cuda())
        for i in range(1, num_iter): # 从1开始
            x_test_tem = x[i*batch_size:(i+1)*batch_size,...]
            z_test_tem = z[i*batch_size:(i+1)*batch_size,...]
            y_pred_tem, attn_tem = model(x_test_tem.cuda(), z_test_tem.cuda())
            for j in range(args.num_task):#0，1，2，3....11
                y_pred[j] = torch.cat((y_pred[j].cpu().detach(),y_pred_tem[j].cpu().detach()),dim=0)

    class_names = test_data.class_name

    #保存所有预测结果/小数点 ,并改变为label的形状
    y_pred_ = [] #保存所有的y_pred [315, sample_num]
    for i in range(len(y_pred)):
        y_pred_.append(list(y_pred[i].cpu().detach().numpy()))
    y_pred_T= np.array(y_pred_).T #[sample_num, 315] # 转换之后的预测结果
    np.savetxt(args.save_dir+'/DNA_lable_True.txt', y_true.cpu().detach().numpy(),fmt='%d') #整数
    np.savetxt(args.save_dir+'/DNA_lable_Pred.txt', y_pred_T,fmt='%f')
    # evaluate the model
    metrics_b, metrics_m = cal_metrics(model_out=y_pred_T, label=y_true, plot=True, args=args)

    print('-'*35+"End Testing"+"-"*35)
    print()
    print('-'*35+"Save Result"+"-"*35)

    #! 保存metries-> pf.json

    metrics_b_save_path = "%s/metrics_b.json" %(args.save_dir)
    metrics_m_save_path = "%s/metrics_m.json" %(args.save_dir)


    with open(metrics_b_save_path,'w') as fp:
        json.dump(metrics_b, fp)
    with open(metrics_m_save_path,'w') as fp:
        json.dump(metrics_m, fp)
    
    print("Storing performance results to %s" %(args.save_dir))

    
