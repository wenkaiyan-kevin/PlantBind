import numpy as np

def highest_score(a,w):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
    """

    assert(len(a)>=w)

    best = -20000
    best_idx_start = 0
    best_idx_end =0
    for i in range(len(a)-w + 1):
        tmp = np.sum(a[i:i+w])
        if tmp > best:
            best = tmp
            best_idx_start = i
            best_idx_end = i + w - 1

    return best, best_idx_start, best_idx_end

def highest_x(a,w,p=1):
    """
    Inputs:
        a: a 1-D numpy array contains the scores of each position
        w: length of window to aggregate the scores
        p: length of padding when maximum sum of consecutive numbers are taken
    """

    lists = [{k:v for (k,v) in zip(range(len(a)),a)}]
    result = {}
    max_idx = len(a) -1
    count = 1
    condition = [True]
    while any(con is True for con in condition):
        starts = []
        ends = []
        bests = []

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())
            start_idx = idx[0]

            if len(values) >= w:
                highest, highest_idx_start, highest_idx_end = highest_score(values,w)
                starts.append(highest_idx_start+start_idx)
                ends.append(highest_idx_end+start_idx)
                bests.append(highest)

        best_idx = max(zip(bests, range(len(bests))))[1]   # calculate the index of maximum sum

        cut_value = bests[best_idx]

        if starts[best_idx] - p >=0:
            cut_idx_start = starts[best_idx] - p
        else:
            cut_idx_start = 0

        if ends[best_idx] + p <=max_idx:
            cut_idx_end = ends[best_idx] + p
        else:
            cut_idx_end = max_idx

        result[count] = (cut_value,starts[best_idx],ends[best_idx])


        copy = lists.copy()

        for ele in lists:
            values = list(ele.values())
            idx = list(ele.keys())

            start_idx, end_idx = idx[0], idx[-1]

            if len(values) < w:
                copy.remove(ele)
            else:
#                 print(cut_idx_start,cut_idx_end)
#                 print(start_idx,end_idx)
#                 print(values)
                if (cut_idx_end < start_idx) or (cut_idx_start > end_idx):

                    pass
                elif (cut_idx_start < start_idx) and (cut_idx_end >= start_idx):
                    copy.remove(ele)
                    values = values[cut_idx_end-start_idx+1:]
                    idx = idx[cut_idx_end-start_idx+1:]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

                elif (cut_idx_start >= start_idx) and (cut_idx_end <= end_idx):
                    copy.remove(ele)
                    values_1 = values[:cut_idx_start-start_idx]
                    idx_1 = idx[:cut_idx_start-start_idx]
                    ele_1 = {k:v for (k,v) in zip(idx_1,values_1)}

                    values_2 = values[cut_idx_end-start_idx+1:]
                    idx_2 = idx[cut_idx_end-start_idx+1:]
                    ele_2 = {k:v for (k,v) in zip(idx_2,values_2)}

                    if ele_1 != {}:
                        copy.append(ele_1)
                    if ele_2 != {}:
                        copy.append(ele_2)

                elif (cut_idx_start <= end_idx) and (cut_idx_end > end_idx):
                    copy.remove(ele)
                    values = values[:cut_idx_start-start_idx]
                    idx = idx[:cut_idx_start-start_idx]
                    ele = {k:v for (k,v) in zip(idx,values)}

                    if ele != {}:
                        copy.append(ele)

        lists = copy
#        print(lists)
        count = count + 1
        condition = [len(i)>=w for i in lists]
#        print(condition)

    return result

def helper(attn_data, nucleos, data_type, length,TF_name, download=False, w=5, k=3, p=1):
    """
    extract short seqs and relative scores
    """
    num_samples = attn_data.shape[0]
    results = []                                 # (score,start_idx,end_idx)
    for i in range(num_samples):
        result = highest_x(attn_data[i,:],w=w,p=p)
        results.append(result)

    short_seqs = []
    scores = []

    for j in range(num_samples):
        seq = results[j]
        new_dict ={}
        for i in range(1,k+1):
            start_idx = seq.get(i)[1]
            end_idx = seq.get(i)[2]
            score = seq.get(i)[0]
            # extract short seqs
            short_seq = nucleos.iloc[j][start_idx:end_idx+1]

            short_seqs.append(short_seq)
            scores.append(score)

            if download:
                with open('./%s_%s_%d_wid%d_top%d.csv'%(data_type,TF_name,length,w,k),'a') as file:

                    file.write(short_seq)
                    file.write('\n')

                with open('./%s_%s_%d_wid%d_top%d_score.csv'%(data_type,TF_name,length,w,k),'a') as file:

                    file.write(str(score))
                    file.write('\n')
    return short_seqs, scores

def cal_attention_every_class(attention_weights):
    length = attention_weights.shape[0]
    attention = np.zeros((1,length+2))
    for i in range(length+2):
        # unravel 3-mers attention
        if i == 0:
            attention[:,0] = attention_weights[0]
        elif i == 1:
            attention[:,1] = attention_weights[0] + attention_weights[1]
        elif i == length +1:
            attention[:,i] = attention_weights[i-2]
        elif i == length:
            attention[:,i] = attention_weights[i-2] + attention_weights[i-1]
        else:
            attention[:,i] = attention_weights[i-2]+attention_weights[i-1]+attention_weights[i]

    return attention

def cal_attention(total_attention_weights):
    """
    Unwarp the 3-mers inputs attention_weights and sum to single nucleotide
        Inputs: Attention weights shape [batch_size, length, num_class]
        Outputs: Unwarped Attention weights shape [batch_size, num_class, length+2]
    """
    num_class = total_attention_weights.shape[-1]
    length = total_attention_weights.shape[1] + 2
    num_samples = total_attention_weights.shape[0]
    total_attention = np.zeros((num_samples,num_class,length))
    for k in range(num_samples):
        tmp = []
        for i in range(num_class):
            tmp.append(cal_attention_every_class(total_attention_weights[k,:,i].detach().cpu().numpy()))
        tmp = np.concatenate(tmp,axis=0)

        total_attention[k,:] = tmp
    return total_attention



