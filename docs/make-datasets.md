# Data processing methods

### ***1. DNA Sequence Data*** <a name="DNA_Sequence_Data"/>

**Input files**: (1) genome fasta file; (2) peak files in bed format.  
**Output file**: TFBSs sequence file  

1. Extract the sequence
```
seqtk subseq Arabidopsis_thaliana.TAIR10.dna.toplevel.fa TF-peaks.bed > model-input-seq.fa
```
2. Generate sequence file
```
import os
from Bio import SeqIO
import re

with open('model-input-seq.table', 'w') as f:
    for seq_record in SeqIO.parse('model-input-seq.fa', "fasta"):
        if re.match('^[ACGT]+$', str(seq_record.seq)) and len(str(seq_record.seq)) == 51:
            print(seq_record.id, seq_record.seq, sep='\t', file=f)
```
3. Generate input files for the model
```
cut -f 2 model-input-seq.table > sequence.table
```

### ***2. DNA Shape Data*** <a name="DNA_Shape_Data"/>
This section uses [translate.py](https://github.com/wenkaiyan-kevin/PlantBind/blob/main/src/translate.py) 
to generate shape information for DNA sequences.  
Here, the `model-input-seq.table` file generated in the previous step is used as input, 
and you should specify the value of the parameter `sample_num`(sample size).

```
import pandas as pd
import numpy as np
from translate  import seq_to_shape

sample_num = 598858
seq_len = 101

output = np.zeros(shape=(sample_num, 97, 14))

i = 0
for line in open('../../03-sequence-dataset/02-seqlen-101/model-input-seq.table','r'):
    print(i)
    line_list = line.strip('\n').split('\t')
    seq_name, seq = line_list[0], line_list[1]
    shape_info = np.array(seq_to_shape(seq, normalize=True).drop([1,2,seq_len-1,seq_len]))
    output[i,:,:] = shape_info
    i+=1

np.save("model-input-DNAshape-normalized-101.npy",output)
```




### ***3. Data Label*** <a name="Data_Label"/>


