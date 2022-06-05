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






### ***3. Data Label*** <a name="Data_Label"/>


