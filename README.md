# PlantBind: An attention-based multi-label neural networks for predicting transcription factor binding sites in plants
##   Introduction
Identify of transcription factor binding sites (TFBSs) are essential for the analysis of gene regulation. Here, we present **PlantBind**, a method for the integrated prediction and interpretation of TFBS events from DNA sequences and DNA shape profiles. Built upon an attention-based multi-label deep learning framework, PlantBind not only simultaneously predicts the binding sites of 315 TFs. As shown in **Figure 1**, the hybrid model consists of data processing module, embedding module, feature extraction module, and multi-label output module.

<p align="center">
<img src="https://github.com/wenkaiyan-kevin/PlantBind/blob/main/images/flow_chart.png" width = "600" height = "300" >
</p>  
<p align="center">Fig. 1 The model workflow</p>  

The model provides researchers with tools to:

1. Identify the TFBSs of transcription factors.
2. Identify DNA-binding motfi of transcription factors

## Tutorials
These are a work in progress, so forgive incompleteness for the moment. If there's a task that you're interested in that I haven't included, feel free to post it as an Issue at the top.

### 1. Software Requirements
We recommend that you use [conda](https://docs.conda.io/en/latest/) to install all of the following software.

***software list***
- python      v3.8.12
- pytorch     v1.10.2
- numpy       v1.16.4
- pandas      v1.4.1
- sklearn     v1.0.2
- scipy       v1.5.3
- matplotlib  v3.5.1

### 2. Data information and processing
In this part, we will first introduce the **data information** used in this model, then introduce the training and predicting **data formats**, and finally introduce how to create a data set that meets the model requirements for prediction.  
All data is in the [data directory](https://github.com/wenkaiyan-kevin/PlantBind/tree/main/data):
- **Ath-TF-peaks**: the TFBS peak info of 315 Ath TFs, and one neg.bed file
- **Maize-TF-peaks**: the TFBS peak info of 4 maize TFs for trans-specise
- **model**: The file that holds the model, which can be loaded to predict new datasets

#### 2.1 Training and Predicting data formats
For training, the data mainly consists of three files: (1)DNA sequence file; (2)DNA shape file; (3)data label file  
For predicting, the data mainly consists of three files: (1)DNA sequence file; (2)DNA shape file  

- [Data Format Details Introduction](docs/data_format.md)
  - [DNA Sequence File](docs/data_format.md#DNA_Sequence_File)
  - [DNA Shape File](docs/data_format.md#DNA_Shape_File)
  - [Data Label File](docs/data_format.md#Data_Label_File)

#### 2.2 Construction of the dataset
Next, we will mainly introduce how to create the files mentioned above.
- [Data processing methods](docs/make-datasets.md)
  - [DNA Sequence Data](docs/make-datasets.md#DNA_Sequence_Data)
  - [DNA Shape Data](docs/make-datasets.md#DNA_Shape_Data)
  - [Data Label](docs/make-datasets.md#Data_Label)

### 3. Train and Test the model
- **Training**  
**Input:** `train_sequence.table`,`train_DNA_shape.npy`,`train_label.txt`.  
All data files need to be placed in the same folder before starting training, such as `data_folder`
**Output:** `trained_model_101_seqs.pkl`  

```
python -m torch.distributed.launch --nproc_per_node=8 \
        main.py --inputs data_folder/ --length 101 \
        --OHEM True --focal_loss True \
        --batch_size 1024 --lr 0.01 \
        --multi_GPU True
```
- **Testing**  
**Input:** `test_sequence.table`,`test_DNA_shape.npy`,`test_label.txt`.   
All data files need to be placed in the same folder before starting testing, such as `data_folder`  
**Model:**`trained_model_101_seqs.pkl`  
**Output:** `metrics_b.json`,`metrics_m.json`,`Gmean_threshold_b.txt`,`Gmean_threshold_m.txt`,`precision_recall_curve_TFs_315.pdf`,`roc_prc_curve_TFs_315.pdf`  

```
python main.py --inputs data_folder/ --length 101 --mode test
```
### 4. Use the model to predict new data

**Input:** `sequence.table`,`DNA_shape.npy`
**Output:** `accuracy`
The analysis is done by using the [inference.py](https://github.com/wenkaiyan-kevin/PlantBind/blob/main/src/inference.py) script and modifying the relevant parameters.



