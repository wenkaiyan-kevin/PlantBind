# PlantBind: An attention-based multi-label neural networks for predicting transcription factor binding sites in plants
###   Introduction
Identify of transcription factor binding sites (TFBSs) are essential for the analysis of gene regulation. Here, we present **PlantBind**, a method for the integrated prediction and interpretation of TFBS events from DNA sequences and DNA shape profiles. Built upon an attention-based multi-label deep learning framework, PlantBind not only simultaneously predicts the binding sites of 315 TFs. As shown in **Figure 1**, the hybrid model consists of data processing module, embedding module, feature extraction module, and multi-label output module.

<p align="center">
<img src="https://github.com/wenkaiyan-kevin/PlantBind/blob/main/images/flow_chart.png" width = "600" height = "300" >
</p>  
<p align="center">Fig. 1 The model workflow</p>  

The model provides researchers with tools to:

1. Identify the TFBSs of transcription factors.
2. Identify DNA-binding motfi of transcription factors

### Tutorials
These are a work in progress, so forgive incompleteness for the moment. If there's a task that you're interested in that I haven't included, feel free to post it as an Issue at the top.

#### Software Requirements
We recommend that you use conda to install all of the following software

***software list***
- python      v3.6.8
- tensorflow  v1.14.0
- numpy       v1.16.4
- cudatoolkit v9.2
- cudnn       v7.6.0



#### The training and testing process of our model

