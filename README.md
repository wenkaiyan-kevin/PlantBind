# PlantBind: An attention-based multi-label neural networks for predicting transcription factor binding sites in plants
###   Introduction
Identify of transcription factor binding sites (TFBSs) are essential for the analysis of gene regulation. Accurate prediction of TFBSs is critical, because it is infeasible to assay all TFs in all sequenced eukaryotic genomes. Although many methods have been proposed for the identification of TFBSs in humans, the research development in the field of plants is lagging behind. Here, we present PlantBind, a method for the integrated prediction and interpretation of TFBS events from DNA sequences and DNA shape profiles. Built upon an attention-based multi-label deep learning framework, PlantBind not only simultaneously predicts the binding sites of 315 TFs, but also returns the motifs bound by transcription factors. And our model revealed a strong association among same families of TFs from the perspective of their associated sequence contexts. Under the idea of transfer learning, trans-species prediction performances on four TFs of Zea mays demonstrate the feasibility of current model. Overall, our work provides a solution for discovering TFBSs, enabling an integrated analysis of these TFs, and providing a better understanding of the transcriptional regulatory mechanism.


<p align="center">
<img src="https://github.com/wenkaiyan-kevin/PlantBind/blob/main/images/flow_chart.png" width = "600" height = "300" >
</p>  
<p align="center">Fig. 1 The model workflow</p>  

