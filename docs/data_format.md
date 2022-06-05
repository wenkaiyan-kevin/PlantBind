# Data format

### ***DNA Sequence File*** <a name="DNA_Sequence_File"/>
This file mainly contains the sequence information of DNA, each row is a sequence.
An example of data is shown below:  
<br />
<img src="https://github.com/wenkaiyan-kevin/PlantBind/blob/main/images/sequence-format.png" width = "600" height = "300" >  
<br />
During training, validation, testing, their filenames must be **train_sequence.table**, **valid_sequence.table** and **test_sequence.table**.  
If it is a prediction process, the file name must save **sequence.table**

### ***DNA Shape File*** <a name="DNA_Shape_File"/>
This file mainly contains the shape information of DNA. The data here is saved in numpy format. If you want to view the content of the data, please use the following method by python.
```
>>> import numpy as np
>>> DNA_shape_data = np.load('test_DNA_shape.npy')
>>> DNA_shape_data.shape
(11977, 97, 14)
```
During training, validation, testing, their filenames must be **train_DNA_shape.npy**, **valid_DNA_shape.npy** and **test_DNA_shape.npy**.  
If it is a prediction process, the file name must save **DNA_shape.npy**

### ***Data Label File*** <a name="Data_Label_File"/>
This file mainly contains the sequence information of DNA, each column is a TF, each row is a peak region.
An example of data is shown below:
<br />
<img src="https://github.com/wenkaiyan-kevin/PlantBind/blob/main/images/data-label.png" width = "600" height = "300" >  
<br />

