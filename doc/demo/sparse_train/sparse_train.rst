Sparse Training with PaddlePaddle
=================================

This Document will guide you to understand the sparse design in PaddlePaddle.

Quick Start
-----------

Assuming that you are familiar with PaddlePaddle. Here show train billion level parameters with sparse training. 

NOTE: main code and data can be found in demo/sparse_train/sparse_binary directory.

Prepare Data
############
Given the Input data sample as follows:
```
0,0,7216737 1,18127654 1, ... 
```
The first slot ```0``` is just one flag, not one feature; the second slot ```0``` is  label; the third slot is feature id, ```7216737 1```means the ```7216737``` dimension is not zero, the ```1``` is just one flag, not one feature; the ```18127654 1``` is another non zero feature. The line data depicts one simple high dimension input data for classification task.

NOTE: You do not need to restrict your RAW data with above format, instead the last data ```yield``` by DataProvider determines the ultimate data format.

Unzip ```train.zip``` to ```train.txt``` firstly, all samples are stored as simple TXT file ```train.txt```, then set it in ```train.list``` index file, which is used in model configuration file. We just provides no more than 2000+ samples to explains following  experiments. Check related HOWTO in other DOC. 

### Prepare DataProvider
Use following dataprovider to feed DataProvider:

.. literalinclude:: ../../../demo/sparse_train/sparse_binary/sparse_data_provider.py

Please consult related DOC for ```sparse_binary_vector``` and ```integer_value```.

### Prepare Model Configuration
Here providing one simple fc_layers network, which is enough for explaining HOW to train DNN with sparse training.

We use ```ParameterAttribute(sparse_update=True)``` to enable sparse training for local job as well as cluster job. The ```trainer_config_helpers``` model will set internal FLAGs automatically. At last, these layers not set with sparse FLAG do dense training while sparse layers automatically use sparse training. Generally RECOMMEND you set these layers with high sparsity input, such as first hidden layers with sparse input.

The full configuration as follows:

.. literalinclude:: ../../../demo/sparse_train/sparse_binary/sparse_trainer_config.py

### Start Training 
For local train, you do not need to do anything to start sparse training. 

.. literalinclude:: ../../../demo/sparse_train/sparse_binary/local.sh

For cluster train, you maybe just care about ```--ports_num_for_sparse=4``` in command line and conf.py files.  You need to understand simple cluster helper scripts in ```paddle/cluster_train/paddle.py```. Check DOC about cluster training.


## Quantitive Analysis on Performance 
We takes several experiments to help you understand ```what happened``` and ```what will benefit``` from sparse training.  
To make you better understand sparse training, we could use some BIGGER hidden size to enlarge the performance gap between Sparse Training and default dense training. All analysis are based on QuickStart data and configuration.

### Without Sparse Training
Just remove ```ParameterAttribute(sparse_update=True)``` setting in fc_layer configuration to disable sparse training.   

With 
```
hidden1 = fc_layer(input=data,
                   size=256)
```
At last, local train exhaust more than 150GB RAM, then going to abort.
```
F0928 22:04:04.030071  8482 Allocator.h:52] Check failed: ptr Fail to allocate CPU memory: size=18618671104
```

The system memories are drained over by allocating HIGH dimension dense matrix.

With 
```
hidden1 = fc_layer(input=data,
                   size=64)
```
it also takes 107seconds(22:16:31 -> 22:18:51.) to train one dummy pass.
```
.....I0928 22:16:31.285207 28823 TrainerInternal.cpp:179]  Pass=0 Batch=11 samples=2002 AvgCost=0.0928303 Eval: classification_error_evaluator=0.039778
I0928 22:16:31.290060 28823 GradientMachine.cpp:112] Saving parameters to ./output/model/pass-00000
I0928 22:16:37.540330 28823 Util.cpp:219] copy ./sparse_trainer_config.py to ./output/model/pass-00000
.....I0928 22:18:51.734253 28823 TrainerInternal.cpp:179]  Pass=1 Batch=11 samples=2002 AvgCost=0.0262592 Eval: classification_error_evaluator=0.00231267
```

### With Local Sparse Training
Enable  ```ParameterAttribute(sparse_update=True)``` , 
With
```
hidden1 = fc_layer(input=non_value,
                   size=256,
                   param_attr=ParameterAttribute(sparse_update=True))
```
At last,  train goes successfully, and quickly finished all passes in ```several ```seconds.
```
.....I0928 21:55:42.169045 17090 TrainerInternal.cpp:179]  Pass=0 Batch=11 samples=2002 AvgCost=0.0913546 Eval: classification_error_evaluator=0.0615171
I0928 21:55:42.169306 17090 GradientMachine.cpp:112] Saving parameters to ./output/model/pass-00000
I0928 21:56:07.312760 17090 Util.cpp:219] copy ./sparse_trainer_config.py to ./output/model/pass-00000
.....I0928 21:56:07.640897 17090 TrainerInternal.cpp:179]  Pass=1 Batch=11 samples=2002 AvgCost=0.0174494 Eval: classification_error_evaluator=0.00185014
```
So, the memories and computation can significantly be reduced with sparse training.

### With Cluster Sparse Training
Go through ```cluster train``` DOC to build cluster environments with local workspace.
Assuming you have already got the ```workspace``` which contains data, model, dataprovider. Here we focus the system performance comparison, so we just ```--job_dispatch_package``` in ```paddle.py``` in cluster scripts to dispatch same train data in all nodes in current experiment.

Set ```PADDLE_PORTS_NUM_FOR_SPARSE = 2``` in ```conf.py``` and ```paddle.py```command options.

Configuration in conf.py :
```python
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 2
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 4
```

Command line

.. literalinclude:: ../../../demo/sparse_train/sparse_binary/cluster.sh

Compared with local sparse training, we further observed that the memories exhausted by trainer process decreased with same hidden size. Because the parameter server will store all parameters within different parameter servers' nodes. That fact shows that the sparse architecture in distributed parameter server can let you train HUGE model, such as the model whose size is larger than single node's RAM. 

### Sparse Train with sparse_vector Data Input
For ```sparse_vector``` input data,  the steps to enable sparse train are almost same with ```sparse_binary_vector``` input data.
Here we randomly generate ```sparse_vector``` train data used  to do  dummy regression training task to make sparse training understood quickly. The convergence could not be well, but it will be useful for explaining sparse training.

Goto ```demo/sparse_train/sparse_vector``` .

- Generate Dummy Train Data
run ```python gen_float_vector.py``` which will generate ``train.txt``` data.  
Sample:
```
52.100122,4166359 4.735778,8620047 4.930204,5973607 4.125353,8592696 2.977571,3270647 0.181225,7463359 2.426782,6412115 2.720940,2230734 2.443729,3660171 2.750104,380473 1.087141,6197803 0.488248,9819341 0.801008,3037143 1.037892,5178823 0.496779,4130425 3.762169,3819458 0.088705,3224943 0.608342,9388980 3.154990,9620311 4.721348,2012684 1.074520,7651424 3.516958,9628692 0.477870,25792 1.548257,8738445 4.085452,9220107 4.890630,7457557 0.508678,999978 3.217060,6684449 3.234488,6852225 1.932241,1684321 4.152327,4586561 3.631591 
```
First float data ```52.100122``` is output label; the second ```4166359 4.735778```
means the ```4166359``` dimension value is ```4.735778```;  other data is same with previous one. Other dimension that is not recorded is ```0```. 

- Prepare Sparse Data Provider
 The data sample looks like
  ```[ [ index, value], [ index, value], .. ]``` and ```[label]```

- Set Model
Set first hidden layer with ```param_attr=ParameterAttribute(sparse_update=True)``` property to enable sparse training ```forward```, ```backward``` and ```optimization``` .
Generally the layers that needed to set sparse property is first hidden layer attach with input data layers.
Full configuration file in ```sparse_float_trainer_config.py```.

- Start Train
Execute ```sh local.sh ``` to start sparse training.

- Performance Gains
 To disable sparse training with removing ```param_attr=ParameterAttribute(sparse_update=True)``` , the initialisation will fail because the HUGE parameters storage allocation failed. 
Training goes quickly within several seconds if sparse training is enabled, the memory size allocated for training also decreases from 170GB to 30GB.
```
.....I0929 18:08:46.986836 19235 TrainerInternal.cpp:179]  Pass=0 Batch=10 samples=2000 AvgCost=3160.34 Eval:
I0929 18:08:47.004266 19235 GradientMachine.cpp:112] Saving parameters to ./output/model/pass-00000
I0929 18:09:36.475224 19235 Util.cpp:219] copy ./sparse_float_trainer_config.py to ./output/model/pass-00000
.....I0929 18:09:36.945237 19235 TrainerInternal.cpp:179]  Pass=1 Batch=10 samples=2000 AvgCost=3150.74 Eval:
I0929 18:09:36.945366 19235 GradientMachine.cpp:112] Saving parameters to ./output/model/pass-00001
```

## Internals of Sparse Training

### Overview

In backpropagation algorithm, the fact that the partial derivative of the weight of parameters will be ```0``` if the input activations in all next mini-batch is ```0``` can help you do sparse optimization for reducing computation, storage, communication. 

The SGD parameters regularisation requires  the weights with 0 gradient needed to be updated, so to best effort to reduce system overhead, PaddlePaddle use ```catchup``` mechanism to trade off the lossless computation accusation requirement from algorithm with computation optimization.

The SPARSE design can reduce total RAM memories and CPU/GPU computation resource for training HIGH dimensions SPARSE model, even reduce the network overhead for cluster training.

PaddlePaddle controls sparse with the unit of ```parameter``` for local train and cluster train. In local model, different parameters can do dense or sparse training simultaneously. 

The ```Sparse``` training located in several aspects:

- ```Argument``` is stored sparsely. 
The Sparse Input Data Layers are stored with sparse matrix.  Except the Input Data Layers, others layers are dense since the activations could be non zero value.

- ```Weight Matrix``` is stored sparsely. 
Weight Matrix and Parameters can be regarded as same entities. Generally the gradient matrix are sparse while value matrix is not sparse.  For cluster training, the full value matrix is stored in parameter server, so the matrix in train end could be sparse.

- ```Computations```in backpropagation algorithm is sparsely.
 These computation contains forward computation, backward computation,  the engine can do sparse forwardbackward computation with the non zero sparse input activation.
 
- SGD optimization computation
The optimization could be applied not all parameter weights to reduce computation resource. PaddlePaddle use special ```catchup``` mechanism to do sparse optimization.

- ```Communication``` in cluster training is sparsely.
With the sparse parameter architecture,  any distributed trainers do ```prefetch``` latest sparse parameters value from parameter servers that currently mini-batch needs. Also trainers push latest sparse gradients to distributed parameter servers. It can significantly reduce memory and bandwidth exhaustion. The distributed architecture allows you to train HUGE model whose size is larger than that of the physical RAM memory in single host.

### Miscellous
With mini-batch training, the sparse training overhead could be related with mini-batch size because the union of all samples in mini-batch determine which weight needs to be updated. In further, the communication overhead is related with mini-batch if sparse training is enabled.

## Supported Layers

Theoretically, most layers can be configured with ```sparse```training, however if the degree of sparsity is not BIG enough the additional computation and storage needs for sparse matrix could hurt the overall performance.

It's recommend that use sparse setting with first hidden layer attach with HIGH dimension sparse input data layer. 

The best supported layer type for sparse training is fc_layer\ProjectedLayer\NCE layer.

## Supported Optimisers
Currently not all SGD Optmimizers support sparse updating.

```AdaDeltaOptimizer``` does not support sparse training.
With momentum setting, it's needed to update new parameter weight with history ```momentum``` even if current weight's gradient is 0,  so PaddlePaddle designs special ```MomentumOptimizer```  for sparse momentum.  The novel design can benefit ```sparse``` cluster communication for HUGE model which can significantly improve performance in cluster training.
The algorithm will be describe in details later.

## HUGE Model Training

PaddlePaddle designs one ```loadAndSaveParameterinPserver``` mechanism to build distributed model parameters storage.  It means the full parameters copy is stored in all parameters servers, all training processes fetch part of parameters as need after seeing next min-batch. Generally, this mechanism works with sparse training.
Check ```-loadsave_parameters_in_pserver``` for details.


## FAQ

 1. Does AdamOptimizer support Sparse Training ?
No.

 2.  ERROR "Protobuf error while training sparse model" ?
libprotobuf ERROR google/protobuf/io/coded_stream.cc:171] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
If the number of distributed sparse parameter server is limited,  trainer could create huge protobuf request whose size exceeds the limit of PROTOBUF. 
You can increase ```--ports_num_for_sparsec``` to BIGGER value, to create more sparse parameters servers to handle it. This techniques also can benifit network efficiency sometimes.

 3.  How does sparse training implemented?
Check ```Internals of Sparse Training``` section

 4.  Whether```sparse_float_vector``` input can used to do sparse train?
Yes
