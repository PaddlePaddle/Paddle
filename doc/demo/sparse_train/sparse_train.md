# Sparse Training with PaddlePaddle

This Document will guide you to understand the sparse design in PaddlePaddle.

## Quick Start (数据来自游戏的真实数据，收集了2000个样本）
Assuming that you are familiar with PaddlePaddle. Here show train billion level parameters with sparse training.

### Prepare Data
Given the Input data sample as follows:
```
0,0,7216737 1,18127654 1, ... 
```
The first slot ```0``` is just one flag, not one feature; the second slot ```0``` is  label; the third slot is feature id, ```7216737 1```means the ```7216737``` dimension is not zero, the ```1``` is just one flat, not one feature; the ```18127654 1``` is another non zero feature. The line data depicts one simple high dimension input data for classification task.

All samples are stored as simple TXT file ```train.txt```, then set it in ```train.list``` index file, which is used in model configuration file. We just provides no more than 2000+ samples to explains following  experiments. Check related HOWTO in other DOC. 

### Prepare DataProvider
Use following dataprovider to feed DataProvider:
```python
# Define a py data provider
@provider(input_types=[
    sparse_binary_vector(18182296),
    integer_value(2)
])
def process(settings, filename):
    f = open(filename, 'r')
    for line in f:  # read each line
        splits = line.split(',')
        label = int(splits[1])
        splits.pop(0)
        splits.pop(0)
        sparse_non_values = []
        for value in splits:
            v = value.split(" ")
            sparse_non_values.append(long(v[0]))
        # give data to paddle.
        yield sparse_non_values, label
    f.close()  # close file
```
Please consult related DOC for ```sparse_binary_vector``` and ```integer_value```.

### Prepare Model Configuration
Here providing one simple fc_layers network, which is enough for explaining HOW to train DNN with sparse training.

We use ```ParameterAttribute(sparse_update=True)``` to enable sparse training for local job as well as cluster job. The ```trainer_config_helpers``` model will set internal FLAGs automatically. At last, these layers not set with sparse FLAG do dense training while sparse layers automatically use sparse training. Generally RECOMMEND you set these layers with high sparsity input, such as first hidden layers with sparse input.

The full configuration as follows:
```python
rom paddle.trainer_config_helpers import *

label_size = 2
data_size = 18182296

""" Algorithm Configuration """
settings(learning_rate=1e-3,
         learning_method=MomentumOptimizer(momentum=0.9),
         batch_size=200)

""" Data Configuration """
define_py_data_sources2(train_list='train.list',
                        test_list=None,
                        module='sparse_data_provider',
                        obj='process')

""" Model Configuration """
non_value = data_layer(name='data',
                       size=data_size)
label = data_layer(name='label',
                   size=label_size)

hidden1 = fc_layer(input=non_value,
                   size=128,
                   param_attr=ParameterAttribute(sparse_update=True))
hidden2 = fc_layer(input=hidden1,
                   size=32)

prediction = fc_layer(input=hidden2, size=label_size, act=SoftmaxActivation())

outputs(classification_cost(input=prediction, label=label))
```

### Start Training 
For local train, you do not need to do anything to start sparse training. 
```bash
paddle train \
    --use_gpu=0 \
    --config=./sparse_trainer_config.py \
    --saving_period=1 \
    --test_period=0 \
    --num_passes=4 \
    --dot_period=2 \
    --log_period=20 \
    --trainer_count=10 \
    --saving_period_by_batches=5000 \
    --local=1
```
For cluster train, you maybe just care about ```--ports_num_for_sparse=4``` in command line and conf.py files.  You need to understand simple cluster helper scripts in ```paddle/cluster_train/paddle.py```. Check DOC about cluster training.

## Quantitive Analysis on Performance (dense的时候，input argument 会稀疏么/ hidden size =128的时候报错了）
We takes several experiments to help you understand ```what happened``` and ```what will benefit``` from sparse training.  
To make you better understand sparse training, we could use some BIGGER hidden size to enlarge the performance gap between Sparse Training and default dense training. All analysis are based on QuickStart data and configuration.

### Without Sparse Training
Just remove ```ParameterAttribute(sparse_update=True)``` setting in fc_layer configuration to disable sparse training.   

With 
```
hidden1 = fc_layer(input=non_value,
                   size=256)
```
At last, local train exhaust more than 150GB RAM, then going to abort.
```
F0928 22:04:04.030071  8482 Allocator.h:52] Check failed: ptr Fail to allocate CPU memory: size=18618671104
```

The system memories are drained over by allocating HIGH dimension dense matrix.

With 
```
hidden1 = fc_layer(input=non_value,
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
```
PATH_TO_LOCAL_WORKSPACE=/home/sparse_test/workspace
python paddle.py \
  --job_dispatch_package="${PATH_TO_LOCAL_WORKSPACE}" \
  --use_gpu=0 \
  --config=./sparse_trainer_config.py \
  --saving_period=1 \
  --test_period=0 \
  --num_passes=4 \
  --dot_period=2 \
  --log_period=20 \
  --trainer_count=10 \
  --saving_period_by_batches=5000 \
  --ports_num_for_sparse=2 \
  --local=0 \
```

Compared with local sparse training, we further observed that the memories exhausted by trainer process decreased with same hidden size. Because the parameter server will store all parameters within different parameter servers' nodes. That fact shows that the sparse architecture in distributed parameter server can let you train HUGE model, such as the model whose size is larger than single node's RAM. 


## Internals of Sparse Training

### Overview

In backpropagation algorithm, the partial derivative of the weight in parameters is ZERO, if the neuron....

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
Without parameters generalisation, these optimizers could just update a few weights value with new non zero gradients. Instead, PaddlePaddle utilizes special techniques to handle the generalisation to trade off parameter optimization in local and cluster mode.
- ```Communication``` in cluster training is sparsely.
With the sparse parameter architecture,  any distributed trainers do ```prefetch``` latest sparse parameters value from parameter servers that currently mini-batch needs. Also trainers push latest sparse gradients to distributed parameter servers. It can significantly reduce memory and bandwidth exhaustion. The distributed architecture allows you to train HUGE model whose size is larger than that of the physical RAM memory in single host.

### Sparse in Forward & Backward
### prefetch
### Catch up Mechanism for Optimisation
### Sparse Momentum

## Supported Layers

Theoretically, most layers can be configured with ```sparse```training, however if the degree of sparsity is not BIG enough the additional computation and storage needs for sparse matrix could hurt the overall performance.
Generally the sparse setting is configured with HIGH dimension SPARSE ```input data layer```, since the sparse input value ```ZERO``` can be regarded as ZERO activation.

TODO:
fc_layer\???NCE???\ProjectedLayer\ are supported well.

## Supported Optimisers
TODO:

Currently not all SGD Optmimizers support sparse updating, AdaDelta???? not support...

## HUGE Model Training

If the size of Models is so large enough that the single node can not hold all parameters, you can use VLSNN training design to handle it.
In theroy,  VLSNN design is not related with sparse training, however,  they will share some APIs within parameter servers.
TODO(yanfei):


## FAQ

 1. Is AdamOptimizer support Sparse Training ?
No.
 2.  Protobuf error while training sparse model.
libprotobuf ERROR google/protobuf/io/coded_stream.cc:171] A protocol message was rejected because it was too big (more than 67108864 bytes).  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
If the number of distributed sparse parameter server is limited,  trainer could create huge protobuf request whose size exceeds the limit of PROTOBUF. 
You can enlarge ```--ports_num_for_sparse``` to BIGGER value, to create more sparse parameters servers to handle it. This techniques also can benifit network efficiency sometimes.


