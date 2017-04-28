# Use Case

## Local Training

These command line arguments are commonly used by local training experiments, such as image classification, natural language processing, et al.

```
paddle train \
  --use_gpu=1/0 \                        #1:GPU,0:CPU(default:true)
  --config=network_config \
  --save_dir=output \
  --trainer_count=COUNT \                #(default:1)
  --test_period=M \                      #(default:0) 
  --num_passes=N \                       #(defalut:100)
  --log_period=K \                       #(default:100)
  --dot_period=1000 \                    #(default:1)
  #[--show_parameter_stats_period=100] \ #(default:0)
  #[--saving_period_by_batches=200] \    #(default:0)
```
`show_parameter_stats_period` and `saving_period_by_batches` are optional according to your task.

### 1) Pass Command Argument to Network config

`config_args` is a useful parameter to pass arguments to network config.

```
--config_args=generating=1,beam_size=5,layer_num=10 \
```
And `get_config_arg` can be used to parse these arguments in network config as follows:

```
generating = get_config_arg('generating', bool, False)
beam_size = get_config_arg('beam_size', int, 3)
layer_num = get_config_arg('layer_num', int, 8)
```

`get_config_arg`:

```
get_config_arg(name, type, default_value)
```
- name: the name specified in the `--config_args`
- type: value type, bool, int, str, float etc.
- default_value: default value if not set.

### 2) Use Model to Initialize Network

add argument:

```
--init_model_path=model_path
--load_missing_parameter_strategy=rand
```

## Local Testing

Method 1:

```
paddle train --job=test \
             --use_gpu=1/0 \ 
             --config=network_config \
             --trainer_count=COUNT \ 
             --init_model_path=model_path \
```
- use init\_model\_path to specify test model.
- only can test one model.

Method 2:

```
paddle train --job=test \
             --use_gpu=1/0 \ 
             --config=network_config \
             --trainer_count=COUNT \ 
             --model_list=model.list \
```
- use model_list to specify test models
- can test several models, where model.list likes:

```
./alexnet_pass1
./alexnet_pass2
```

Method 3:

```
paddle train --job=test \
             --use_gpu=1/0 \
             --config=network_config \
             --trainer_count=COUNT \
             --save_dir=model \
             --test_pass=M \
             --num_passes=N \
```
This way must use model path saved by Paddle like this: `model/pass-%5d`. Testing model is from M-th pass to (N-1)-th pass. For example: M=12 and N=14 will test `model/pass-00012` and `model/pass-00013`.

## Sparse Training

Sparse training is usually used to accelerate calculation when input is sparse data with highly dimension. For example, dictionary dimension of input data is 1 million, but one sample just have several words. In paddle, sparse matrix multiplication is used in forward propagation and sparse updating is perfomed on weight updating after backward propagation.

### 1) Local training

You need to set **sparse\_update=True** in network config.  Check the network config documentation for more details.

### 2) cluster training

Add the following argument for cluster training of a sparse model. At the same time you need to set **sparse\_remote\_update=True** in network config. Check the network config documentation for more details.

```
--ports_num_for_sparse=1    #(default: 0)
```

## parallel_nn
`parallel_nn` can be set to mixed use of GPUs and CPUs to compute layers. That is to say, you can deploy network to use a GPU to compute some layers and use a CPU to compute other layers. The other way is to split layers into different GPUs, which can **reduce GPU memory** or **use parallel computation to accelerate some layers**.

If you want to use these characteristics, you need to specify device ID in network config (denote it as deviceId) and add command line argument:

```
--parallel_nn=true
```
### case 1: Mixed Use of GPU and CPU
Consider the following example:

```
#command line:
paddle train --use_gpu=true --parallel_nn=true trainer_count=COUNT

default_device(0)

fc1=fc_layer(...)
fc2=fc_layer(...)
fc3=fc_layer(...,layer_attr=ExtraAttr(device=-1))

```
- default_device(0): set default device ID to 0. This means that except the layers with device=-1, all layers will use a GPU, and the specific GPU used for each layer depends on trainer\_count and gpu\_id (0 by default). Here, layer fc1 and fc2 are computed on the GPU.

- device=-1: use the CPU for layer fc3.

- trainer_count:
  - trainer_count=1: if gpu\_id is not set, then use the first GPU to compute layers fc1 and fc2. Otherwise use the GPU with gpu\_id.

  - trainer_count>1: use trainer\_count GPUs to compute one layer using data parallelism. For example, trainer\_count=2 means that GPUs 0 and 1 will use data parallelism to compute layer fc1 and fc2.

### Case 2: Specify Layers in Different Devices

```
#command line:
paddle train --use_gpu=true --parallel_nn=true --trainer_count=COUNT

#network:
fc2=fc_layer(input=l1, layer_attr=ExtraAttr(device=0), ...)
fc3=fc_layer(input=l1, layer_attr=ExtraAttr(device=1), ...)
fc4=fc_layer(input=fc2, layer_attr=ExtraAttr(device=-1), ...)
```
In this case, we assume that there are 4 GPUs in one machine.

- trainer_count=1:
  - Use GPU 0 to compute layer fc2.
  - Use GPU 1 to compute layer fc3.
  - Use CPU to compute layer fc4.

- trainer_count=2:
  - Use GPU 0 and 1 to compute layer fc2.
  - Use GPU 2 and 3 to compute layer fc3.
  - Use CPU to compute fc4 in two threads.

- trainer_count=4:
  - It will fail (note, we have assumed that there are 4 GPUs in machine), because argument `allow_only_one_model_on_one_gpu` is true by default.

**Allocation of device ID when `device!=-1`**:

```
(deviceId + gpu_id + threadId * numLogicalDevices_) % numDevices_

deviceId:             specified in layer.
gpu_id:               0 by default.
threadId:             thread ID, range: 0,1,..., trainer_count-1
numDevices_:          device (GPU) count in machine.
numLogicalDevices_:   min(max(deviceId + 1), numDevices_)
```
