# 使用案例

## 本地训练

本地训练的实验，诸如图像分类，自然语言处理等，通常都会使用下面这些命令行参数。

```
paddle train \
  --use_gpu=1/0 \                        #1:GPU,0:CPU(默认为1)
  --config=network_config \
  --save_dir=output \
  --trainer_count=COUNT \                #(默认为1)
  --test_period=M \                      #(默认为0) 
  --num_passes=N \                       #(默认为100)
  --log_period=K \                       #(默认为100)
  --dot_period=1000 \                    #(默认为1)
  #[--show_parameter_stats_period=100] \ #(默认为0)
  #[--saving_period_by_batches=200] \    #(默认为0)
```
根据你的任务，可以选择是否使用参数`show_parameter_stats_period`和`saving_period_by_batches`。

### 1) 将命令参数传给网络配置

`config_args`是一个很有用的参数，用于将参数传递给网络配置。

```
--config_args=generating=1,beam_size=5,layer_num=10 \
```
`get_config_arg`可用于在网络配置中解析这些参数，如下所示：

```
generating = get_config_arg('generating', bool, False)
beam_size = get_config_arg('beam_size', int, 3)
layer_num = get_config_arg('layer_num', int, 8)
```

`get_config_arg`:

```
get_config_arg(name, type, default_value)
```
- name: `--config_args`中指定的名字
- type: 值类型，包括bool, int, str, float等
- default_value: 默认值

### 2) 使用模型初始化网络

增加如下参数：

```
--init_model_path=model_path
--load_missing_parameter_strategy=rand
```

## 本地测试

方法一：

```
paddle train --job=test \
             --use_gpu=1/0 \ 
             --config=network_config \
             --trainer_count=COUNT \ 
             --init_model_path=model_path \
```
- 使用init\_model\_path指定测试的模型
- 只能测试单个模型

方法二：

```
paddle train --job=test \
             --use_gpu=1/0 \ 
             --config=network_config \
             --trainer_count=COUNT \ 
             --model_list=model.list \
```
- 使用model_list指定测试的模型列表
- 可以测试多个模型，文件model.list如下所示：

```
./alexnet_pass1
./alexnet_pass2
```

方法三：

```
paddle train --job=test \
             --use_gpu=1/0 \
             --config=network_config \
             --trainer_count=COUNT \
             --save_dir=model \
             --test_pass=M \
             --num_passes=N \
```
这种方式必须使用Paddle存储的模型路径格式，如：`model/pass-%5d`。测试的模型包括从第M轮到第N-1轮存储的所有模型。例如，M=12，N=14这种写法将会测试模型`model/pass-00012`和`model/pass-00013`。

## 稀疏训练

当输入是维度很高的稀疏数据时，通常使用稀疏训练来加速计算过程。例如，输入数据的字典维数是1百万，但是每个样本仅包含几个词。在Paddle中，稀疏矩阵的乘积应用于前向传播过程，而稀疏更新在反向传播之后的权重更新时进行。

### 1) 本地训练

用户需要在网络配置中指定**sparse\_update=True**。请参照网络配置的文档了解更详细的信息。

### 2) 集群训练

在集群上训练一个稀疏模型需要加上下面的参数。同时用户需要在网络配置中指定**sparse\_remote\_update=True**。请参照网络配置的文档了解更详细的信息。

```
--ports_num_for_sparse=1    #(默认为0)
```

## parallel_nn
用户可以设置`parallel_nn`来混合使用GPU和CPU计算网络层的参数。也就是说，你可以将网络配置成某些层使用GPU计算，而其他层使用CPU计算。另一种方式是将网络层划分到不同的GPU上去计算，这样可以减小GPU内存，或者采用并行计算来加速某些层的更新。

如果你想使用这些特性，你需要在网络配置中指定设备的ID号(表示为deviceId)，并且加上下面的命令行参数:

```
--parallel_nn=true
```
### 案例一：GPU和CPU混合使用
请看下面的例子：

```
#command line:
paddle train --use_gpu=true --parallel_nn=true trainer_count=COUNT

default_device(0)

fc1=fc_layer(...)
fc2=fc_layer(...)
fc3=fc_layer(...,layer_attr=ExtraAttr(device=-1))

```
- default_device(0): 设置默认设备号为0。这意味着除了指定device=-1的层之外，其他所有层都会使用GPU计算，每层使用的GPU号依赖于参数trainer\_count和gpu\_id(默认为0)。在此，fc1和fc2层在GPU上计算。

- device=-1: fc3层使用CPU计算。

- trainer_count:
  - trainer_count=1: 如果未设置gpu\_id，那么fc1和fc2层将会使用第1个GPU来计算。否则使用gpu\_id指定的GPU。

  - trainer_count>1: 在trainer\_count个GPU上使用数据并行来计算某一层。例如，trainer\_count=2意味着0号和1号GPU将会使用数据并行来计算fc1和fc2层。

### 案例二：在不同设备上指定层

```
#command line:
paddle train --use_gpu=true --parallel_nn=true --trainer_count=COUNT

#network:
fc2=fc_layer(input=l1, layer_attr=ExtraAttr(device=0), ...)
fc3=fc_layer(input=l1, layer_attr=ExtraAttr(device=1), ...)
fc4=fc_layer(input=fc2, layer_attr=ExtraAttr(device=-1), ...)
```
在本例中，我们假设一台机器上有4个GPU。

- trainer_count=1:
  - 使用0号GPU计算fc2层。
  - 使用1号GPU计算fc3层。
  - 使用CPU计算fc4层。

- trainer_count=2:
  - 使用0号和1号GPU计算fc2层。
  - 使用2号和3号GPU计算fc3层。
  - 使用CPU两线程计算fc4层。

- trainer_count=4:
  - 运行失败（注意到我们已经假设机器上有4个GPU），因为参数`allow_only_one_model_on_one_gpu`默认设置为真。

**当`device!=-1`时设备ID号的分配：**

```
(deviceId + gpu_id + threadId * numLogicalDevices_) % numDevices_

deviceId:             在层中指定
gpu_id:               默认为0
threadId:             线程ID号，范围: 0,1,..., trainer_count-1
numDevices_:          机器的设备(GPU)数目
numLogicalDevices_:   min(max(deviceId + 1), numDevices_)
```
