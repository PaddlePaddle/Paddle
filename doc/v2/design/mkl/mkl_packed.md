# Intel® MKL Packed on PaddlePaddle: Design Doc


## Contents

- [Overview](#overview)
- [Key Points](#key-points) 
   - [Background](#background)
   - [Solution](#solution)
- [Actions](#actions)
    - [CMake](#cmake)
	- [Layers](#layers)
	- [Unit Tests](#unit-tests)
	- [Python API](#python-api)
	- [Benchmarking](#benchmarking)


## Overview
我们计划将 Intel® MKL 中引入的 GEMM Packed APIs\[[1](#references)\] 集成到 PaddlePaddle 中，充分发挥英特尔平台的优势，有效提升PaddlePaddle在英特尔架构上的性能。
现阶段的优化主要针对 Recurrent Neural Network（以下简称RNN）相关层（包括`RecurrentLayer`, `GatedRecurrentLayer`和`LstmLayer`）， 以及 PaddlePaddle V1 API。

## Key Points

### Background
目前PaddlePaddle采用了 Intel® MKL库的[cblas_?gemm](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm)函数，这个函数本身会在计算前将原数据转换为更适合英特尔平台的内部格式。

1. 转换耗时 \
这一数据格式的转换操作（Packing），在问题本身的计算量比较小的时候，显得相对来说较为耗时。例如在DeepSpeech2 \[[2](#references)\] 的Vanilla RNN部分中，矩阵大小是`batch_size * 2048`。
2. 转换冗余 \
由于在现有的某些情况下（例如RNN），多次调用 cblas_?gemm 会使用相同的原数据，因此，每次调用时对原数据的重复Packing便成为了冗余。

为了最大程度减少多次调用 cblas_?gemm 在Packing上的耗时，Intel® MKL 引入了以下四个API:
   * [cblas_?gemm_alloc](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-alloc)
   * [cblas_?gemm_pack](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-pack)
   * [cblas_?gemm_compute](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-compute)
   * [cblas_?gemm_free](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm-free)

通过使用这些API，我们可以先完成对原数据的Packing操作，再把已转换为Packed格式的数据传递给那些复用同一数据的gemm_compute函数，从而避免了Packing冗余。

### Solution
在RNN的情况下，同一次前向、后向（forward/backward）过程中所有时间步（time step）共享同一个权重（weight）。当只做推断（inference）时，各次前向之间也都使用了相同的权重，没有必要在每次前向中每个时间步的计算时对权重进行重复的Packing操作。

我们通过使用新引入的GEMM Packed APIs，在层初始化的时候，先完成对权重的Packing操作，然后在前向，后向时复用已经转换过的权重，并在每次权重更新后，对新的权重进行转换用于下次迭代。

* 优化前，对于序列长度（sequence length）为`T`的网络模型（model）, `N`次迭代执行的转换次数为：
  - `inference`： `N * T`  
  - `training`： `2 * N * T`
* 优化后，对于同样设置的网络模型，其转换次数减少至：
  - `inference`： `1`    
  - `training`： `2 * N`

## Actions

添加的相关文件和目录结构如下：

```txt
PaddlePaddle/Paddle
├── ...
└── paddle/
    ├── ...
    └── gserver/
        ├── ...
        ├── layers/
        │   ├── ...
        │   ├── MKLPackedRecurrentLayer.*
        |   ├── MKLPackedGatedRecurrentLayer.*
        |   ├── MKLPackedLstmLayer.*
        |   └── MKLPackedGemm.h
        └── tests/
            ├── ...
            └── test_MKLPacked.cpp
```

### CMake
在对应的`CMakeLists.txt`中根据`WITH_MKL`是否打开，来决定是否开启MKL Packed相关功能。

### Layers
所有的`MKLPacked*Layer`都继承于PaddlePaddle的基类`Layer`, 并添加头文件 `MKLPackedGemm.h`，该文件对相关GEMM Packed APIs做了封装。

### Unit Tests
我们会添加`test_MKLPacked.cpp`用于MKL Packed优化后layer的测试。
对于每一个新加的RNN layer，我们会对比如下2个方面：
1. 对比优化后layer自身，sequence mode（`rnn_use_batch=false`）与batch mode(`rnn_use_batch=true`)的结果。
2. 对比优化后layer与相对应的PaddlePaddle原有layer, 在batch mode下的结果。

### Python API
计划在`paddle/utils.Flags`中添加`use_mkl_packed`的flag，用于选择是否使用相关功能，并且当编译时`WITH_MKL=ON`的情况下，默认设置为`true`。

同时，在`python/paddle/trainer/config_parser.py`中对应的layer处，添加`use_mkl_packed`这个选择，方便用户在Python端选择是否启用这个功能。

具体实现方式比如：

```python
use_mkl_packed = bool(int(g_command_config_args.get("use_mkl_packed", 0)))
if use_mkl_packed:
    self.layer_type = mkl_packed_*
```

所有相关的`layer_type`会以*mkl_packed_*开头，这些会在`MKLPacked*Layer`注册layer的时候保证，以示区分。 


### Benchmarking
会添加相应的脚本用于测试和对比在使用MKL Packed recurrent layers 前后的网络性能。

## References 
1. [Introducing the new Packed APIs for GEMM](https://software.intel.com/en-us/articles/introducing-the-new-packed-apis-for-gemm)
2. [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech#deepspeech2-on-paddlepaddle)

