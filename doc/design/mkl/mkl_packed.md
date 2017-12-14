# Intel® MKL Packed Optimization on PaddlePaddle: Design Doc


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
现阶段的优化主要针对 Recurrent Neural Network(以下简称RNN)相关层（包括`RecurrentLayer`, `GatedRecurrentLayer`和`LstmLayer`）， 以及 PaddlePaddle V1 API。

## Key Points

### Background
为了达到最佳性能， Intel® MKL 中的 cblas_?gemm 会在计算前将原数据转换为更适合英特尔平台的Packed格式， 这一数据格式的转换操作 (Packing)，在问题本身的计算量比较小的时候显得相对来说较为耗时。
在现有的某些情况下（例如RNN），多次调用 cblas_?gemm 时会使用相同的原数据，每次调用时对原数据的重复Packing便成为了冗余。

为了最大程度减少多次调用 cblas_?gemm 在Packing上的耗时，Intel® MKL 引入了以下四个API:
   * cblas_?gemm_alloc
   * cblas_?gemm_pack 
   * cblas_?gemm_compute
   * cblas_?gemm_free

通过使用这些API，我们可以先完成对原数据的Packing操作，再把已转换为Packed格式的数据传递给那些复用同一数据的gemm_compute函数，从而避免了Packing冗余。

### Solution
在RNN的case下，同一次 forward/backward 过程中所有time state共享同一个weight矩阵。当只做 inference 时，各次 forward 之间也都使用相同的weight矩阵，没有必要在每次forward中每个time state的计算时对weight进行重复的packing操作。

我们通过使用新引入的GEMM Packed APIs，在layer init时先完成对weight的packing操作，然后在 forward/backward 时复用已pack过后的weight，并在每次weight更新后重新Packing。

* 优化前，对于sequence length = `T` 的model, `N` 次iteration执行的Packing次数为：   
  - `inference`: `N * T`  
  - `training`: `2 * N * T`
* 优化后，对于sequence length = `T` 的model, `N` 次iteration执行的Packing次数减少至：
  - `inference`: `1`    
  - `training`: `2 * N`

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
所有的`MKLPacked*Layer`都继承于PaddlePaddle的基类`Layer`, 并添加头文件 `MKLPackedGemm.h`，该文件中实现的对相关GEMM Packed APIs做了封装。

### Unit Tests
我们会添加`test_MKLPacked.cpp`用于MKL Packed优化后layer的测试。
对于每一个新加的RNN layer，我们会对比如下2个方面：
1. 对比优化后layer自身，sequence mode（`rnn_use_batch=false`）与batch mode(`rnn_use_batch=true`)的结果。
2. 对比优化后layer与相对应的PaddlePaddle原有layer, 在batch mode下的结果。

### Python API
TBD

### Benchmarking
会添加相应的脚本用于测试和对比在使用MKL Packed recurrent layers 前后的网络性能。

## References 
1. [Introducing the new Packed APIs for GEMM](https://software.intel.com/en-us/articles/introducing-the-new-packed-apis-for-gemm)


