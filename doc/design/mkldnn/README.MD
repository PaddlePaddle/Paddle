# Intel® MKL-DNN on PaddlePaddle: Design Doc

我们计划将Intel深度神经网络数学库(**MKL-DNN**\[[1](#references)\])集成到PaddlePaddle，充分展现英特尔平台的优势，有效提升PaddlePaddle在英特尔架构上的性能。

我的短期内的基本目标是：

- 完成常用layer的MKLDNN实现。
- 完成常见深度神经网络VGG，GoogLeNet 和 ResNet的MKLDNN实现。


## Contents

- [Overall](#Overall)
	- [Cmake](#Cmake)
	- [Layer](#Layer)
	- [Activation](#Activations)
	- [Unit Test](#Unit-Test)
	- [Proto](#Proto)
	- [Python API](#Python-API)
	- [Demo](#Demo)
	- [Benchmark](#Benchmark)
	- [Others](#Others)
- [Optimized Design](#Optimized-Design)
	- [New](#new)
	- [Add](#add)


## Overall

整体上，我们粗略的把集成方案分为了如下几个方面。

### Camke
我们会在`CMakeLists.txt`中会添加`WITH_MKLDNN`的选项，当设置这个值为`ON`的时候会启用编译MKLDNN功能。同时会自动开启`OpenMP`用于提高MKLDNN的性能。

为了让PaddlePaddle更好的发挥MKLDNN的性能，我们还会引入了`WITH_MKLML`的选项，用于选择是否用MKLDNN自带的MKL cblas的安装包。这个安装包可以独立于MKLDNN使用，但是建议在开启MKLDNN的同时也打开MKLML的开关，这样才能发挥最好的性能。

所以，我们会在`cmake\external`新建一个`MKLDNN.cmake`和`MKLML.cmake`文件，并作为第三方库安装到PaddlePaddle的third party目录中。

**备注**：当`WITH_MKLML=ON`的时候，会优先使用这个包作为PaddlePaddle的Cblas和Lapack库，所以会稍微改动`cmake\cblas.cmake`中的逻辑。

### Layer
所有的layer相关的C++代码会在安装PaddlePaddle的目录结构存放在
`paddle\gserver\layers`中，文件名以*Mkldnn*开头。
并且有可能会在Layer.h和Layer.cpp里面添加少量的code，用宏定义`PADDLE_USE_MKLDNN`隔开。

所有MKLDNN的Layer都会继承一个MKLDNN的父类layer，这个父类mkldnnlayer继承Paddle的基类layer。

### Activation
由于在PaddlePaddle中，激活函数是独立于layer概念的，所以会在`paddle\gserver\activations`目录下添加一个`MkldnnActivation.h`文件定义一些用于Mkldnn的接口，实现方法还是在`ActivationFunction.cpp`里面

### Unit Test
会在`paddle\gserver\test`里面添加`test_Mkldnn.cpp`和`MkldnnTester.*`用于mkldnn的测试。

Activation的测试会在Paddle原有基础上直接添加测试type。

### Proto
根据具体layer的需求会在`proto\ModelConfig.proto`里面添加必要的选项

### Python API
目前只考虑**v1 API**。

所有layer相关的会在`python/paddle/trainer/config_parser.py`里面添加`use_mkldnn`这个选择，方便user选择使用mkldnn的layers。

具体实现方式比如：


	use_mkldnn = bool(int(g_command_config_args.get("use_mkldnn", 0)))
	if use_mkldnn
            self.layer_type = mkldnn_*

所有mkldnn的type我会以"mkldnn_"开头。 
并且可能在`python/paddle/trainer_config_helper`目录下的`activations.py `和`layers.py`里面添加必要的mkldnn的接口。

### Demo

会在`v1_api_demo`目录下添加一个`mkldnn`的文件夹，里面放入一些用于mkldnn测试的demo model。

### Benchmark
会考虑改变(或者添加)在`benchmark\paddle\image\run.sh`，添加使用mkldnn的测试。

### Others
1. 如果使用MKLDNN的情况下，会把cpu的allocate的align为64。
2. 深入PaddlePaddle，找到其他可以优化的可能，进一步优化。比如可能会用`OpenMP`改进SGD的更新性能。

## Optimized Design

为了更好的符合PaddlePaddle的代码风格，决定尽可能少的在PaddlePaddle的父类Layer中添加变量或者函数。
使用已有的`deviceId_`变量来区分layer的属性，定义`-2`为MkldnnLayer特有的设备值。

### New
1. 创建**MkldnnLayer**，并override父类Layer的init函数，修改deviceId_为`-2`代表这个layer是用于跑在MKLDNN的环境下。
2. 创建**MkldnnMatrix**，用于管理MKLDNN会用到的各种memory函数和接口。
3. 创建**MkldnnBase**，定义一些除了layer和memory相关的类和函数。包括MKLDNN会用到Stream和CpuEngine，和未来可能还会用到FPGAEngine等。

### Add
1. 在现有的**Argument**里面添加两个**MkldnnMatrixPtr**，取名为mkldnnValue和mkldnnGrad，用于存放MkldnnLayer会用到的memory buffer。 并且添加函数cvt(会改一个更加合适的函数名)，用于处理CPU device和MKLDNN device之间memory的相互转化。
2. 在父类Layer中的`getOutput`函数中添加一段逻辑，用于判断`deviceId`，并针对device在MKLDNN和CPU之间不统一的情况，做一个提前转换。 也就是调用Argument的cvt函数把output统一到需要的device上。
3. 在原来的Flag中添加一个`use_mkldnn`的flag，用于选择是否只使用MKLDNN的功能。
## References

1. [Intel Math Kernel Library for Deep Neural Networks (Intel MKL-DNN)](https://github.com/01org/mkl-dnn "Intel MKL-DNN")
