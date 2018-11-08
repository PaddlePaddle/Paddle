# 移动PaddlePaddle

PaddlePaddle支持在移动设备上，使用训练好的模型进行离线推断。这里，我们主要介绍如何在移动设备上部署PaddlePaddle推断库，以及移动设备上可以使用到的一些优化方法。

## 构建PaddlePaddle库
PaddlePaddle可以通过原生编译、交叉编译的方式，构建多种移动平台上的推断库。

- [Android平台编译指南](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_cn.md)
- [iOS平台编译指南](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_ios_cn.md)
- [Rapsberry Pi3平台编译指南](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_raspberry_cn.md)
- NVIDIA Driver PX2平台，采用原生编译的方式，可直接依照[PaddlePaddle源码编译指南](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/v2/build_and_install/build_from_source_cn.rst)进行编译

## 使用示例

- [命令行示例程序](./benchmark/tool/C/README.cn.md)
- [iOS示例应用：PDCamera](./Demo/iOS/AICamera/README.cn.md)

## 部署优化方法
移动端对接入库的大小通常都有要求，在编译PaddlePaddle库时，用户可以通过设置一些编译选项来进行优化。

- [如何构建最小的PaddlePaddle推断库](./deployment/library/build_for_minimum_size.md)

训练得到的模型，可在不降低或者轻微降低模型推断精度的前提下，进行一些变换，优化移动设备上的内存使用和执行效率。

- [合并网络中的BN层](./deployment/model/merge_batch_normalization/README.md)
- [压缩模型大小的rounding方法](./deployment/model/rounding/README.md)
- [如何合并模型](./deployment/model/merge_config_parameters/README.cn.md)
- INT8量化方法

## 模型压缩
基于PaddlePaddle框架，可以使用模型压缩训练进一步裁剪模型的大小。

- [Pruning稀疏化方法](./model_compression/pruning/README.md)

## 性能数据
我们列出一些移动设备上的性能测试数据，给用户参考和对比。

- [Mobilenet模型性能数据](./benchmark/README.md)
- ENet模型性能数据
- [DepthwiseConvolution优化效果](https://github.com/hedaoyuan/Function/blob/master/src/conv/README.md)

本教程由[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)创作，采用[Apache-2.0 license](LICENSE)许可协议进行许可。
