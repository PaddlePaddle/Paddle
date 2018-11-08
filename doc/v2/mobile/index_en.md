# Mobile

Here mainly describes how to deploy PaddlePaddle to the mobile end, as well as some deployment optimization methods and some benchmark.

## Build PaddlePaddle
- [Build PaddlePaddle for Android](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_android_en.md)
- [Build PaddlePaddle for IOS](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_ios_en.md)
- [Build PaddlePaddle for Raspberry Pi3](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/mobile/cross_compiling_for_raspberry_en.md)
- Build PaddlePaddle for NVIDIA Driver PX2

## Demo
- [A command-line inference demo.](./benchmark/tool/C/README.md)
- [iOS demo of PDCamera](./Demo/iOS/AICamera/README.md)

## Deployment optimization methods
Optimization for the library:

- [How to build PaddlePaddle mobile inference library with minimum size.](./deployment/library/build_for_minimum_size.md)

Optimization for models：

- [Merge batch normalization layers](./deployment/model/merge_batch_normalization/README.md)
- [Compress the model based on rounding](./deployment/model/rounding/README.md)
- [Merge model's config and parameters](./deployment/model/merge_config_parameters/README.md)
- How to deploy int8 model in mobile inference with PaddlePaddle

## Model compression
- [How to use pruning to train smaller model](./model_compression/pruning/README.md)

## PaddlePaddle mobile benchmark
- [Benchmark of Mobilenet](./benchmark/README.md)
- Benchmark of ENet
- [Benchmark of DepthwiseConvolution](https://github.com/hedaoyuan/Function/blob/master/src/conv/README.md)

This tutorial is contributed by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) and licensed under the [Apache-2.0 license](LICENSE).
