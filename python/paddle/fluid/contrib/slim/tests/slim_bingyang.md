# Paddle Slim support MKL-DNN post-training quantization test

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert the FP32 model to INT8 model on ResNet-50, ResNet101, MobileNet-V1, Mobilenet-V2, GoogleNet, VGG16 and VGG19. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle Slim and show the the above 7 models' results in accuracy and performance.

## 0. Install PaddlePaddle 

Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you [build from source](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/install/compile/compile_Ubuntu_en.md), please use the following cmake arguments.
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON  -DWITH_TESTING=ON  -WITH_FLUID_ONLY=ON  -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON
```
Note: MKLDNN and MKL are required.
