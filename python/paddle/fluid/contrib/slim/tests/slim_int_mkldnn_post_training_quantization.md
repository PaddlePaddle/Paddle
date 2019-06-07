# SLIM INT8 MKL-DNN Post Training Quantization

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a FP32 ProgramDesc with FP32 weights to an INT8 ProgramDesc with FP32 with weights on ResNet-50, ResNet101, MobileNet-V1, Mobilenet-V2, GoogleNet, VGG16 and VGG19. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle Slim and show the the above 7 models' results in accuracy and performance.

## 0. Install PaddlePaddle
Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you [build from source](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/install/compile/compile_Ubuntu_en.md), please use the following cmake arguments.

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON  -DWITH_TESTING=ON  -WITH_FLUID_ONLY=ON  -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON -DWITH_SLIM_MKLDNN_FULL_TEST=ON
```

Note: MKL-DNN and MKL are required.

## 2. Accuracy and Performance benchmark

We provide the results of accuracy and performance measured on Intel(R) Xeon(R) Gold 8280 on single core.

   >**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 8280**

| Model  | Dataset  | FP32 Accuracy  | INT8 Accuracy  | Accuracy Diff  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| GoogleNet    | Full ImageNet Val  |  70.50% |  70.20% |  0.30% |
| MobileNet-V1 | Full ImageNet Val  |  70.78% |  70.36% |  0.42% |
| Mobilenet-V2 | Full ImageNet Val  |  71.90% |  71.57% |  0.33% |
| ResNet101    | Full ImageNet Val  |  77.50% |  77.53% |  -0.03% |
| ResNet-50    | Full ImageNet Val  |  76.63% |  76.48% |  0.15%  |
| VGG16        | Full ImageNet Val  |  72.08% |  72.01% |  0.07% |
| VGG19        | Full ImageNet Val  |  72.56% |  72.56% |  0.01% |

   >**II. Throughput on Intel(R) Xeon(R) Gold 8280 (batch size 1 on single core)**

| Model  | Dataset  | FP32 Throughput  | INT8 Throughput  |  Ratio(INT8/FP32)  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| GoogleNet    | Full ImageNet Val  |  170.04   images/s |  558.9    images/s |  3.29  |
| MobileNet-V1 | Full ImageNet Val  |  31.63    images/s |  761.94   images/s |  24.09 |
| MobileNet-V2 | Full ImageNet Val  |  19.18    images/s |  598.53   images/s |  31.21 |
| ResNet101    | Full ImageNet Val  |  22.2     images/s |  339.9    images/s |  15.31 |
| ResNet-50    | Full ImageNet Val  |  43.5     images/s |  483.39   images/s |  11.11 |
| VGG16        | Full ImageNet Val  |  10.94    images/s |  195.75   images/s |  17.89 |
| VGG19        | Full ImageNet Val  |  20.25    images/s |  169.4    images/s |  8.37  |

Notes:
* CPU turbo off.

## 3. Commands to reproduce the above accuracy and performance benchmark
* #### Full dataset (Single core)
   * ##### Download full ImageNet Validation Dataset
```bash
cd /PATH/TO/PADDLE/build
python ../paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in ~/.cache/paddle/dataset/int8/download/int8_full_val.bin
   * ##### ResNet50 Full dataset benchmark
```bash
cd /PATH/TO/PADDLE/build/python/paddle/fluid/contrib/slim/tests/
python ./test_mkldnn_int8_quantization_strategy.py --infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/resnet50/model --infer_data /path/to/converted/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```
   * ##### Mobilenet-v1 Full dataset benchmark
```bash
cd /PATH/TO/PADDLE/build/python/paddle/fluid/contrib/slim/tests/
python ./test_mkldnn_int8_quantization_strategy.py --infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/mobilenet/model --infer_data /path/to/converted/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```
