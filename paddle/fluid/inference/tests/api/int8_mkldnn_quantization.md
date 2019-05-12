# INT8 MKL-DNN quantization 

This document describes how to use Paddle inference Engine to convert the FP32 model to INT8 model on ResNet-50 and MobileNet-V1. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle inference and show the ResNet-50 and MobileNet-V1 results in accuracy and performance.

## 0. Install PaddlePaddle 
Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you build PaddlePaddle yourself, please use the following cmake arguments. 
```
cmake ..  -DWITH_TESTING=ON -WITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_MKL=ON  -WITH_SWIG_PY=OFF -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON

```  
Note: MKL-DNN and MKL are required.

## 1. Enable INT8 MKL-DNN quantization 
For reference, please examine the code of unit test enclosed in [analyzer_int8_image_classification_tester.cc](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/analyzer_int8_image_classification_tester.cc).

* ### Create Analysis config
INT8 quantization is one of the optimizations in analysis config. More information about analysis config can be found [here](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/deploy/inference/native_infer_en.md#upgrade-performance-based-on-contribanalysisconfig-prerelease) 

* ### Create quantize config by analysis config
We enable the MKL-DNN quantization procedure by calling an appropriate method from analysis config. Afterwards, all the required quantization parameters (quantization op names, quantization strategies etc.) can be set through quantizer config which is present in the analysis config. It is also necessary to specify a pre-processed warmup dataset and desired batch size.

```cpp
//Enable MKL-DNN quantization
cfg.EnableMkldnnQuantizer();

//use analysis config to call the MKL-DNN quantization config
cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data); 
cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(100);
```

## 2. Accuracy and Performance benchmark

We provide the results of accuracy and performance measured on Intel(R) Xeon(R) Gold 6271 on single core.

   >**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model  | Dataset  | FP32 Accuracy  | INT8 Accuracy  | Accuracy Diff  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ResNet-50  | Full ImageNet Val  | 76.63%  | 76.48%  | 0.15% |
| MobileNet-V1 | Full ImageNet Val  | 70.78%  | 70.36%  | 0.42%  |

   >**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 1 on single core)**

| Model  | Dataset  | FP32 Throughput  | INT8 Throughput  |  Ratio(INT8/FP32)  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ResNet-50  | Full ImageNet Val  |  13.17 images/s | 49.84 images/s | 3.78 |
| MobileNet-V1 | Full ImageNet Val  | 75.49 images/s | 232.38 images/s | 3.07  |

Notes:
* Measurement of accuracy requires a model which accepts two inputs: data and labels.
* Different sampling batch size data may cause slight difference on INT8 top accuracy.
* CAPI performance data is better than python API performance data because of the python overhead. Especially for the small computational model, python overhead will be more obvious. 


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
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=third_party/inference_demo/int8v2/resnet50/model --infer_data=/path/to/converted/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
```
   * ##### Mobilenet-v1 Full dataset benchmark
```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_mobilenet --infer_model=third_party/inference_demo/int8v2/mobilenet/model --infer_data=/path/to/converted/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
```
