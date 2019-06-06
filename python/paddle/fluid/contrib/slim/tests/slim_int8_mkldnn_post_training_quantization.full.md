# SLIM INT8 MKL-DNN Post Training Quantization

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a FP32 ProgramDesc with FP32 weights to an INT8 ProgramDesc with FP32 with weights on ResNet-50, ResNet101, MobileNet-V1, Mobilenet-V2, GoogleNet, VGG16 and VGG19. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle Slim and show the the above 7 models' results in accuracy and performance.

## 0. Install PaddlePaddle
Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you [build from source](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/install/compile/compile_Ubuntu_en.md), please use the following cmake arguments.

```
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON  -DWITH_TESTING=ON  -WITH_FLUID_ONLY=ON  -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON -DWITH_SLIM_MKLDNN_FULL_TEST=ON
```

Note: MKL-DNN and MKL are required.

## 1. Enable SLIM INT8 MKL-DNN quantization
For reference, please examine the code of unit test enclosed in [test_mkldnn_int8_quantization_strategy.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/test_mkldnn_int8_quantization_strategy.py).

* ### Create yaml config
For reference, please examine the code of unit test enclosed in  [config_mkldnn_int8.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/quantization/config_mkldnn_int8.yaml)
``` yaml
version: 1.0
strategies:
    mkldnn_post_training_strategy:
        class: 'MKLDNNPostTrainingQuantStrategy'
        int8_model_save_path: 'OUTPUT_PATH'
        fp32_model_path: 'MODEL_PATH'
        cpu_math_library_num_threads: 1
compressor:
    epoch: 0
    checkpoint_path: ''
    strategies:
        - mkldnn_post_training_strategy
```

* ### Create quantize config by analysis config
We enable the MKL-DNN quantization procedure by calling an appropriate method from analysis config. Afterwards, all the required quantization parameters (quantization op names, quantization strategies etc.) can be set through quantizer config which is present in the analysis config. It is also necessary to specify a pre-processed warmup dataset and desired batch size.

```yaml
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
| ResNet-50  | Full ImageNet Val  |  * |  * | * |
| MobileNet-V1 | Full ImageNet Val  |  * |  * |  * |

   >**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 1 on single core)**

| Model  | Dataset  | FP32 Throughput  | INT8 Throughput  |  Ratio(INT8/FP32)  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ResNet-50  | Full ImageNet Val  |   images/s |  images/s | * |
| MobileNet-V1 | Full ImageNet Val  |  images/s |  images/s | *  |

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
python ./paddle/python/paddle/fluid/contrib/slim/tests/test_mkldnn_int8_quantization_strategy.py --infer_model third_party/inference_demo/int8v2/resnet50/model --infer_data /path/to/converted/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```
   * ##### Mobilenet-v1 Full dataset benchmark
```bash
python ./paddle/python/paddle/fluid/contrib/slim/tests/test_mkldnn_int8_quantization_strategy.py --infer_model third_party/inference_demo/int8v2/mobilenet/model --infer_data /path/to/converted/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```
