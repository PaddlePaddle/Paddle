# INT8 MKLDNN quantization 

This document describes how to use Paddle inference to convert the FP32 model to INT8 model on ResNet-50 and MobileNet-V1. We provide the instructions on enabling INT8 MKLDNN quantization in Paddle inference and show the ResNet-50 and MobileNet-V1 results in accuracy and performance.

## 0. Install PaddlePaddle 
Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you build PaddlePaddle yourself, please use the following cmake arguments. 
```
cmake ..  -DWITH_TESTING=ON -WITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_MKL=ON  -WITH_SWIG_PY=OFF -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON

```  
Note: MKLDNN and MKL are required.

## 1. Enable INT8 MKLDNN quantization 
You can refer to the unit test in `analyzer_int8_image_classification_teser.cc`. 

* ### Create quantize config by analysis config

We enable the MKLDNN quantization procedure by analysis config, and pass the prepared data. Meanwhile, you can set all the quantization parameters by using quantize config. For example, set quantization op names, set quantization strategies etc.

```cpp
//Enable MKLDNN quantization
cfg.EnableMkldnnQuantizer();

//use analysis config to call the MKLDNN quantization config
cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data); 
cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(100);
```

## 2. Accuracy and Performance benchmark

We provide the results of accuracy and performance measured on Intel(R) Xeon(R) Gold 6271 on single core.

   >**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model  | Dataset  | FP32 Accuracy  | INT8 Accuracy  | Accuracy Diff  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ResNet-50  | Full ImageNet Val  |  Baidu QA  | Baidu QA  | Baidu QA |
| MobileNet-V1 | Full ImageNet Val  | Baidu QA  | Baidu QA  | Baidu QA  |

   >**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 1 on single core)**

| Model  | Dataset  | FP32 Throughput  | INT8 Throughput  |  Ratio(INT8/FP32)  |
| :------------: | :------------: | :------------: | :------------: | :------------: |
| ResNet-50  | Full ImageNet Val  |  Baidu QA images/s | Baidu QA images/s | Baidu QA |
| MobileNet-V1 | Full ImageNet Val  | Baidu QA images/s | Baidu QA images/s | Baidu QA  |

Notes:
* The accuracy measurement requires the model with `label`.

## 3. Commands to reproduce the above accuracy and performance benchmark

* #### Small dataset (Single core)
```bash
cd /PATH/TO/PADDLE/build
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=third_party/inference_demo/int8/resnet50/model --infer_data=third_party/inference_demo/int8/data.txt --paddle_num_threads=1 
```
* #### Full dataset (Single core) (WIP)
```bash
python preprocess.py --data_dir=/path/to/converted/data.txt
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=third_party/inference_demo/int8/resnet50/model --infer_data=/path/to/converted/data.txt --paddle_num_threads=1 --test_all_data
```
* #### Full dataset (Multi-core) (WIP)
```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=third_party/inference_demo/int8/resnet50/model --infer_data=/path/to/converted/data.txt --paddle_num_threads=20 --test_all_data
```
   - Notes: This is an example command with 20 cores.
