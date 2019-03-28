# INT8 MKLDNN quantization 

This document describes how to use Paddle inference to convert the FP32 model to INT8 model on ResNet-50 and MobileNet-V1. We provide the instructions on enabling INT8 MKLDNN quantization in Paddle inference and show the ResNet-50 and MobileNet-V1 results in accuracy and performance.

## 0. Prerequisite
You need to build install at least PaddlePaddle-1.4.The test dataset and model will downloaded automatically in the build process.

## 1. Enable INT8 MKLDNN quantization
* #### Prepare the quantization warmup data 

For the INT8 quantization , we should firstly  run several iterations of FP32 inference to calculate the quantization scales for each layers' inputs and outputs. FP32 running process will be executed with analysis predictor. This warmup run data is prepared for this running process.

```cpp
std::vector<std::vector<PaddleTensor>> input_slots_all;
SetInput(&input_slots_all); //user create the SetInput function to load the data from the file
```  

* #### Create quantize config by analysis config

We enable the MKLDNN quantization procedure by analysis config, and pass the warmup data we prepared before. Meanwhile, you can set all the quantization parameters by using quantize config. For example, set quantization op names, set quantization strategies etc.

```cpp
cfg.EnableMkldnnQuantizer();//Enable MKLDNN quantization
cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data); //use analysis config to call the MKLDNN quantization config
cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(100);
```

* #### Run the accuracy compare test and profile test

We provide two tests: one is to test if the accuracy drop is within 1% after MKLDNN quantization, the other is to test the performance with MKLDNN quantized INT8 model.

- To compare the top 1 accuracy before and after MKLDNN quantization
    
```cpp
CompareQuantizedAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
      reinterpret_cast<const PaddlePredictor::Config *>(&q_cfg),
      input_slots_all); 
```
- Performance benchmark with MKLDNN quantized INT8 model
    
```cpp
TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
```

## 2. Accuracy and Performance benchmark

We provide the results of accuracy and performance measured on Intel(R) Xeon(R) Gold 6271 (single core).

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
```
```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=/home/bingyang/paddle-latest/build/third_party/inference_demo/int8/resnet50/model --infer_data=/home/bingyang/paddle-latest/build/third_party/inference_demo/int8/data.txt --paddle_num_threads=1 
```
* #### Full dataset (Single core) (WIP)
```bash
python preprocess.py --data_dir=/path/to/converted/data.txt
```
```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=/home/bingyang/paddle-latest/build/third_party/inference_demo/int8/resnet50/model --infer_data=/path/to/converted/data.txt --paddle_num_threads=1 --test_all_data
```
* #### Full dataset (Multi-core) (WIP)
```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_resnet50 --infer_model=/home/bingyang/paddle-latest/build/third_party/inference_demo/int8/resnet50/model --infer_data=/path/to/converted/data.txt --paddle_num_threads=20 --test_all_data
```
   - Notes: This is an example command with 20 cores.
