# INT8 MKL-DNN quantization

This document describes how to use Paddle inference Engine to convert the FP32 model to INT8 model on ResNet-50 and MobileNet-V1. We provide the instructions on enabling INT8 MKL-DNN quantization in Paddle inference and show the ResNet-50 and MobileNet-V1 results in accuracy and performance.

## 0. Install PaddlePaddle

Follow PaddlePaddle [installation instruction](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification#installation) to install PaddlePaddle. If you build PaddlePaddle yourself, please use the following cmake arguments.

```bash
cmake ..  -DWITH_TESTING=ON -WITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON

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

>**Dataset: ILSVRC2012 Validation dataset**

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model        | FP32 Accuracy   | INT8 Accuracy   | Accuracy Diff   |
| :----------: | :-------------: | :------------:  | :--------------:|
| GoogleNet    |  70.51%         |  70.20%         |  0.31%          |
| MobileNet-V1 |  70.78%         |  70.36%         |  0.42%          |
| MobileNet-V2 |  71.91%         |  71.58%         |  0.33%          |
| ResNet-101   |  77.49%         |  77.53%         | -0.04%          |
| ResNet-50    |  76.62%         |  76.47%         |  0.15%          |
| VGG16        |  72.08%         |  72.01%         |  0.07%          |
| VGG19        |  72.57%         |  72.56%         |  0.01%          |

>**II. Throughput on Intel(R) Xeon(R) Gold 6271 (batch size 1 on single core)**

| Model        | FP32 Throughput(images/s)  | INT8 Throughput(images/s) | Ratio(INT8/FP32)|
| :-----------:| :------------:             | :------------:            | :------------:  |
| GoogleNet    |    38.62                   |    85.20                  |   2.21          |
| MobileNet-V1 |    93.05                   |   270.36                  |   2.91          |
| MobileNet-V2 |   117.35                   |   248.70                  |   2.12          |
| ResNet-101   |     8.19                   |    30.56                  |   3.73          |
| ResNet-50    |    15.35                   |    58.41                  |   3.80          |
| VGG16        |     3.97                   |    11.36                  |   2.86          |
| VGG19        |     3.20                   |     9.80                  |   3.06          |

Notes:

* Measurement of accuracy requires a model which accepts two inputs: data and labels.

* Different sampling batch size data may cause slight difference on INT8 top accuracy.
* CAPI performance data is better than python API performance data because of the python overhead. Especially for the small computational model, python overhead will be more obvious.

## 3. Commands to reproduce the above accuracy and performance benchmark

Two steps to reproduce the above-mentioned accuracy results, and we take GoogleNet benchmark as an example:

* ### Prepare dataset

Running the following commands to download and preprocess the ILSVRC2012 Validation dataset.

```bash
cd /PATH/TO/PADDLE/build
python ../paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```

Then the ILSVRC2012 Validation dataset will be preprocessed and saved by default in `~/.cache/paddle/dataset/int8/download/int8_full_val.bin`

* ### Commands to reproduce GoogleNet benchmark

You can run `test_analyzer_int8_imagenet_classification` with the following arguments to reproduce the accuracy result on GoogleNet.

```bash
./paddle/fluid/inference/tests/api/test_analyzer_int8_imagenet_classification --infer_model=third_party/inference_demo/int8v2/resnet50/model --infer_data=/~/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
```

To verify all the 7 models, you need to set the parameter of `--infer_model` to one of the following values in command line:

```bash
--infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/MODEL_NAME/model
```

MODEL_NAME = googlenet, mobilenet, mobilenetv2, resnet101, resnet50, vgg16, vgg19
