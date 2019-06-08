# SLIM Post-training quantization (INT8 MKL-DNN)

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a FP32 ProgramDesc with FP32 weights to an INT8 ProgramDesc with FP32 weights on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. Paddle Slim MKL-DNN post-training quantization strategy accuracy results on above 7 models are as follows:

## Accuracy benchmark

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model        | Dataset                        | FP32 Accuracy   | INT8 Accuracy   | Accuracy Diff   |
| :----------: | :----------------------------: | :-------------: | :------------:  | :--------------:|
| GoogleNet    | ILSVRC2012 Validation dataset  |  70.50%         |  70.20%         |  0.30%          |
| MobileNet-V1 | ILSVRC2012 Validation dataset  |  70.78%         |  70.36%         |  0.42%          |
| MobileNet-V2 | ILSVRC2012 Validation dataset  |  71.90%         |  71.57%         |  0.33%          |
| ResNet-101   | ILSVRC2012 Validation dataset  |  77.50%         |  77.53%         | -0.03%          |
| ResNet-50    | ILSVRC2012 Validation dataset  |  76.63%         |  76.48%         |  0.15%          |
| VGG16        | ILSVRC2012 Validation dataset  |  72.08%         |  72.01%         |  0.07%          |
| VGG19        | ILSVRC2012 Validation dataset  |  72.56%         |  72.56%         |  0.00%          |

Notes:

* MKL-DNN and MKL are required.

## Instructions to reproduce the above accuracy benchmark

### 0. Install PaddlePaddle

Follow PaddlePaddle [installation instruction](http://www.paddlepaddle.org/documentation/docs/en/1.4/beginners_guide/install/index_en.html) to install PaddlePaddle. If you [build from source](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/install/compile/compile_Ubuntu_en.md), please use the following cmake arguments.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_MKLDNN=ON  -DWITH_TESTING=ON -DWITH_INFERENCE_API_TEST=ON -DON_INFER=ON -DWITH_SLIM_MKLDNN_FULL_TEST=ON
```

### 1. Data Preparation

Running the following commands to download and preprocess the ILSVRC2012 Validation dataset.

```bash
cd /PATH/TO/PADDLE/build
python ../paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```

Then the ILSVRC2012 Validation dataset will be preprocessed and saved by default in ~/.cache/paddle/dataset/int8/download/int8_full_val.bin.

### 2. Commands

To reproduce the above-mentioned accuracy results, we take GoogleNet benchmark as an example:

``` bash
cd /PATH/TO/PADDLE/build/python/paddle/fluid/contrib/slim/tests/
python ./test_mkldnn_int8_quantization_strategy.py --infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/googlenet/model --infer_data ~/.cache/paddle/dataset/int8/download/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```

Notes:

* The above commands will cost maybe several hours in the prediction stage (include int8 prediction and fp32 prediction) since there have 50000 pictures need to be predicted in `int8_full_val.bin`.

To verify all the 7 models, you need to set the parameter of `--infer_model` to one of the following values in command line:

--infer_model /PATH/TO/PADDLE/build/third_party/inference_demo/int8v2/MODEL_NAME/model

MODEL_NAME = googlenet, mobilenet, mobilenetv2, resnet101, resnet50, vgg16,vgg19
