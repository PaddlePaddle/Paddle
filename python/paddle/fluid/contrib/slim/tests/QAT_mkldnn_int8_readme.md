# SLIM Quantization-aware training (QAT) on INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a quantization-aware trained model to an INT8 MKL-DNN. In **Release 1.5**, we have released the QAT MKL-DNN 1.0 which enabled the INT8 MKL-DNN kernel for QAT trained model within 0.05% accuracy diff on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. In **Release 1.6**, QAT MKL-DNN 2.0, we did the performance optimization based on fake QAT models: ResNet50, ResNet101, Mobilenet-v1, Mobilenet-v2, VGG16 and VGG19 with the minor accuracy drop. Compared with Release 1.5, the QAT MKL-DNN 2.0 got better performance gain on inference compared with fake QAT models but got a little bit bigger accuracy diff. We provide the accuracy benchmark both for QAT MKL-DNN 1.0 and QAT MKL-DNN 2.0, and performance benchmark on QAT MKL-DNN 2.0.  
MKL-DNN INT8 quantization performance gain can only be obtained with AVX512 series CPU servers.

## 0. Prerequisite
You need to install at least PaddlePaddle-1.6 python package `pip install paddlepaddle==1.6`.

## 1. How to generate INT8 MKL-DNN QAT model
You can refer to the unit test in [test_quantization_mkldnn_pass.py](test_quantization_mkldnn_pass.py). Users firstly use PaddleSlim quantization strategy to get a saved fake QAT model by [QuantizationFreezePass](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api), then use the `FakeQAT2MkldnnINT8KernelPass` to get the graph which can be run with MKL-DNN INT8 kernel. In Paddle Release 1.6, this pass supports `conv2d` and `depthwise_conv2d` ops with channel-wise quantization for weights. Apart from it, another pass called FakeQAT2MkldnnINT8PerfPass is available for use. This pass allows users to transform their QAT INT8 model into a highly performance-optimized model that is ran using INT8 MKL-DNN kernels.

```python
    import paddle.fluid as fluid
    from paddle.fluid.contrib.slim.quantization import FakeQAT2MkldnnINT8KernelPass
    from paddle.fluid.framework import IrGraph
    from paddle.fluid import core	
    
    # Create the IrGraph by Program
    graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
    place = fluid.CPUPlace()
    # Convert the IrGraph to MKL-DNN supported INT8 IrGraph by using
    # QAT MKL-DNN 1.0
    # FakeQAT2MkldnnINT8KernelPass
    mkldnn_pass = FakeQAT2MkldnnINT8KernelPass(fluid.global_scope(), place)
    # Apply FakeQAT2MkldnnINT8KernelPass to IrGraph
    mkldnn_pass.apply(graph)
    # QAT MKL-DNN 2.0
    # FakeQAT2MkldnnINT8PerfPass
    mkldnn_pass = FakeQAT2MkldnnINT8PerfPass(fluid.global_scope(), place, fluid.core, False)
    # Apply FakeQAT2MkldnnINT8PerfPass to IrGraph
    mkldnn_pass.apply(graph)

```

## 2. Accuracy benchmark

>**I. QAT1.0 MKL_DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model        | Fake QAT Top1 Accuracy | Fake QAT Top5 Accuracy |MKL-DNN INT8 Top1 Accuracy |  Top1 Diff   | MKL-DNN INT8 Top5 Accuracy | Top5 Diff  |
| :----------: | :--------------------: | :--------------------: |:-----------------------:  | :----------: | :------------------------: | :--------: |
| GoogleNet    |         70.40%         |          89.46%        |           70.39%          |    -0.01%    |           89.46%           |   0.00%    |
| MobileNet-V1 |         70.84%         |          89.58%        |           70.85%          |    +0.01%    |           89.61%           |  +0.03%    |
| MobileNet-V2 |         72.07%         |          90.71%        |           72.06%          |    -0.01%    |           90.69%           |  -0.02%    |
| ResNet-101   |         77.52%         |          93.68%        |           77.52%          |     0.00%    |           93.67%           |  -0.01%    |
| ResNet-50    |         76.66%         |          93.08%        |           76.62%          |    -0.04%    |           93.10%           |  +0.02%    |
| VGG16        |                        |                        |                           |              |                            |            |
| VGG19        |                        |                        |                           |              |                            |            |

Notes:

* MKL-DNN and MKL are required. AVX512 CPU server is required.

>**II. QAT2.0 MKL-DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model        | Fake QAT Top1 Accuracy | Fake QAT Top5 Accuracy |MKL-DNN INT8 Top1 Accuracy |  Top1 Diff  | MKL-DNN INT8 Top5 Accuracy | Top5 Diff |
| :----------: | :--------------------: | :--------------------: |:-----------------------:  | :----------:| :------------------------: | :--------:|
| MobileNet-V1 |         70.70%         |          89.48%        |           70.78%          |       |           89.39%           |      |
| MobileNet-V2 |         72.09%         |          90.69%        |           72.17%          |       |           90.63%           |      |
| ResNet101    |         77.83%         |          93.58%        |           77.59%          |        |           93.54%           |      |
| ResNet50     |         76.64%         |          93.01%        |           76.53%          |        |           92.98%           |       |
| VGG16        |         71.75%         |          89.95%        |           71.75%          |        |           89.73%           |      |
| VGG19        |         72.29%         |          90.19%        |           72.09%          |        |           90.13%           |      |

>**III. QAT2.0 MKL-DNN C-API Performance on Intel(R) Xeon(R) Gold 6271**

| Model        | FP32 Post Training Throughput(images/s)| INT8 QAT Throughput(images/s) | Ratio(INT8/FP32)|
| :-----------:| :-----------------------------------:  | :----------------------------: | :------------:  |
| MobileNet-V1 |       77.11                            |       228.49                   |                 |
| MobileNet-V2 |       92.95                            |       203.15                   |                 |
| ResNet101    |       7.23                             |       26.37                    |                 |
| ResNet50     |       13.25                            |       48.12                    |                 |
| VGG16        |                                        |       10.24                    |                 |
| VGG19        |                                        |                                |                 |

## 3. How to reproduce the results
Three steps to reproduce the above-mentioned accuracy results, and we take ResNet50 benchmark as an example:
 * ### Prepare dataset
```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `~/.cache/paddle/dataset/int8/download/int8_full_val.bin`
 * ### Prepare model
You can run the following commands to download ResNet50 model. The exemplary code snippet provided below downloads a ResNet50 QAT model. The reason for having two different versions of the same model originates from having a two different QAT training strategies: One for an non-optimized and second for an optimized graph transform which correspond to QAT 1.0 and QAT 2.0 respectively.

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
# uncomment for QAT 1.0 MKL-DNN
# export MODEL_NAME=ResNet50
# export MODEL_FILE_NAME= QAT_models/${MODEL_NAME}_qat_model.tar.gz
# uncomment for QAT 2.0 MKL-DNN
# export MODEL_NAME=resnet50
# export MODEL_FILE_NAME= QAT2_models/${MODEL_NAME}_quant.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/${MODEL_FILE_NAME}
```

Unzip the downloaded model to the folder. To verify all the 7 models, you need to set `MODEL_NAME` to one of the following values in command line:
```text
QAT MKL-DNN 1.0
MODEL_NAME=ResNet50, ResNet101, GoogleNet, MobileNetV1, MobileNetV2, VGG16, VGG19
QAT MKL-DNN 2.0
MODEL_NAME=resnet50, resnet101, mobilenetv1, mobilenetv2, vgg16, vgg19 
```
* ### Commands to reproduce benchmark
You can run `qat_int8_comparison.py` with the following arguments to reproduce the accuracy result on ResNet50. The difference of command line between the QAT MKL-DNN 1.0 and QAT MKL-DNN 2.0 is that we use argument `qat2` to enable QAT MKL-DNN 2.0. To perform QAT MKL-DNN 2.0 the performance test, the environmental variable `OMP_NUM_THREADS=1` and `batch_size=1` parameter should be set.
>*QAT 1.0*

```bash
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_comparison.py --qat_model=/PATH/TO/DOWNLOAD/MODEL/${MODEL_NAME}/model --infer_data=~/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.001
```
>*QAT 2.0*

- Accuracy benchamrk
```bash
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_comparison.py --qat_model=/PATH/TO/DOWNLOAD/MODEL/${MODEL_NAME} --infer_data=~/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.01 --qat2
```

* Performance benchmark

```bash
# 1. Save QAT_INT8 model
python  /home/danqing/repo/Paddle/python/paddle/fluid/contrib/slim/tests/qat_save_model.py --qat_model_path /PATH/TO/DOWNLOAD/MODEL/${QAT2_MODEL_NAME} --qat_int8_model_path /PATH/TO/${QAT2_MODEL_NAME}_qat_int8

# 2. Run the performance benchmark C-API 
/home/danqing/repo/Paddle/build/paddle/fluid/inference/tests/api/test_analyzer_qat_image_classification ARGS --enable_fp32=false --with_label=false --int8_model=/PATH/TO/${QAT2_MODEL_NAME}_qat_int8 --infer_data=~/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
```

> Notes: Due to a high amount of images contained in `int8_full_val.bin` dataset (50 000), the accuracy benchmark which includes comparison of unoptimized and optimized QAT model may last long (even several hours). To accelerate the process, it is recommended to set `OMP_NUM_THREADS` to the max number of physical cores available on the server. Since performance test doesn't require running through the whole dataset, it is sufficient to keep the number of iterations to as low as 1000, with batch size and `OMP_NUM_THRADS` both set to 1.
