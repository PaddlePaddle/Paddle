# SLIM Quantization-aware training (QAT) on INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a quantization-aware trained model to INT8 MKL-DNN quantized model. In **Release 1.5**, we have released the QAT1.0 MKL-DNN which enabled the INT8 MKL-DNN kernel for QAT trained model within 0.05% accuracy diff on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. In **Release 1.6**, QAT2.0 MKL-DNN, we did the performance optimization based on fake QAT models: ResNet50, ResNet101, Mobilenet-v1, Mobilenet-v2, VGG16 and VGG19 with the minor accuracy drop. Compared with Release 1.5, the QAT2.0 MKL-DNN got better performance gain on inference compared with fake QAT models but got a little bit bigger accuracy diff. In **Release 1.7**, a support for [Ernie (NLP) QAT trained model](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie) was added to the QAT2.0 MKL-DNN. We provide the accuracy benchmark both for QAT1.0 MKL-DNN and QAT2.0 MKL-DNN, and performance benchmark on QAT2.0 MKL-DNN.  

Notes:

* MKL-DNN and MKL are required. The performance gain can only be obtained with AVX512 series CPU servers.
* INT8 accuracy is best on CPU servers supporting AVX512 VNNI extension.

## 0. Prerequisite
You need to install at least PaddlePaddle-1.7.1 python package `pip install paddlepaddle==1.7.1`.

## 1. How to generate your INT8 model

PaddlePaddle provide two different kinds of quantization: [Post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) and Quantization-aware training. In post-training quantization, we calibrate the scales of quantizable ops on warmup batches, because of this, the accuracy may be influenced by warmup batch size. In quantization-aware training, we insert fake quantization ops and train the model and calibrate the sacles during the whole process of training. In theory Quantization-aware training is more accurate. In both quantization methods, we do all potential fuses, such as conv + elementwise_add, etc. so that the performance will be improved further. In this document, we mainly focus on Quantization-aware training.

### How to transform ordinary models to fake QAT models

You can refer to the unit test in [test_quantization_mkldnn_pass.py](test_quantization_mkldnn_pass.py) [This test need to be improved]. Users firstly use passes in quantization_strategy.py and quantization_pass.py to insert fake ops and generate fake QAT model. 

Until now, we support following quantizable ops:

   * QuantizationTransformPass: `conv2d`, `depthwise_conv2d`, `mul`,
   * AddQuantDequantPass: `pool2d`, `elementwise_add`, `concat`, `softmax`, `argmax`, `transpose`,
        `equal`, `gather`, `greater_equal`, `greater_than`, `less_equal`,
        `less_than`, `mean`, `not_equal`, `reshape`, `reshape2`,
        `bilinear_interp`, `nearest_interp`, `trilinear_interp`, `slice`,
        `squeeze`, `elementwise_sub`, `mul`, `matmul`

The fake quantization ops inserted in quantization-aware training could have input `X` and `Inscale`, output `Out` and `OutScale`, attribute `bit_length`(means how many bits one wants to use for quantized ops. Until now, MKLDNN Quantization-aware training only support 8 bits). The formula to calculate the output is as follows:
```bash
$$scale = max(max(abs(x)), history_abs_max)$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$
```

### How to quantize fake QAT models to INT8 models

* Simply speaking, user can use the `save_qat_model.py` script to transform the fake QAT model to INT8 model. The `--quantized_ops` is by default set to all MKL-DNN quantizable ops. 
```bash
python ../python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=/PATH/TO/DOWNLOAD/MODEL/${QAT2_MODEL_NAME} --int8_model_save_path=/PATH/TO/${QAT2_MODEL_NAME}_qat_int8
```
* If user want to choose specific ops to quantize and leave others as fp32 ops, use the `Qat2Int8MkldnnPass` to convert. Until release/1.7, the pass supports following quantizable ops: `conv2d`, `depthwise_conv2d`, `pool2d`, `fc`, `reshape2`, `transpose2`, with all possible fusions. 

```python
    import paddle.fluid as fluid
    from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass
    from paddle.fluid.framework import IrGraph
    from paddle.fluid import core	
    
    # Create the IrGraph by Program
    graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
    place = fluid.CPUPlace()
    # QAT2.0 MKL-DNN
    # Qat2Int8MkldnnPass, it requires a list of operators to be quantized
    mkldnn_pass = Qat2Int8MkldnnPass({'conv2d', 'pool2d',...}, fluid.global_scope(), place, fluid.core, False)
    # Apply Qat2Int8MkldnnPass to IrGraph
    mkldnn_pass.apply(graph)
```

## 2. Accuracy and Performance benchmark

### Image classification QAT models benchmark resutls

>**I. QAT2.0 MKL-DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

|     Model    | Fake QAT Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | Fake QAT Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|:----------------------:|:----------------------:|:---------:|
| MobileNet-V1 |         70.72%         |         70.78%         |   +0.06%  |         89.47%         |         89.39%         |   -0.08%  |
| MobileNet-V2 |         72.07%         |         72.17%         |   +0.10%  |         90.65%         |         90.63%         |   -0.02%  |
|   ResNet101  |         77.86%         |         77.59%         |   -0.27%  |         93.54%         |         93.54%         |   0.00%   |
|   ResNet50   |         76.62%         |         76.53%         |   -0.09%  |         93.01%         |         92.98%         |   -0.03%  |
|     VGG16    |         71.74%         |         71.75%         |   +0.01%  |         89.96%         |         89.73%         |   -0.23%  |
|     VGG19    |         72.30%         |         72.09%         |   -0.21%  |         90.19%         |         90.13%         |   -0.06%  |


>**II. QAT2.0 MKL-DNN C-API Performance on Intel(R) Xeon(R) Gold 6271**

|     Model    |        FP32 Optimized Throughput     |       INT8 QAT Throughput     | Ratio(INT8/FP32) |
|:------------:|:------------------------------------:|:-----------------------------:|:----------------:|
| MobileNet-V1 |                 73.98                |             227.73            |       3.08       |
| MobileNet-V2 |                 86.59                |             206.74            |       2.39       |
|   ResNet101  |                 7.15                 |             26.69             |       3.73       |
|   ResNet50   |                 13.15                |             49.33             |       3.75       |
|     VGG16    |                 3.34                 |             10.15             |       3.04       |
|     VGG19    |                 2.83                 |              8.67             |       3.07       |

Notes:

* FP32 Optimized Throughput (images/s) is from [int8_mkldnn_quantization.md](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md).

### Natural language processing QAT models benchmark resutls

>**I. Ernie QAT2.0 MKL-DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |      80.20%            |        79.96%         |  -0.24%   |               


>**II. Ernie QAT2.0 MKL-DNN Performance on Intel(R) Xeon(R) Gold 6271**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:---------:|
| 1 thread     |        252.131         |         93.8023    |     2.687x   |
| 20 threads   |        29.1853         |         17.3765    |     1.680x   |


## 3. How to reproduce the benchmark results
To reproduce the above-mentioned Image Classification models accuracy and performance, follow steps as below (taking ResNet50 as an example).
To reproduce NLP models results (Ernie), please follow [How to reproduce Ernie QAT results on MKL-DNN](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn/README.md).

### Prepare dataset

In order to download the dataset for image classification models benchmarking, execute:

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

### Prepare model

You can run the following commands to download ResNet50 model. The exemplary code snippet provided below downloads a ResNet50 QAT model. The reason for having two different versions of the same model originates from having two different QAT training strategies: One for an non-optimized and second for an optimized graph transform which correspond to QAT1.0 and QAT2.0 respectively.

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
# uncomment for QAT2.0 MKL-DNN
# export MODEL_NAME=resnet50
# export MODEL_FILE_NAME=QAT2_models/${MODEL_NAME}_quant.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/${MODEL_FILE_NAME}
```

Extract the downloaded model to the folder. To verify all the 7 models, you need to set `MODEL_NAME` to one of the following values in command line:
```text
QAT2.0 models
MODEL_NAME=resnet50, resnet101, mobilenetv1, mobilenetv2, vgg16, vgg19
```
### Commands to reproduce benchmark

You can use the `qat_int8_image_classification_comparison.py` script to reproduce the accuracy result on ResNet50. The difference between commands usedin the QAT1.0 MKL-DNN and QAT2.0 MKL-DNN is that for QAT2.0 MKL-DNN two additional options are required: the `--qat2` option to enable QAT2.0 MKL-DNN, and the `--quantized_ops` option with a comma-separated list of operators to be quantized. To perform the QAT2.0 MKL-DNN performance test, the environment variable `OMP_NUM_THREADS=1` and `--batch_size=1` option should be set.

>*QAT2.0*

- Accuracy benchamrk command on QAT2.0 models
```bash
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_image_classification_comparison.py ----qat_model=/PATH/TO/DOWNLOAD/MODEL/${MODEL_NAME}_quant --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.01 --qat2
```

* Performance benchmark command on QAT2.0 models

In order to run performance benchmark, follow the steps below.

1. Save QAT2.0 INT8 model. You can use the script `save_qat_model.py` for this purpose. It also requires the option `--quantized_ops`  to indicate which operators are to be quantized.

   ```bash
   cd /PATH/TO/PADDLE/build
   python ../python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=/PATH/TO/DOWNLOAD/MODEL/${QAT2_MODEL_NAME} --int8_model_save_path=/PATH/TO/${QAT2_MODEL_NAME}_qat_int8
   ```

2. Run the QAT2.0 C-API test for performance benchmark.

   ```bash
   cd /PATH/TO/PADDLE/build
   OMP_NUM_THREADS=1 paddle/fluid/inference/tests/api/test_analyzer_qat_image_classification ARGS --enable_fp32=false --with_accuracy_layer=false --int8_model=/PATH/TO/${QAT2_MODEL_NAME}_qat_int8 --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
   ```

> Notes: Due to a large amount of images contained in `int8_full_val.bin` dataset (50 000), the accuracy benchmark which includes comparison of unoptimized and optimized QAT model may last long (even several hours). To accelerate the process, it is recommended to set `OMP_NUM_THREADS` to the max number of physical cores available on the server.
