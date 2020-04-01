# SLIM Quantization-aware training (QAT) for INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a quantization-aware trained model into INT8 MKL-DNN quantized model.

In **Release 1.5**, we have released the first approach to the MKL-DNN-based quantization of QAT models, called QAT1. It enabled the `conv2d` and `mul` INT8 MKL-DNN kernels for QAT trained models (GoogleNet, MobileNetV1, MobileNetV2, ResNet50, ResNet101, VGG16, and VGG19) with 0.05% accuracy diff.

In **Release 1.6**, a new approach was introduced, called QAT2, which adds support for more performance optimizations and more INT8 MKL-DNN kernels. INT8 MKL-DNN models obtained using QAT2 have much better inference performance than using QAT1, with only a little bit bigger accuracy diff.

In **Release 1.7**, a support for [Ernie (NLP) QAT trained model](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie) was added to the QAT2.

In this document we focus on the QAT2 approach only. 

## 0. Prerequisites
* PaddlePaddle in version 1.7.1 or higher is required. It can be installed as a python package using
  ```pip install paddlepaddle==1.7.1``` command.

* MKL-DNN and MKL are required. The highest performance gain can be observed using CPU servers supporting AVX512 instructions.
* INT8 accuracy is best on CPU servers supporting AVX512 VNNI extension (e.g. CLX class Intel processors).

## 1. Introduction

There are two forms of quantization supported in PaddlePaddle: [post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) (PTQ) and quantization-aware training (QAT). PTQ is more automatic and requires less model preparation than QAT, but usually QAT gives better accuracy with similar performance. In this document we focus on QAT2 approach to the QAT and INT8 quantization.

## 2. How to turn an FP32 model into a QAT model?

A procedure on how to transform an FP32 model into a QAT model supported by the QAT2 approach is described in [this document](https://github.com/PaddlePaddle/PaddleSlim/blob/80c9fab3f419880dd19ca6ea30e0f46a2fedf6b3/demo/mkldnn_quant/quant_aware/PaddleCV_mkldnn_quantaware_tutorial.md).

## 3. How to turn a QAT model into an INT8 MKL-DNN model?

A QAT model can be transformed into an INT8 quantized model if it contains enough information about quantization scales for every quantized operator in the graph. The process of quantization is done by the `Qat2Int8MkldnnPass` pass which comprises several steps:

### Gathering scales

The information about the quantization scales is being collected from three types of operators:

* `fake_quantize_moving_average_abs_max` - imitates INT8 quantization of FP32 tensors, but keeps quantized output values as floats; is used before quantized operator (e.g. `conv2d`) to gather scale information for the op's input.
* `fake_dequantize_max_abs` - imitates dequantization of INT8 tensors back into floats; it is used after quantized operator, and contains scale used for the op's weights dequantization.
* `fake_quantize_dequantize_moving_average_abs_max` - imitates immediate quantization and dequantization; it can be used after a quantized operator to get the scale value for the op's output.

Notes:

1. As the next steps describe, quantization will be applied later to an optimized FP32 model. It means that quantization scales for inputs and outputs of each quantized operator have to be gathered for tensors which are inputs and outputs of already optimized or fused operators. For example, if a model contains the following sequence of tensors and operators in the graph
   ```... → input1 → conv2d → output1 → batch_norm → output2 → relu → output3 → ...```
   and we want to quantize the `conv2d` op, then after applying FP32 optimizations the sequence will become
   ```... → input1 → conv2d → output3 → ...```
   and the quantization scales have to be collected for the `input1` and `outpu3` tensors in the QAT model.
2. Quantization of the following operators is supported: `conv2d`, `depthwise_conv2d`, `mul`, `fc`, `pool2d`, `reshape2`, `transpose2`, `concat`.

### Removing fake operators

All the `fake_quantize_*` and `fake_dequantize_*` operators are being removed from the graph.

### Dequantizing weights

Weights of `conv2d` and `mul` operators are assumed to be fake-quantized (quantized, but kept as floats) in QAT models. Here, the information about the scale from `fake_dequantize_max_abs` operators is used to fake-dequantize the weights back to the full float range of values. At this moment the model becomes an unoptimized clean FP32 inference model.

### Optimizing FP32 graph

A series of standard optimization passes are being applied to the FP32 graph. This gives us an optimized FP32 inference model and we can proceed with INT8 quantization.

### Computing weight scales

After optimization fuses, the weight tensors of `conv2d` or `fc` operators are likely to have different values and require new quantization scales. The weights are static, i.e. they do not change during the inference process, and the scales can be calculated simply as a maximum of absolute values from the tensor. To improve the inference accuracy we calculate the scales for each output channel separately, getting an array of quantization scales for a weight tensor.

### Taking activations into account

The basic datatype used during INT8 inference is signed INT8, with possible values from -128 to 127. However, if `conv2d` or `fc` operator has `relu` or `relu6` activation integrated in it, the output of the operator is known to have non-negative values. In that case we use unsigned INT8 datatype for output tensors, with a wider range for positive values (0 to 255), improving the inference accuracy further.

### Propagation of scales

Some of the operators (e.g. `reshape2`, `transpose2`, `pool2d` with max pooling) transform the data without changing the quantization scale. For this reason we propagate the quantization scale values through these operators without any modifications. We propagate the quantization scales also through the `scale` operator, updating the quantization scale accordingly. This approach lets us minimize the number of `fake_quantize` and `fake_dequantize` operators in the graph, because the information about the scales required for the quantization process to succeed spreads between quantized operators.

### Applying quantization passes

Having gathered all the data needed for quantization we apply the `cpu_quantize_pass` which quantizes the graph, and the `cpu_quantize_squash_pass` which optimizes the INT8 graph.

## 4. Code example

The code snipped shows how the `Qat2Int8MkldnnPass` can be applied to a model graph:

```python
    import paddle.fluid as fluid
    from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass
    from paddle.fluid.framework import IrGraph
    from paddle.fluid import core	
    
    # Create the IrGraph by Program
    graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
    place = fluid.CPUPlace()
    # Convert the IrGraph to MKL-DNN supported INT8 IrGraph using the
    # Qat2Int8MkldnnPass. It requires a list of operators to be quantized
    mkldnn_pass = Qat2Int8MkldnnPass({'conv2d', 'pool2d'}, fluid.global_scope(), place, fluid.core, False)
    # Apply Qat2Int8MkldnnPass to IrGraph
    mkldnn_pass.apply(graph)

```

## 5. Accuracy and Performance benchmark

### Image classification models benchmark results

>**I. QAT2 MKL-DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

|     Model    | Fake QAT Top1 Accuracy | INT8 QAT Top1 Accuracy | Top1 Diff | Fake QAT Top5 Accuracy | INT8 QAT Top5 Accuracy | Top5 Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|:----------------------:|:----------------------:|:---------:|
| MobileNet-V1 |         70.72%         |         70.78%         |   +0.06%  |         89.47%         |         89.39%         |   -0.08%  |
| MobileNet-V2 |         72.07%         |         72.17%         |   +0.10%  |         90.65%         |         90.63%         |   -0.02%  |
|   ResNet101  |         77.86%         |         77.59%         |   -0.27%  |         93.54%         |         93.54%         |   0.00%   |
|   ResNet50   |         76.62%         |         76.53%         |   -0.09%  |         93.01%         |         92.98%         |   -0.03%  |
|     VGG16    |         71.74%         |         71.75%         |   +0.01%  |         89.96%         |         89.73%         |   -0.23%  |
|     VGG19    |         72.30%         |         72.09%         |   -0.21%  |         90.19%         |         90.13%         |   -0.06%  |


>**II. QAT2 MKL-DNN C-API Performance on Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 (images/s) | INT8 QAT (images/s) | Ratio (INT8/FP32) |
| :----------: | :-------------: | :-----------------: | :---------------: |
| MobileNet-V1 |      73.98      |       227.73        |       3.08        |
| MobileNet-V2 |      86.59      |       206.74        |       2.39        |
|  ResNet101   |      7.15       |        26.69        |       3.73        |
|   ResNet50   |      13.15      |        49.33        |       3.75        |
|    VGG16     |      3.34       |        10.15        |       3.04        |
|    VGG19     |      2.83       |        8.67         |       3.07        |

Notes:

* Performance FP32 (images/s) values come from [INT8 MKL-DNN post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) document.

### NLP models benchmark results

>**I. Ernie QAT2 MKL-DNN Accuracy on Intel(R) Xeon(R) Gold 6271**

|     Model    |  FP32 Accuracy | QAT INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |      80.20%            |        79.96%         |  -0.24%   |


>**II. Ernie QAT2 MKL-DNN Performance on Intel(R) Xeon(R) Gold 6271**

|     Threads  | FP32 Latency (ms) | QAT INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:---------:|
| 1 thread     |        252.131         |         93.8023    |     2.687x   |
| 20 threads   |        29.1853         |         17.3765    |     1.680x   |

## 6. How to reproduce the results

The steps below show, taking ResNet50 as an example, how to reproduce the above accuracy and performance results for Image Classification models. 
To reproduce NLP models results (Ernie), please follow [How to reproduce Ernie QAT results on MKL-DNN](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn/README.md).

### Prepare dataset

Download the dataset for image classification models benchmarking by executing:

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

### Prepare models

Run the following commands to download and extract QAT model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QAT_MODEL_NAME=resnet50
export QAT_MODEL_ARCHIVE=${QAT_MODEL_NAME}_quant.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT2_models/${QAT_MODEL_ARCHIVE}
mkdir ${QAT_MODEL_NAME} && tar -xvf ${QAT_MODEL_ARCHIVE} -C ${QAT_MODEL_NAME}
```

To download other QAT models, set the `QAT_MODEL_NAME` variable in the above commands to one of the values: `resnet101`, `mobilenetv1`, `mobilenetv2`, `vgg16`, `vgg19`.

Download clean FP32 model for accuracy comparison against the INT8 model:

```bash
cd /PATH/TO/DOWNLOAD/MODEL/
export FP32_MODEL_NAME=resnet50
export FP32_MODEL_ARCHIVE=${FP32_MODEL_NAME}_int8_model.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/${FP32_MODEL_ARCHIVE}
mkdir ${FP32_MODEL_NAME} && tar -xzvf ${FP32_MODEL_ARCHIVE} -C ${FP32_MODEL_NAME}
```

To download other FP32 models, set the `FP32_MODEL_NAME` variable to on of the values: `Res101`, `mobilenetv1`, `mobilenet_v2`, `VGG16`, and `VGG19`.

### Run benchmark

#### Accuracy benchmark commands

You can use the `qat2_int8_image_classification_comparison.py` script to reproduce the accuracy result of the INT8 QAT models. The following options are required:

* `--qat_model` - a path to a QAT model that will be transformed into INT8 model.
* `--fp32_model` - a path to an FP32 model whose accuracy will be measured and compared to the accuracy of the INT8 model.
* `--quantized_ops` - a comma-separated list of names of operators to be quantized. For Image Classification models mentioned above the list comprises of `conv2d` and `pool2d` operators.
* `--infer_data` - a path to the validation dataset.

```bash
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_image_classification_comparison.py --qat_model=/PATH/TO/DOWNLOADED/QAT/MODEL --fp32_model=/PATH/TO/DOWNLOADED/FP32/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.01 --quantized_ops="conv2d,pool2d"
```

#### Performance benchmark commands

To reproduce the performance results, the environment variable `OMP_NUM_THREADS=1` and `--batch_size=1` option should be set.

1. Transform the QAT model into INT8 model by applying the `Qat2Int8MkldnnPass` pass and save the result. You can use the script `save_qat_model.py` for this purpose. It also requires the option `--quantized_ops`  with a list of operators to be quantized.

   ```bash
   cd /PATH/TO/PADDLE/build
   python ../python/paddle/fluid/contrib/slim/tests/save_qat_model.py --qat_model_path=/PATH/TO/DOWNLOADED/QAT/MODEL --int8_model_save_path=/PATH/TO/SAVE/QAT/INT8/MODEL --quantized_ops="conv2d,pool2d"
   ```

2. Run the C-API test for performance benchmark.

   ```bash
   cd /PATH/TO/PADDLE/build
   OMP_NUM_THREADS=1 paddle/fluid/inference/tests/api/test_analyzer_qat_image_classification ARGS --enable_fp32=false --with_accuracy_layer=false --int8_model=/PATH/TO/SAVED/QAT/INT8/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
   ```

> Notes: Due to a large amount of images in the `int8_full_val.bin` dataset (50 000), the accuracy benchmark which includes comparison of unoptimized and optimized QAT model may last long (even several hours). To accelerate accuracy measuring, it is recommended to set `OMP_NUM_THREADS` to the max number of physical cores available on the server.
