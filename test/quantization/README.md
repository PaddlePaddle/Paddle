# SLIM Quantization-aware training (QAT) for INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://paddlepaddle.github.io/PaddleSlim/index.html) to convert a quantization-aware trained model (Quant model) into INT8 MKL-DNN quantized model and run it.

In **Release 1.5**, we have released the first approach to the MKL-DNN-based quantization of Quant models, called Quant1. It enabled the `conv2d` and `mul` INT8 MKL-DNN kernels for Quant trained models (GoogleNet, MobileNetV1, MobileNetV2, ResNet50, ResNet101, VGG16, and VGG19) with 0.05% accuracy diff.

In **Release 1.6**, a new approach was introduced, called Quant2, which adds support for more performance optimizations and more INT8 MKL-DNN kernels. INT8 MKL-DNN models obtained using Quant2 have much better inference performance than using Quant1, with only a little bit bigger accuracy diff.

In **Release 1.7**, a support for [Ernie (NLP) Quant trained model](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn) was added to the Quant2.

In **Release 2.0**, further optimizations were added to the Quant2: INT8 `matmul` kernel, inplace execution of activation and `elementwise_add` operators, and broader support for quantization aware strategy from PaddleSlim.

In this document we focus on the Quant2 approach only.

## 0. Prerequisites
* PaddlePaddle in version 2.0 or higher is required. For instructions on how to install it see the [installation document](https://www.paddlepaddle.org.cn/install/quick).

* MKL-DNN and MKL are required. The highest performance gain can be observed using CPU servers supporting AVX512 instructions.
* INT8 accuracy is best on CPU servers supporting AVX512 VNNI extension (e.g. CLX class Intel processors). A linux server supports AVX512 VNNI instructions if the output of the command `lscpu` contains the `avx512_vnni` entry in the `Flags` section. AVX512 VNNI support on Windows can be checked using the [`coreinfo`]( https://docs.microsoft.com/en-us/sysinternals/downloads/coreinfo) tool.

## 1. Introduction

There are two approaches to quantization supported in PaddlePaddle: [post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) (PTQ) and quantization-aware training (QAT). Using both PTQ and QAT a user can convert models created by PaddleSlim into INT8 models and run INT8 inference on CPU. PTQ is more automatic and requires less model preparation. However, QAT usually gives better accuracy with similar performance. In this document we focus on a transformation from intermediate models obtained during the QAT process (Quant models) into MKL-DNN INT8 models. We call this procedure Quant2.

## 2. How to turn an FP32 model into a Quant model?

A procedure on how to transform an FP32 model into a Quant model supported by the Quant2 approach is described in [this document](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/demo/mkldnn_quant/README.md).

## 3. How to turn a Quant model into an INT8 MKL-DNN model?

A Quant model can be transformed into an INT8 quantized model if it contains enough information about quantization scales for every quantized operator in the graph. The process of quantization is done by the `Quant2Int8MkldnnPass` pass which comprises several steps:

### Gathering scales

The information about the quantization scales is collected from two sources:

1. the `out_threshold` attribute of quantizable operators - it contains a single value quantization scale for the operator's output,
2.  fake quantize/dequantize operators - they imitate quantization from FP32 into INT8, or dequantization in reverse direction, but keep the quantized tensor values as floats.

There are three types of fake quantize/dequantize operators:

* `fake_quantize_moving_average_abs_max` and `fake_quantize_range_abs_max` - used before quantized operator (e.g. `conv2d`), gather single value scale information for the op's input,
* `fake_dequantize_max_abs` and `fake_channel_wise_dequantize_max_abs` - used after quantized operators, contain scales used for the operators' weights dequantization; the first one collects a single value scale for the weights tensor, whereas the second one collects a vector of scales for each output channel of the weights,
* `fake_quantize_dequantize_moving_average_abs_max` - used after a quantized operator to get the scale value for the op's output; imitates immediate quantization and dequantization.

Scale values gathered from the fake quantize/dequantize operators have precedence over the scales collected from the `out_threshold` attributes.

Notes:

1. As the next steps describe, quantization will be applied later to an optimized FP32 model. It means that quantization scales for inputs and outputs of each quantized operator have to be gathered for tensors which are inputs and outputs of already optimized or fused operators. For example, if a model contains the following sequence of tensors and operators in the graph
   ```... → input1 → conv2d → output1 → batch_norm → output2 → relu → output3 → ...```
   and we want to quantize the `conv2d` op, then after applying FP32 optimizations the sequence will become
   ```... → input1 → conv2d → output3 → ...```
   and the quantization scales have to be collected for the `input1` and `output3` tensors in the Quant model.
2. Quantization of the following operators is supported: `conv2d`, `depthwise_conv2d`, `mul`, `fc`, `matmul`, `pool2d`, `reshape2`, `transpose2`, `concat`.
3. The longest sequence of consecutive quantizable operators in the model, the biggest performance boost can be achieved through quantization:
   ```... → conv2d → conv2d → pool2d → conv2d → conv2d → ...```
   Quantizing single operator separated from other quantizable operators can give no performance benefits or even slow down the inference:
   ```... → swish → fc → softmax → ...`

### Removing fake operators

All the `fake_quantize_*` and `fake_dequantize_*` operators are being removed from the graph.

### Dequantizing weights

Weights of `conv2d`, `depthwise_conv2d` and `mul` operators are assumed to be fake-quantized (with integer values in the `int8` range, but kept as `float`s) in Quant models. Here, the information about the scale from `fake_dequantize_max_abs` and `fake_channel_wise_dequantize_max_abs` operators is used to fake-dequantize the weights back to the full float range of values. At this moment the model becomes an unoptimized clean FP32 inference model.

### Optimizing FP32 graph

A series of standard optimization passes are being applied to the FP32 graph. This gives us an optimized FP32 inference model and we can proceed with INT8 quantization.

### Computing weight scales

After optimization fuses, the weight tensors of `conv2d` or `fc` operators are likely to have different values and require new quantization scales. The weights are static, i.e. they do not change during the inference process, and the scales can be calculated simply as a maximum of absolute values from the tensor. To improve the inference accuracy we calculate the scales for each output channel separately, getting an array of quantization scales for a weight tensor.

### Taking activations into account

The basic datatype used during INT8 inference is signed INT8, with possible values from -128 to 127. However, if `conv2d` or `fc` operator has `relu` or `relu6` activation integrated in it, the output of the operator is known to have non-negative values. In that case we use unsigned INT8 datatype for output tensors, with a wider range for positive values (0 to 255), improving the inference accuracy further.

### Propagation of scales

Some of the operators (e.g. `reshape2`, `transpose2`, `pool2d` with max pooling) transform the data without changing the quantization scale. For this reason we propagate the quantization scale values through these operators without any modifications. We propagate the quantization scales also through the `scale` operator, updating the quantization scale accordingly. This approach lets us minimize the number of fake quantize/dequantize operators in the graph, because the information about the scales required for the quantization process to succeed spreads between quantized operators.

### Applying quantization passes

Having gathered all the data needed for quantization we apply the `cpu_quantize_pass` which quantizes the graph, and the `cpu_quantize_squash_pass` which optimizes the INT8 graph.

## 4. Code example

The code snipped shows how the `Quant2Int8MkldnnPass` can be applied to a model graph:

```python
    import paddle
    import paddle.static as static
    from paddle.static.quantization import Quant2Int8MkldnnPass
    from paddle.base.framework import IrGraph
    from paddle.framework import core

    # Create the IrGraph by Program
    graph = IrGraph(core.Graph(static.Program().desc), for_test=False)
    place = paddle.CPUPlace()
    # Convert the IrGraph to MKL-DNN supported INT8 IrGraph using the
    # Quant2Int8MkldnnPass. It requires a list of operators to be quantized
    mkldnn_pass = Quant2Int8MkldnnPass({'conv2d', 'pool2d'}, static.global_scope(), place, core, False)
    # Apply Quant2Int8MkldnnPass to IrGraph
    mkldnn_pass.apply(graph)

```

## 5. Accuracy and Performance benchmark

This section contain Quant2 MKL-DNN accuracy and performance benchmark results measured on the following server:

* Intel(R) Xeon(R) Gold 6271 (with AVX512 VNNI support),

Performance benchmarks were run with the following environment settings:

* The benchmark threads were assigned to cores by setting

  ```bash
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_BLOCKTIME=1
  ```

* Turbo Boost was set to OFF using the command

  ```bash
  echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
  ```

### Image classification models benchmark results

#### Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 Top1 Accuracy | INT8 Quant Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 Quant Top5 Accuracy | Top5 Diff |
| :----------: | :----------------: | :--------------------: | :-------: | :----------------: | :--------------------: | :-------: |
| MobileNet-V1 |       70.78%       |         70.71%         |  -0.07%   |       89.69%       |         89.41%         |  -0.28%   |
| MobileNet-V2 |       71.90%       |         72.11%         |  +0.21%   |       90.56%       |         90.62%         |  +0.06%   |
|  ResNet101   |       77.50%       |         77.64%         |  +0.14%   |       93.58%       |         93.58%         |   0.00%   |
|   ResNet50   |       76.63%       |         76.47%         |  -0.16%   |       93.10%       |         92.98%         |  -0.12%   |
|    VGG16     |       72.08%       |         71.73%         |  -0.35%   |       90.63%       |         89.71%         |  -0.92%   |
|    VGG19     |       72.57%       |         72.12%         |  -0.45%   |       90.84%       |         90.15%         |  -0.69%   |

#### Performance

Image classification models performance was measured using a single thread. The setting is included in the benchmark reproduction commands below.


>**Intel(R) Xeon(R) Gold 6271**

|    Model     | FP32 (images/s) | INT8 Quant (images/s) | Ratio (INT8/FP32)  |
| :----------: | :-------------: | :-----------------: | :---------------:  |
| MobileNet-V1 |      74.05      |       196.98        |      2.66          |
| MobileNet-V2 |      88.60      |       187.67        |      2.12          |
|  ResNet101   |      7.20       |       26.43         |      3.67          |
|   ResNet50   |      13.23      |       47.44         |      3.59          |
|    VGG16     |      3.47       |       10.20         |      2.94          |
|    VGG19     |      2.83       |       8.67          |      3.06          |

Notes:

* Performance FP32 (images/s) values come from [INT8 MKL-DNN post-training quantization](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/int8_mkldnn_quantization.md) document.

### NLP models benchmark results

#### Accuracy

>**Intel(R) Xeon(R) Gold 6271**

|     Model    |  FP32 Accuracy | Quant INT8 Accuracy | Accuracy Diff |
|:------------:|:----------------------:|:----------------------:|:---------:|
|   Ernie      |      80.20%            |        79.44%        |  -0.76%  |


#### Performance


>**Intel(R) Xeon(R) Gold 6271**

|  Model  |     Threads  | FP32 Latency (ms) | Quant INT8 Latency (ms)    | Ratio (FP32/INT8) |
|:------------:|:----------------------:|:-------------------:|:---------:|:---------:|
| Ernie | 1 thread     |       237.21        |     79.26    |   2.99x    |
| Ernie | 20 threads   |       22.08         |     12.57    |   1.76x    |


## 6. How to reproduce the results

The steps below show, taking ResNet50 as an example, how to reproduce the above accuracy and performance results for Image Classification models.
To reproduce NLP models results (Ernie), please follow [How to reproduce Ernie Quant results on MKL-DNN](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn/README.md).

### Prepare dataset

Download the dataset for image classification models benchmarking by executing:

```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin`

### Prepare models

Run the following commands to download and extract Quant model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QUANT_MODEL_NAME=ResNet50
export QUANT_MODEL_ARCHIVE=${QUANT_MODEL_NAME}_qat_model.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${QUANT_MODEL_ARCHIVE}
mkdir ${QUANT_MODEL_NAME} && tar -xvf ${QUANT_MODEL_ARCHIVE} -C ${QUANT_MODEL_NAME}
```

To download other Quant models, set the `QUANT_MODEL_NAME` variable in the above commands to one of the values: `ResNet101`, `MobileNetV1`, `MobileNetV2`, `VGG16`, `VGG19`.

Moreover, there are other variations of these Quant models that use different methods to obtain scales during training, run these commands to download and extract Quant model:

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export QUANT_MODEL_NAME=ResNet50_qat_perf
export QUANT_MODEL_ARCHIVE=${QUANT_MODEL_NAME}.tar.gz
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${QUANT_MODEL_ARCHIVE}
mkdir ${QUANT_MODEL_NAME} && tar -xvf ${QUANT_MODEL_ARCHIVE} -C ${QUANT_MODEL_NAME}
```

To download other Quant models, set the `QUANT_MODEL_NAME` variable to on of the values: `ResNet50_qat_perf`, `ResNet50_qat_range`, `ResNet50_qat_channelwise`, `MobileNet_qat_perf`, where:
- `ResNet50_qat_perf`, `MobileNet_qat_perf` with input/output scales in `fake_quantize_moving_average_abs_max` operators, with weight scales in `fake_dequantize_max_abs` operators
- `ResNet50_qat_range`, with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes, with weight scales in `fake_dequantize_max_abs` operators
- `ResNet50_qat_channelwise`, with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes, with weight scales in `fake_channel_wise_dequantize_max_abs` operators

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

You can use the `quant2_int8_image_classification_comparison.py` script to reproduce the accuracy result of the INT8 Quant models. The following options are required:

* `--quant_model` - a path to a Quant model that will be transformed into INT8 model.
* `--fp32_model` - a path to an FP32 model whose accuracy will be measured and compared to the accuracy of the INT8 model.
* `--infer_data` - a path to the validation dataset.

The following options are also accepted:
* `--ops_to_quantize` - a comma-separated list of operator types to quantize. If the option is not used, an attempt to quantize all quantizable operators will be made, and in that case only quantizable operators which have quantization scales provided in the Quant model will be quantized. When deciding which operators to put on the list, the following have to be considered:
  * Only operators which support quantization will be taken into account.
  * All the quantizable operators from the list, which are present in the model, must have quantization scales provided in the model. Otherwise, quantization of the operator will be skipped with a message saying which variable is missing a quantization scale.
  * Sometimes it may be suboptimal to quantize all quantizable operators in the model (cf. *Notes* in the **Gathering scales** section above). To find the optimal configuration for this option, user can run benchmark a few times with different lists of quantized operators present in the model and compare the results. For Image Classification models mentioned above the list usually comprises of `conv2d` and `pool2d` operators.
* `--op_ids_to_skip` - a comma-separated list of operator ids to skip in quantization. To get an id of a particular operator run the script with the `--debug` option first (see below for the description of the option), and having opened the generated file `int8_<some_number>_cpu_quantize_placement_pass.dot` find the id number written in parentheses next to the name of the operator.
* `--debug` - add this option to generate a series of `*.dot` files containing the model graphs after each step of the transformation. For a description of the DOT format see [DOT]( https://graphviz.gitlab.io/_pages/doc/info/lang.html). The files will be saved in the current location. To open the `*.dot` files use any of the Graphviz tools available on your system (e.g. `xdot` tool on Linux or `dot` tool on Windows, for documentation see [Graphviz](http://www.graphviz.org/documentation/)).

```bash
cd /PATH/TO/PADDLE
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/static/quantization/slim/tests/quant2_int8_image_classification_comparison.py --quant_model=/PATH/TO/DOWNLOADED/QUANT/MODEL --fp32_model=/PATH/TO/DOWNLOADED/FP32/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.01 --ops_to_quantize="conv2d,pool2d"
```

> Notes: Due to a large amount of images in the `int8_full_val.bin` dataset (50 000), the accuracy benchmark may last long. To accelerate accuracy measuring, it is recommended to set `OMP_NUM_THREADS` to the maximum number of physical cores available on the server.

#### Performance benchmark commands

To reproduce the performance results, the environment variable `OMP_NUM_THREADS=1` and `--batch_size=1` option should be set.

1. Transform the Quant model into INT8 model by applying the `Quant2Int8MkldnnPass` pass and save the result. You can use the script `save_quant_model.py` for this purpose. It also accepts the option `--ops_to_quantize` with a list of operators to quantize.

   ```bash
   cd /PATH/TO/PADDLE/build
   python ../python/paddle/static/quantization/slim/tests/save_quant_model.py --quant_model_path=/PATH/TO/DOWNLOADED/QUANT/MODEL --int8_model_save_path=/PATH/TO/SAVE/QUANT/INT8/MODEL --ops_to_quantize="conv2d,pool2d"
   ```

2. Run the C-API test for performance benchmark.

   ```bash
   cd /PATH/TO/PADDLE/build
   OMP_NUM_THREADS=1 paddle/fluid/inference/tests/api/test_analyzer_quant_image_classification ARGS --enable_fp32=false --with_accuracy_layer=false --int8_model=/PATH/TO/SAVED/QUANT/INT8/MODEL --infer_data=$HOME/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=1 --paddle_num_threads=1
   ```
