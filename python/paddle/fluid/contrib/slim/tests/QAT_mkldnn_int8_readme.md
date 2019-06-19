# SLIM Quantization-aware training (QAT) on INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a quantization-aware trained model to an INT8 MKL-DNN runnable model which has almost the same accuracy as QAT on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. We provide the accuracy results compared with fake QAT accuracy by running the QAT trained model with MKL-DNN int8 kernel on above 7 models.

## 0. Prerequisite
You need to install at least PaddlePaddle-1.5 python package `pip install paddlepaddle==1.5`.

## 1. How to generate INT8 MKL-DNN QAT model
You can refer to the unit test in [test_quantization_mkldnn_pass.py](test_quantization_mkldnn_pass.py). Users firstly use PaddleSlim quantization strategy to get a saved fake QAT model by [QuantizationFreezePass](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api), then use the `TransformForMkldnnPass` to get the graph which can be run with MKL-DNN INT8 kernel. In Paddle Release 1.5, this pass only supports `conv2d` and `depthwise_conv2d` with channel-wise quantization for weights.

```python
    import paddle.fluid as fluid
    from paddle.fluid.contrib.slim.quantization import TransformForMkldnnPass
    from paddle.fluid.framework import IrGraph
    from paddle.fluid import core	
    
    # Create the IrGraph by Program
    graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
    place = fluid.CPUPlace()
    # Convert the IrGraph to MKL-DNN supported INT8 IrGraph by using
    # TransformForMkldnnPass
    mkldnn_pass = TransformForMkldnnPass(fluid.global_scope(), place)
    # Apply TransformForMkldnnPass to IrGraph
    mkldnn_pass.apply(graph)
```

## 2. Accuracy benchmark

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

| Model        | Fake QAT Top1 Accuracy | Fake QAT Top5 Accuracy |MKL-DNN INT8 Top1 Accuracy |  Top1 Diff   | MKL-DNN INT8 Top5 Accuracy | Top5 Diff  |
| :----------: | :--------------------: | :--------------------: |:-----------------------:  | :----------: | :------------------------: | :--------: |
| GoogleNet    |         70.40%         |          89.46%        |           70.39%          |     0.010%   |           89.46%           |   0.000%   |
| MobileNet-V1 |         70.83%         |          89.56%        |           70.84%          |    -0.010%   |           89.56%           |   0.000%   |
| MobileNet-V2 |         72.17%         |          90.67%        |           72.13%          |     0.040%   |           90.67%           |   0.000%   |
| ResNet-101   |         77.49%         |          93.65%        |           77.51%          |    -0.020%   |           93.67%           |  -0.020%   |
| ResNet-50    |         76.62%         |          93.08%        |           76.61%          |     0.010%   |           93.09%           |  -0.010%   |
| VGG16        |         72.71%         |          91.11%        |           72.69%          |     0.020%   |           91.09%           |   0.020%   |
| VGG19        |         73.37%         |          91.40%        |           73.37%          |     0.000%   |           91.41%           |  -0.010%   |

Notes:

* MKL-DNN and MKL are required.

## 3. How to reproduce the results
Three steps to reproduce the above-mentioned accuracy results, and we take ResNet50 benchmark as an example:
 * ### Prepare dataset
```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in `~/.cache/paddle/dataset/int8/download/int8_full_val.bin`
 * ### Prepare model
You can run the following commands to download ResNet50 model.

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export MODEL_NAME=ResNet50
wget http://paddle-inference-dist.bj.bcebos.com/int8/QAT_models/${MODEL_NAME}_qat_model.tar.gz
mkdir -p ${MODEL_NAME}
tar -xvf ${MODEL_NAME}_qat_model.tar.gz -C ${MODEL_NAME}
```

To download and verify all the 7 models, you need to set `MODEL_NAME` to one of the following values in command line:

```text
MODEL_NAME=ResNet50, ResNet101, GoogleNet, MobileNetV1, MobileNetV2, VGG16, VGG19
```
* ### Commands to reproduce benchmark
You can run `qat_int8_comparison.py` with the following arguments to reproduce the accuracy result on ResNet50.

```bash
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_comparison.py --qat_model=/PATH/TO/DOWNLOAD/MODEL/${MODEL_NAME}/model --infer_data=~/.cache/paddle/dataset/int8/download/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.001
```
> Notes: The above commands will cost maybe several hours in the prediction stage (include int8 prediction and fp32 prediction) since there have 50000 pictures need to be predicted in `int8_full_val.bin`. User can set `OMP_NUM_THREADS` to the max number of physical cores of the used server to accelerate the process.
