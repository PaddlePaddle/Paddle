# SLIM Quantization-aware training (QAT) on INT8 MKL-DNN

This document describes how to use [Paddle Slim](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/advanced_usage/paddle_slim/paddle_slim.md) to convert a quantization-aware trained model to an INT8 MKL-DNN runnable model which has almost the same accuracy as QAT on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. We provide the accuracy results compared with fake QAT accuracy by running the QAT trained model with MKL-DNN int8 kernel on above 7 models.

## 0. Prerequisite
You need to install at least PaddlePaddle-1.5 python package `pip install paddlepaddle==1.5`.

## 1. How to generate INT8 MKL-DNN QAT model
You can refer to the unit test in [test_quantization_mkldnn_pass.py](test_quantization_mkldnn_pass.py). Users firstly use PaddleSlim quantization strategy to get a saved fake QAT model by [QuantizationFreezePass](https://github.com/PaddlePaddle/models/tree/develop/PaddleSlim/quant_low_level_api), then use the `TransformForMkldnnPass` to get the graph which can be run with MKL-DNN INT8 kernel. In Paddle Release 1.5, this pass only support `conv2d` and `depthwise_conv2d`.

```python
    import paddle.fluid as fluid
    from paddle.fluid.contrib.slim.quantization \
        import TransformForMkldnnPass
    from paddle.fluid.framework import IrGraph
    from paddle.fluid import core	

    graph = IrGraph(core.Graph(fluid.Program().desc), for_test=False)
    place = fluid.CPUPlace()
    mkldnn_pass = TransformForMkldnnPass(fluid.global_scope(),
    place)
    mkldnn_pass.apply(graph)
```

## 2. Accuracy benchmark

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 8280**

| Model        | Dataset         | QAT Top1 Accuracy | MKL-DNN INT8 Top1 Accuracy | Accuracy Diff   |
| :----------: | :-------------: | :---------------: | :-----------------------:  | :--------------:|
| GoogleNet    | ILSVRC2012 Val  |    70.40%         |            70.39%          |     0.010%      |
| MobileNet-V1 | ILSVRC2012 Val  |    70.83%         |            70.84%          |    -0.010%      |
| MobileNet-V2 | ILSVRC2012 Val  |    72.09%         |            72.09%          |     0.000%      |
| ResNet-101   | ILSVRC2012 Val  |    77.47%         |            77.47%          |     0.000%      |
| ResNet-50    | ILSVRC2012 Val  |    76.66%         |            76.62%          |     0.040%      |
| VGG16        | ILSVRC2012 Val  |    72.71%         |            72.69%          |     0.020%      |
| VGG19        | ILSVRC2012 Val  |    73.37%         |            73.37%          |     0.000%      |

Notes:

* MKL-DNN and MKL are required.

## 3. Commands
* #### Full dataset (Single core)
 * ##### Download full ImageNet Validation Dataset
```bash
cd /PATH/TO/PADDLE
python paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```
The converted data binary file is saved by default in ~/.cache/paddle/dataset/int8/download/int8_full_val.bin
 * ##### ResNet50 Full dataset benchmark
```bash
OMP_NUM_THREADS=28 FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/slim/tests/qat_int8_comparison.py --qat_model=build/third_party/inference_demo/int8v2/ResNet50_QAT/model --infer_data=/path/to/converted/int8_full_val.bin --batch_size=50 --batch_num=1000 --acc_diff_threshold=0.001
```
Set argument `--qat_model` with the following values in command line to benchmark 6 other models:

| Model Name   | --infer_model  |
| :----------: | :------------: |
| MobileNet-V1 | build/third_party/inference_demo/int8v2/MobileNetV1_QAT/model |
| MobileNet-V2 | build/third_party/inference_demo/int8v2/MobileNetV2_QAT/model |
| ResNet-101   | build/third_party/inference_demo/int8v2/ResNet101_QAT/model   |
| VGG16        | build/third_party/inference_demo/int8v2/VGG16_QAT/model       |
| VGG19        | build/third_party/inference_demo/int8v2/VGG19_QAT/model       |
| GoogleNet    | build/third_party/inference_demo/int8v2/GoogleNet_QAT/model   |
