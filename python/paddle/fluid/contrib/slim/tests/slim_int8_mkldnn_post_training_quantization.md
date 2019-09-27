# PaddleSlim Post-training quantization (MKL-DNN INT8)

This document describes how to use [PaddleSlim](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md) to convert a FP32 ProgramDesc with FP32 weights to an INT8 ProgramDesc with FP32 weights on GoogleNet, MobileNet-V1, MobileNet-V2, ResNet-101, ResNet-50, VGG16 and VGG19. We provide the instructions on how to enable MKL-DNN INT8 calibration in PaddleSlim and show the results of accuracy on all the 7 models as mentioned.

## 0. Prerequisite

You need to install at least PaddlePaddle-1.5 python package `pip install paddlepaddle==1.5`.

## 1. How to generate INT8 ProgramDesc with FP32 weights

You can refer to the usage doc of [PaddleSlim](https://github.com/PaddlePaddle/models/blob/develop/PaddleSlim/docs/usage.md) in section 1.2 for details that how to use PaddleSlim Compressor. But for PaddleSlim Post-training quantization with MKL-DNN INT8, there are two differences.

* Differences in `paddle.fluid.contrib.slim.Compressor` arguments

Since the only one requirement in PaddleSlim Post-training quantization with MKL-DNN INT8 is the reader of warmup dataset, so you need to set other parameters of `paddle.fluid.contrib.slim.Compressor` to None, [] or ''.

```python
com_pass = Compressor(
    place=None, # not required, set to None
    scope=None, # not required, set to None
    train_program=None, # not required, set to None
    train_reader=None, # not required, set to None
    train_feed_list=[], # not required, set to []
    train_fetch_list=[], # not required, set to []
    eval_program=None, # not required, set to None
    eval_reader=reader, # required, the reader of warmup dataset
    eval_feed_list=[], # not required, set to []
    eval_fetch_list=[], # not required, set to []
    teacher_programs=[], # not required, set to []
    checkpoint_path='', # not required, set to ''
    train_optimizer=None, # not required, set to None
    distiller_optimizer=None # not required, set to None
    )
```

* Differences in yaml config

An example yaml config is listed below, for more details, you can refer to [config_mkldnn_int8.yaml](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/quantization/config_mkldnn_int8.yaml) which is used in unit test.

```yaml
version: 1.0
strategies:
    mkldnn_post_training_strategy:
        class: 'MKLDNNPostTrainingQuantStrategy' # required, class name of MKL-DNN INT8 Post-training quantization strategy
        int8_model_save_path: 'OUTPUT_PATH' # required, int8 ProgramDesc with fp32 weights
        fp32_model_path: 'MODEL_PATH' # required, fp32 ProgramDesc with fp32 weights
        cpu_math_library_num_threads: 1 # required, The number of cpu math library threads
compressor:
    epoch: 0 # not required, set to 0
    checkpoint_path: '' # not required, set to ''
    strategies:
        - mkldnn_post_training_strategy
```

## 2. How to run INT8 ProgramDesc with fp32 weights

You can load INT8 ProgramDesc with fp32 weights by load_inference_model [API](https://github.com/PaddlePaddle/Paddle/blob/8b50ad80ff6934512d3959947ac1e71ea3fb9ea3/python/paddle/fluid/io.py#L991) and run INT8 inference similar as [FP32](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/eval.py "FP32").

```python
[infer_program, feed_dict, fetch_targets] = fluid.io.load_inference_model(model_path, exe)
```

## 3. Result

We provide the results of accuracy measured on Intel(R) Xeon(R) Gold 6271.

>**I. Top-1 Accuracy on Intel(R) Xeon(R) Gold 6271**

>**Dataset: ILSVRC2012 Validation dataset**

| Model        | FP32 Accuracy   | INT8 Accuracy   | Accuracy Diff(FP32-INT8)   |
| :----------: | :-------------: | :------------:  | :--------------:           |
| GoogleNet    |  70.50%         |  69.81%         |   0.69%                    |
| MobileNet-V1 |  70.78%         |  70.42%         |   0.36%                    |
| MobileNet-V2 |  71.90%         |  71.35%         |   0.55%                    |
| ResNet-101   |  77.50%         |  77.42%         |   0.08%                    |
| ResNet-50    |  76.63%         |  76.52%         |   0.11%                    |
| VGG16        |  72.08%         |  72.03%         |   0.05%                    |
| VGG19        |  72.57%         |  72.55%         |   0.02%                    |

Notes:

* MKL-DNN and MKL are required.

## 4. How to reproduce the results

Three steps to reproduce the above-mentioned accuracy results, and we take GoogleNet benchmark as an example:

* ### Prepare dataset

You can run the following commands to download and preprocess the ILSVRC2012 Validation dataset.

```bash
cd /PATH/TO/PADDLE
python ./paddle/fluid/inference/tests/api/full_ILSVRC2012_val_preprocess.py
```

Then the ILSVRC2012 Validation dataset will be preprocessed and saved by default in `~/.cache/paddle/dataset/int8/download/int8_full_val.bin`

* ### Prepare model

You can run the following commands to download GoogleNet model.

```bash
mkdir -p /PATH/TO/DOWNLOAD/MODEL/
cd /PATH/TO/DOWNLOAD/MODEL/
export MODEL_NAME=GoogleNet
wget http://paddle-inference-dist.bj.bcebos.com/int8/${MODEL_NAME}_int8_model.tar.gz
mkdir -p ${MODEL_NAME}
tar -xvf ${MODEL_NAME}_int8_model.tar.gz -C ${MODEL_NAME}
```

To download and verify all the 7 models, you need to set `MODEL_NAME` to one of the following values in command line:

```text
MODEL_NAME=GoogleNet, mobilenetv1, mobilenet_v2, Res101, resnet50, VGG16, VGG19
```

* ### Commands to reproduce benchmark

You can run `test_mkldnn_int8_quantization_strategy.py` with the following arguments to reproduce the accuracy result on GoogleNet.

``` bash
cd /PATH/TO/PADDLE/python/paddle/fluid/contrib/slim/tests/
python ./test_mkldnn_int8_quantization_strategy.py --infer_model /PATH/TO/DOWNLOAD/MODEL/${MODEL_NAME}/model --infer_data ~/.cache/paddle/dataset/int8/download/int8_full_val.bin --warmup_batch_size 100 --batch_size 1
```

Notes:

* The above commands will cost maybe several hours in the prediction stage (include int8 prediction and fp32 prediction) since there have 50000 pictures need to be predicted in `int8_full_val.bin`
* Running the above command with environment variable `FLAGS_use_mkldnn=true` will make the FP32 part of the test running using MKL-DNN (the INT8 part uses MKL-DNN either way).
