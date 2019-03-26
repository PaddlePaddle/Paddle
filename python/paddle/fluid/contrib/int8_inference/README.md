# Offline INT8 Calibration Tool

PaddlePaddle supports offline INT8 calibration to accelerate the inference speed. In this document, we provide the instructions on how to enable INT8 calibration and show the ResNet-50 and MobileNet-V1 results in accuracy.

## 0. Prerequisite
You need to install at least PaddlePaddle-1.3 python package `pip install paddlepaddle==1.3`.

## 1. How to generate INT8 model
You can refer to the unit test in [test_calibration.py](../tests/test_calibration.py). Basically, there are three steps:
* Construct calibration object.

```python
calibrator = int8_utility.Calibrator( # Step 1
    program=infer_program, # required, FP32 program
    pretrained_model=model_path, # required, FP32 pretrained model
    algo=algo, # required, calibration algorithm; default is max, the alternative is KL (Kullback–Leibler divergence)
    exe=exe, # required, executor
    output=int8_model, # required, INT8 model
    feed_var_names=feed_dict, # required, feed dict
    fetch_list=fetch_targets) # required, fetch targets
```

* Call the calibrator.sample_data() after executor run.
```python
_, acc1, _ = exe.run(
    program,
    feed={feed_dict[0]: image,
          feed_dict[1]: label},
    fetch_list=fetch_targets)

calibrator.sample_data() # Step 2
```

* Call the calibrator.save_int8_model() after sampling over specified iterations (e.g., iterations = 50)
```python
calibrator.save_int8_model() # Step 3
```

## 2. How to run INT8 model
You can load INT8 model by load_inference_model [API](https://github.com/PaddlePaddle/Paddle/blob/8b50ad80ff6934512d3959947ac1e71ea3fb9ea3/python/paddle/fluid/io.py#L991) and run INT8 inference similar as [FP32](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/object_detection/eval.py "FP32").

```python
[infer_program, feed_dict,
    fetch_targets] = fluid.io.load_inference_model(model_path, exe)
```

## 3. Result
We provide the results of accuracy measurd on [Intel® Xeon® Platinum Gold Processor](https://ark.intel.com/products/120489/Intel-Xeon-Gold-6148-Processor-27-5M-Cache-2-40-GHz- "Intel® Xeon® Gold 6148 Processor") (also known as Intel® Xeon® Skylake6148).

| Model  | Dataset  | FP32 Accuracy  | INT8 Accuracy  | Accuracy Diff  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| ResNet-50  | Small  | 72.00%  | 72.00%  |  0.00% |
| MobileNet-V1  | Small  | 62.00%  | 62.00%  | 0.00%  |
| ResNet-50  | Full ImageNet Val  |  76.63%  | 76.17%  | 0.46% |
| MobileNet-V1 | Full ImageNet Val  | 70.78%  | 70.49%  | 0.29%  |

Please note that [Small](http://paddle-inference-dist.bj.bcebos.com/int8/calibration_test_data.tar.gz "Small") is a subset of [full ImageNet validation dataset](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar "full ImageNet validation dataset"). 

Notes:
* The accuracy measurement requires the model with `label`.
* The INT8 theoretical speedup is ~1.33X on Intel® Xeon® Skylake Server (please refer to `This allows for 4x more input at the cost of 3x more instructions or 33.33% more compute` in  [Reference](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training "Reference")).

## 4. How to reproduce the results
* Small dataset
```bash
FLAGS_use_mkldnn=true python python/paddle/fluid/contrib/tests/test_calibration.py
```

* Full dataset
```bash
FLAGS_use_mkldnn=true DATASET=full python python/paddle/fluid/contrib/tests/test_calibration.py
```
