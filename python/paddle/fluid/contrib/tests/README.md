PaddlePaddle supports offline INT8 calibration to accelerate the inferene speed. In this document, we provide the instructions on how to enable INT8 calibration and show the results in both accuracy and performance.

## 0. Prerequisite
You need to build PaddlePaddle python package (e.g., paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64.whl) and install the package successfully.

## 1. How to generate INT8 model
You can refer to the unit test in test_calibration.py. Basically, there are three steps:
* Construct calibration object. The required parameters are: 1) **program**: FP32 program; 2) **pretrained_model**: FP32 pretrained model; 3) **algo**: calibration algorithm; default is np.max, the alternative is KL (Kullback–Leibler divergence); 4) **exe**: executor; 5) **output**: INT8 model; 6) **feed_var_names**: feed dict; and 7) **fetch_list**: fetch targets. Here is the example:

            calibrator = int8_utility.Calibrator( # Step 1
                program=infer_program,
                pretrained_model=model_path,
                algo=algo,
                exe=exe,
                output=int8_model,
                feed_var_names=feed_dict,
                fetch_list=fetch_targets)

* Sample data. Call the calibrator.sample_data() after executor run.
            _, acc1, _ = exe.run(
                program,
                feed={feed_dict[0]: image,
                      feed_dict[1]: label},
                fetch_list=fetch_targets)

                calibrator.sample_data() # Step 2

* Save INT8 model. Call the calibrator.save_int8_model() after sampling over specified iterations (e.g., iterations = 50)
            calibrator.save_int8_model() # Step 3
			

Please note that we assume FP32 pre-trained model has the `__model__ ` file with image and label feed variables which will be used by the accuracy measurememnt.

## 2. How to run INT8 model
You can load INT8 model by load_inference_model [API](https://github.com/PaddlePaddle/Paddle/blob/8b50ad80ff6934512d3959947ac1e71ea3fb9ea3/python/paddle/fluid/io.py#L991) and run INT8 inference similar as [FP32](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleCV/object_detection/eval.py "FP32").

        [infer_program, feed_dict,
         fetch_targets] = fluid.io.load_inference_model(model_path, exe)
		 

## 3. Result
We provide the results of accuracy and performance, measurd on [Intel® Xeon® Platinum 8180 Processor](https://ark.intel.com/products/120496/Intel-Xeon-Platinum-8180-Processor-38-5M-Cache-2-50-GHz- "Intel® Xeon® Platinum 8180 Processor") (also known as Intel® Xeon® Skylake8180).

| Model  | Dataset  | FP32 Accuracy  | INT8 Accuracy  | Accuracy Diff  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| ResNet50  | Small  | 72.00%  | 72.00%  |  0.00% |
| MobileNet-V1  | Small  | 62.00%  | 62.00%  | 0.00%  |
| ResNet-50  | Full ImageNet Val  |  76.63%  | 76.14%  | 0.48% |
| MobileNet-V1 | Full ImageNet Val  | 70.78%  | 70.41%  | 0.37%  |

Please note that [Small](http://paddle-inference-dist.cdn.bcebos.com/int8/calibration_test_data.tar.gz "Small") is a subset of [full ImageNet validation dataset](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar "full ImageNet validation dataset"). Here is the typical dataset structure, similar to the requirement of [PaddlePaddle models](https://github.com/PaddlePaddle/models/tree/develop/fluid/PaddleCV/image_classification/data "PaddlePaddle models"):

        $ ls data/ILSVRC2012/
		val  val_list.txt

| Model  | Batch Size  | FP32 Throughput  | INT8 Throughput  | Improvement  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| ResNet50  | 512  | 326  |   542 |  1.66X |
| MobileNet-V1  | 512  | 1133 | 2246   | 1.98X  |
| ResNet-50  | 1  |   65  | 101  | 1.55X |
| MobileNet-V1 | 1  | 165  | 217  | 1.32X  |

Please note that the performance improvement is ~1.33X on Intel® Xeon® Skylake Server ([Reference](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training "Reference")).
