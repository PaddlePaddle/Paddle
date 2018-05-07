# Float16 Inference in PaddlePaddle Fluid
Author: Kexin Zhao

## Introduction
Working with deep neural networks (DNN) is a two-stage process. First we train DNN using labeled examples of inputs and desired outputs to obtain the model parameters (weights), then we deploy DNN along with the trained weights to run inference on unknown inputs. Typically, these weights are in float data type and hence we run inference in float mode using these weights. This post focuses on the discussion of how to use low precision float16 data type to represent these trained weights and run inference in float16 mode as well as the advantages of float16 inference over its float counterpart by showing some experiment results. 

## What is float16?
float16 (or FP16) is a half-precision floating-point format that uses 16 bits in memory to represent a value. The advantage over 32-bit single-precision floating-point format (commonly known as float data type) is that it requires half the storage and bandwidth at the expense of precision and range. Fortunately, DNN inference has high tolerance against the loss of precision and range when using float16 to represent the weights and the inference accuracy will only be minimally affected in most cases. This gives us the opportunity to use float16 data type to speedup the inference.

Interested readers can refer to our [design doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/data_type/float16.md) and [code](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/platform/float16.h) for more details on how we implement the float16 data type.

## Why float16?
The trend in today's deep learning community is to use bigger and deeper model. This translates to larger memory footprint, higher computation demands, and as a result higher energy consumption on computing devices. The advantages of float16 over float are correspondingly three-fold:

1. We only need half the memory size to load the same model using float16 representations. Moreover, most of the intermediate results generated during float16 inference are also of float16 data type. This makes the whole memory footprint of float16 inference roughly about half of its float counterpart. This is especially useful when deploying inference on mobile devices with limited available memory. Also given the same available memory, the maximum batch size for float16 inference is about twice that for float inference.

2. Because float16 occupies less memory than float, in theory hardware devices can achieve much higher floating point operators per second (FLOPS) for float16 data than float data. Right now, an outstanding example of hardware devices that actually deliver such advantages is Nvidia's latest Volta architecture GPUs, including Tesla V100 and Titan V. Moreover float16 takes less time to read from or write to memory and hence float16 can make inference more efficient especially in memory-bound applications where the performance is largely affected by how fast it is to read and write data.

3. From the energy efficiency perspective, the energy needed to read, write, and compute float16 data is much less that its float counterpart, which can significantly reduce the battery power consumption on mobile devices or the total cost of ownership (TCO) of data centers.

## Fluid implementation of float16 inference
### Overview
Fluid use [Program](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#program) instead of computation graph to describe a neural network model and the optimization procedure. Fluid program is a python wrapper around a protobuf message called [ProgramDesc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/program.md). Similar to programming languages, the basic structure of a Fluid program is some nested [blocks](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#block), where each block consists of some [variable](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#variable) definitions and a sequence of [operators](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/modules/python_api.md#operator). An [executor](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/executor.md) will run a given program by sequentially executing the operators in the entrance block. 

### Basic requirement
When an operator is run by an executor, it uses a kernel to perform computations on tensors contained in the input variables, and then write the results to the tensors in the output variables. Each operator has multiple kernels for different combinations of data types, devices, and library types, respectively. The operator will select the appropriate kernel to run based on, among other things, the data type of the input tensors. By default, every Fluid operator has a kernel for float data type that takes float inputs and generates float outputs.

This means that if we provide float input to the first operator in a program, then each operator will use float kernel to compute float output and send it as input to the next operator to trigger its float kernel. This chain effect will makes the program run in float mode and gives us a final output of float data type. 

The same principle applies if we want a program to run in float16 mode. We provide input variable of float16 data type to the first operator and every subsequent operator will invoke the float16 kernel until we get the final output in float16 data type. So the preliminary requirements for float16 inference is to add float16 kernels to operators that are needed in a specific kind of neural networks. Our current focus is on Convolutional Neural Networks (CNN) and hence we have added float16 kernels to the following operators: convolution, pooling, GEMM, elementwise addition, batch norm, dropout, various activations including relu and tanh, and softmax.

### float16 transpiler
Furthermore, we need a float16 transpiler to achieve the following usage code:

```python
# Get the float32 inference program and load the associated float32 weights
[inference_program, feed_target_names,
 fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

# Prepare the float input data
batch_size = 1
tensor_img = numpy.random.rand(batch_size, 3, 32, 32).astype(numpy.float32)

# Running inference_program in float mode
float_results = exe.run(inference_program,
                        feed={feed_target_names[0]: tensor_img},
                        fetch_list=fetch_targets)

# Use float16 transpiler to speedup
float16_inference_program = float_inference_program.clone()
t = Float16Transpiler()
t.transpile(float16_inference_program, GPUPlace)

# Running float16_inference_program in float16 mode using the same input data
float16_results = exe.run(float16_inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)

# Do some tests to verify the correctness of float16 inference
...
np.testing.assert_almost_equal(float_results, float16_results, ...)
...

# Save the float16 inference program and float16 weights for future deployment
fluid.io.save_inference_model(fp16_save_dirname, feed_target_names,
                              fetch_targets, exe,
                              float16_inference_program)
```

In this scenario, we already have a float32 inference program and some associated float32 weights that can do float32 inference. We can easily use the `transpile` method of the `Float16Transpiler` class to do certain modifications to the existing program and weights so that we have a new float16 program and the associated float16 weights.

We can then run various inference experiments in float16 mode and save the float16 program and weights on disk for future deployment. To enhance the code usability, we maintain a consistent API so that user can use the same float32 input data to run inference program in either float32 and float16 mode and obtain output data both of float32 data type. This requires us to add some cast operators in the program to convert between float16 tensor and float32 tensor.

The float16 transpiler is implemented to fulfill the requirements mentioned above. The details of the float16 transpiler can be found [here](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/data_type/float16.md#float16-inference).

### Experiment results
We provide demo codes that can be used to reproduce the experiment results by doing:
```bash
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
# This line will generate a paddle development docker image with cuda 8 and cudnn 7
# If you want test on cuda 9 instead, change the line 5 in Paddle/Dockerfile 
# from `FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04`
# to `FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04` and similarly for other configurations
nvidia-docker build -t paddle:float16 .
# After running this, different results will be written to different log files in Paddle/contrib/float16/
nvidia-docker run -it -v $PWD:/paddle paddle:float16 /paddle/contrib/float16/run_float16_demo.sh
```

#### Correctness
As is mentioned before, DNN inference has been found to be tolerant against the loss of precision and range incured by float16 and we want to see how good this tolerance is.

We train a resnet32 model using cifar10 data set, save it when test set accuracy is above 60%, and then test the inference accuracy on the 10000 examples of the cifar10 test set in float16 and float32 mode, respectively.

We repeat the test ten times and get the following results:

|        | float16 | float32  |
|--------|--------:|--------: |
| # 1    | 62.75%  | 62.72%   |
| # 2    | 61.27%  | 61.28%   |
| # 3    | 62.24%  | 62.23%   |
| # 4    | 64.16%  | 64.17%   |
| # 5    | 60.75%  | 60.77%   |
| # 6    | 63.25%  | 63.24%   |
| # 7    | 62.15%  | 62.13%   |
| # 8    | 62.05%  | 62.02%   |
| # 9    | 65.19%  | 65.20%   |
| #10    | 62.53%  | 62.48%   |
| average| 62.63%  | 62.62%   |

We can see that the accuracy of float16 inference is very close to that of float32 inference in every experiment (within 0.05% difference) and is overall 0.01% better than its float32 counterpart averaged over 10 tests. 

#### Performance benchmark
Currently, Fluid inference in float16 mode is only supported on Nvidia GPU device. There is no motivation to support float16 inference on non-ARM CPUs because float16 is not natively supported there and float16 calculation will only be slower than its float counterpart. 

Nvidia started to support its native float16 data type (which has the same internal memory representation as Fluid float16 class) on CUDA 7.5. Moreover, float16 speedups on common computational intensive tasks including GEMM (general matrix-matrix multiplication) and convolution are supported since cublas 7.5 and cuDNN 5.0.

Recently, the introduction of [tensor core](https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/) in volta architecture GPUs and the support of tensor core calculation in CUDA 9.0 and cuDNN 7 make float16 truly superior to float in certain deep learning applications.

We thus benchmark the float16 inference performance on a single Nvidia Tesla V100 GPU (volta architecture and with tensor cores) and compare it with its float32 counterpart. All the following results are in ms (millisecond) averaged over 1000 mini-batches with respective to different mini-batch(mb) sizes.

Average inference time for one mini-batch on Vgg16 model tested on imagenet data set:

| total | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 14.01 | 9.70  | 22.99 | 28.26 | 53.87  | 84.42 | 178.95 | 
|float16|  3.32 | 4.11  |  5.88 |  9.41 | 16.54  | 30.47 |  60.23 |
|Speedup|  4.22 | 2.36  |  3.91 |  3.00 |  3.26  |  2.77 |   2.97 |

We can see that float16 inference provides 2x ~ 4x speedup on different batch sizes. 

Convolution operation is ususally the computational bottleneck of CNN, so we also check the average time spent on the Fluid convolution operators for one mini-batch as follows:

|conv op| mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | 
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 11.95 | 6.96  | 18.65 | 21.42 | 41.35  | 60.58 | 130.11 |
|float16|  1.78 | 2.10  |  2.93 |  4.55 |  7.99  | 14.63 |  28.67 |
|Speedup|  6.71 | 3.31  |  6.37 |  4.71 |  5.18  |  4.14 |   4.54 |

Fluid convolution operator uses cuDNN 7 to implement the kernel and we can see that with the help of tensor core, float16 convolution is significantly faster than its float32 counterpart, which makes the overall float16 inference performance much better.

Similarly, we also list the benchmark results of Resnet50 model tested on imagenet data set:

| total | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | mb=128 |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|-------:|
|float32| 7.03  | 7.41  | 9.16  | 12.55 | 21.13  | 38.27 | 67.93  | 127.02 | 
|float16| 6.13  | 6.32  | 6.24  |  7.40 | 10.90  | 18.18 | 33.20  |  64.52 |
|Speedup| 1.15  | 1.17  | 1.47  |  1.70 |  1.94  |  2.11 |  2.05  |   1.97 |

|conv op| mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | mb=128 |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|-------:|
|float32| 5.43  | 5.46  | 6.50  | 8.36  | 13.80  | 24.45 | 41.21  | 73.44  |
|float16| 4.19  | 4.30  | 3.96  | 4.21  |  5.63  |  8.77 | 15.24  | 28.40  |
|Speedup| 1.30  | 1.27  | 1.64  | 1.99  |  2.45  |  2.79 |  2.70  |  2.59  |

We find that the speedup provided by float16 inference starts relatively small at 1.15x for batch size 1 and gradually increase to about 2x for larger batch sizes. Similar trend can be found for the time spent on the convolution operator. Note that right now the tensor core will only be utilized in the convolution operation when certain dimentional requirements are met for the input data and filter. The speedup by float16 inference for Resnet50 is smaller than the Vgg16 counterpart partially because the convolution operation in Resnet is much simpler than the Vgg counterpart and this makes the tensor core less utilized in Resnet than in Vgg.

We also did the same benchmark on a Nvidia GeForce GTX 1080 Ti GPU that does not support tensor core. The results show that for Vgg16, float16 inference provides consistent small speedup (around 1.15x) for all mini-batch sizes, while for Resnet50, float16 inference is slower than its float32 counterpart in small batch sizes (mb = 1 and 2) and then deliver around 1.15x speedup for all larger batch sizes. By comparing the benchmarks on 1080 Ti and V100, we find that tensor core, which is specialized for float16 computations, is a critical component for high performance float16 inference.

Please refer to [here](https://github.com/PaddlePaddle/Paddle/blob/develop/contrib/float16/float16_benchmark.md) for comprehensive benchmark results.

### Summary
1. Fluid is now able to run inference in float16 mode via a float16 transpiler. We currently support CNN programs, including Vgg and Resnet, to run in float16 inference mode.
2. The accuracy of float16 inference is verified to be almost identical to the float32 counterpart at least on CNNs.
3. float16 inference provides significant speedup on large and computationally intensive Vgg16 network on image net data set. For the much smaller and simpler Resnet50, the speedup provided by float16 inference is less significant than on Vgg16 but still favorable especially for large batch size.
4. We cannot achieve the superior float16 inference performance without the help of the newly introduced tensor cores on the Nvidia Volta architecture GPUs.
