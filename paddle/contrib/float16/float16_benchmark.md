# float16 benchmark

## Description
We want to compare the inference benchmark of float16 vs float32 on the "image_classification" example on Nvidia Tesla V100 GPU, where we can enable the tensor core computation for float16 mode. We test Vgg16 and Resnet50 on the imagenet data set, and Vgg16 and Resnet32 on the cifar10 data set. For completeness, we also add the inference benchmark of Vgg16 and Resnet50 on imagenet data set tested on Nvidia GeForce GTX 1080 Ti GPU.

For more details about tensor core, please refer to https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/

## Test environment
- GPU: single Nvidia Tesla V100 or single Nvidia GeForce GTX 1080 Ti 
- CUDNN: 7.1.1
- CUDA: 9.0
- Code: https://github.com/PaddlePaddle/Paddle/pull/10331 (Tensor core is enabled in float16 mode)

## Benchmark on V100
All times are in ms (millisecond) averaged over 1000 iterations tested on a single Nvidia V100 GPU with respective to different mini-batch(mb) sizes.

### Vgg16 on imagenet (flowers data set: image.shape = [3, 224, 224]):

Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 14.01 | 9.70  | 22.99 | 28.26 | 53.87  | 84.42 | 178.95 | 
|float16|  3.32 | 4.11  |  5.88 |  9.41 | 16.54  | 30.47 |  60.23 |
|Speedup|  4.22 | 2.36  |  3.91 |  3.00 |  3.26  |  2.77 |   2.97 |

Total time spent on conv op for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | 
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 11.95 | 6.96  | 18.65 | 21.42 | 41.35  | 60.58 | 130.11 |
|float16|  1.78 | 2.10  |  2.93 |  4.55 |  7.99  | 14.63 |  28.67 |
|Speedup|  6.71 | 3.31  |  6.37 |  4.71 |  5.18  |  4.14 |   4.54 |


### Resnet50 on imagenet (flowers data set: image.shape = [3, 224, 224]):

Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | mb=128 |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|-------:|
|float32| 7.03  | 7.41  | 9.16  | 12.55 | 21.13  | 38.27 | 67.93  | 127.02 | 
|float16| 6.13  | 6.32  | 6.24  |  7.40 | 10.90  | 18.18 | 33.20  |  64.52 |
|Speedup| 1.15  | 1.17  | 1.47  |  1.70 |  1.94  |  2.11 |  2.05  |   1.97 |

Total time spent on conv op for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | mb=128 |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|-------:|
|float32| 5.43  | 5.46  | 6.50  | 8.36  | 13.80  | 24.45 | 41.21  | 73.44  |
|float16| 4.19  | 4.30  | 3.96  | 4.21  |  5.63  |  8.77 | 15.24  | 28.40  |
|Speedup| 1.30  | 1.27  | 1.64  | 1.99  |  2.45  |  2.79 |  2.70  |  2.59  |


### Vgg16 on cifar10 (image.shape = [3, 32, 32]):

Total inference time for one batch:

|       | mb=1 | mb=2 | mb=4 | mb=8 | mb=16 | mb=32 | mb=64 | mb=128 | mb=256 | mb=512 |
|-------|-----:|-----:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:| 
|float32| 3.13 | 3.17 | 3.19 | 3.58 | 3.98  | 6.23  | 8.42  | 13.44  | 24.19  | 44.97  | 
|float16| 2.72 | 2.77 | 2.76 | 2,88 | 2.96  | 3.24  | 4.01  |  5.78  |  9.65  | 17.37  |
|Speedup| 1.15 | 1.14 | 1.16 | 1.24 | 1.34  | 1.92  | 2.10  |  2.33  |  2.51  |  2.59  |


### Resnet32 on cifar10 (image.shape = [3, 32, 32]):

Total inference time for one batch:

|       | mb=1 | mb=2 | mb=4 | mb=8 | mb=16 | mb=32 | mb=64 | mb=128 | mb=256 | mb=512 |
|-------|-----:|-----:|-----:|-----:|------:|------:|------:|-------:|-------:|-------:|
|float32| 3.11 | 3.14 | 2.99 | 3.04 | 3.10  | 3.28  | 4.47  | 6.86   | 11.63  | 21.16  |
|float16| 3.70 | 3.81 | 3.75 | 3.83 | 3.77  | 3.97  | 3.92  | 4.15   |  6.41  | 11.02  | 
|Speedup|      |      |      |      |       |       | 1.14  | 1.65   |  1.81  |  1.92  |


## Benchmark on 1080 Ti
All times are in ms (millisecond) averaged over 1000 iterations tested on a single Nvidia GeForce GTX 1080 Ti GPU with respective to different mini-batch(mb) sizes.

### Vgg16 on imagenet (flowers data set: image.shape = [3, 224, 224]):
Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32  |
|-------|-----: |-----: |-----: |-----: |------: |-------:|
|float32| 5.60  | 9.38  | 15.86 | 29.79 | 57.60  | 117.73 |
|float16| 4.99  | 7.79  | 13.47 | 26.02 | 52.30  | 102.34 |
|Speedup| 1.12  | 1.20  |  1.18 |  1.15 |  1.10  |   1.15 |


### Resnet50 on imagenet (flowers data set: image.shape = [3, 224, 224]):
Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32  | mb=64  |
|-------|-----: |-----: |-----: |-----: |------: |-------:|-------:|
|float32| 5.63  | 6.23  | 8.85  | 14.71 | 26.07  | 52.86  | 108.95 |
|float16| 5.89  | 6.44  | 7.94  | 12.57 | 22.03  | 45.06  |  92.68 |
|Speedup|       |       | 1.12  |  1.17 |  1.18  |  1.17  |   1.18 |
