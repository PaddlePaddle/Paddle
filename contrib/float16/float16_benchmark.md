# float16 benchmark

## Description
We want to compare the inference benchmark of float16 vs float32 on the "image_classification" example on V100 GPU, where we can enable the tensor core computation for float16 mode, where we have Vgg16 and Resnet 50 on the imagenet data set, and Vgg16 and Resnet32 on the cifar10 data set. For completeness, we also add the inference benchmark on Titan Xp GPU of Vgg16 and Resnet50 on the imagenet data set.

For more details about tensor core, please refer to https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/

## Test environment
Test environment I:
- GPU: single Nvidia V100
- CUDNN: 7.1.1
- CUDA: 9.0

Test environment II:
- GPU: single Nvidia Titan Xp
- CUDNN: 5
- CUDA: 8.0

Code: https://github.com/PaddlePaddle/Paddle/pull/9488 (Tensor core is enabled for float16 mode)


## Total time on V100
All times are in ms (millisecond) averaged over 1000 iterations tested on a single Nvidia V100 GPU with respective to different mini-batch(mb) sizes.

### Vgg16 on imagenet (flowers data set: image.shape = [3, 224, 224]):

Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 14.64 | 10.24 | 23.54 | 28.41 | 53.62  | 83.84 | Out of Memory | 
|float16| 3.94  | 4.62  | 6.21  | 9.39  | 15.82  | 28.54 | 56.23  |
|Speedup| 3.72  | 2.22  | 3.79  | 3.03  | 3.39   | 2.94  | |

Total time spent on conv op for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 |
|-------|-----: |-----: |-----: |-----: |------: |------:|
|float32| 12.0 | 6.96 | 18.6 | 21.4 | 41.3  | 60.7 |
|float16| 1.81  | 2.11  | 2.95  | 4.57  | 8.0  | 14.6 |
|Speedup| 6.63  | 3.30  | 6.31  | 4.68  | 5.16   | 4.16  | 


### Resnet50 on imagenet (flowers data set: image.shape = [3, 224, 224]):

Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | mb=128  |
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|-------:|
|float32| 9.34 | 9.59 | 11.10 | 14.46 | 22.89  | 39.9 |  69.31   | Out of Memory | 
|float16| 8.97  | 8.55  | 9.14  | 8.90  | 12.13  | 18.74 | 31.92  |  59.47       |
|Speedup| 1.04  | 1.12  | 1.22  | 1.63  | 1.89   | 2.13  |  2.17  | |

Total time spent on conv op for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | mb=32 | mb=64  | 
|-------|-----: |-----: |-----: |-----: |------: |------:|-------:|
|float32| 5.8  | 5.54  | 6.59  | 8.45  | 13.87  | 24.5 |  41.1   | 
|float16| 4.5  | 4.67  | 4.23  | 4.27  | 5.66   | 8.84 | 15.3   | 
|Speedup| 1.29 | 1.19  | 1.56  | 1.98  | 2.45   | 2.77  |  2.69  |



### Vgg16 on cifar10 (image.shape = [3, 32, 32]):

Total inference time for one batch:

|       | mb=1 | mb=2 | mb=4 | mb=8 | mb=32 | mb=64 | mb=128 | mb=256 | mb=512 |
|-------|-----:|-----:|-----:|-----:|------:|------:|-------:|-------:|-------:| 
|float32| 3.94 | 4.10 | 4.08 | 4.48 | 6.90  | 9.03  | 14.04  | 24.63  | 45.36  | 
|float16| 3.78 | 3.68 | 3.76 | 3.79 | 4.14  | 4.64  | 6.45   | 10.29  | 17.90  |
|Speedup| 1.04 | 1.12 | 1.09 | 1.18 | 1.67  | 1.95  | 2.18   | 2.39   | 2.53   |



### Resnet32 on cifar10 (image.shape = [3, 32, 32]):

Total inference time for one batch:

|       | mb=1 | mb=2 | mb=4 | mb=8 | mb=32 | mb=64 | mb=128 | mb=256 | mb=512 |
|-------|-----:|-----:|-----:|-----:|------:|------:|-------:|-------:|-------:| 
|float32| 5.30 | 4.87 | 4.77 | 4.98 | 5.26  | 5.80  | 8.10   | 12.91  | 22.2   |
|float16| 5.77 | 5.77 | 5.29 | 5.80 | 5.74  | 5.03  | 5.23   | 7.37   | 11.53  | 
|Speedup|      |      |      |      |       | 1.15  | 1.55   | 1.75   | 1.93   |



## Total time on Titan Xp
All times are in ms (millisecond) averaged over 1000 iterations tested on a single Nvidia Titan Xp GPU with respective to different mini-batch(mb) sizes.

### Vgg16 on imagenet (flowers data set: image.shape = [3, 224, 224]):
Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | 
|-------|-----: |-----: |-----: |-----: |------: |
|float32| 5.79  | 9.02 | 14.75 | 30.87 | 71.33   | 
|float16| 5.47  | 8.01  | 13.41  | 27.82  | 67.04| 
|Speedup| 1.06  | 1.13  | 1.10  | 1.11  | 1.06   | 


### Resnet50 on imagenet (flowers data set: image.shape = [3, 224, 224]):
Total inference time for one batch:

|       | mb=1  | mb=2  | mb=4  | mb=8  | mb=16  | 
|-------|-----: |-----: |-----: |-----: |------: |
|float32| 8.56  | 8.23 | 9.97 | 14.63    | 24.78   | 
|float16| 8.50  | 8.18  | 9.89  | 13.90  | 22.98| 
|Speedup| 1.01  | 1.01  | 1.01  | 1.05  | 1.08   | 
