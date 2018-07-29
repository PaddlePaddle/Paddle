# Anakin GPU Benchmark

## Machine:

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`
>  cuDNN: `v7`


## Counterpart of anakin  :

The counterpart of **`Anakin`** is the acknowledged high performance inference engine **`NVIDIA TensorRT 3`** ,   The models which TensorRT 3 doesn't support we use the custom plugins  to support.

## Benchmark Model

The following convolutional neural networks are tested with both `Anakin` and `TenorRT3`.
 You can use pretrained caffe model or the model trained by youself.

> Please note that you should transform caffe model or others into anakin model with the help of [`external converter ->`](../docs/Manual/Converter_en.md)


- [Vgg16](#1)   *caffe model can be found [here->](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)*
- [Yolo](#2)  *caffe model can be found [here->](https://github.com/hojel/caffe-yolo-model)*
- [Resnet50](#3)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Resnet101](#4)  *caffe model can be found [here->](https://github.com/KaimingHe/deep-residual-networks#models)*
- [Mobilenet v1](#5)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [Mobilenet v2](#6)  *caffe model can be found [here->](https://github.com/shicai/MobileNet-Caffe)*
- [RNN](#7)  *not support yet*

We tested them on single-GPU with single-thread.

### <span id = '1'>VGG16 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 8.8690 | 8.2815 |
| 2 | 15.5344 | 13.9116 |
| 4 | 26.6000 | 21.8747 |
| 8 | 49.8279 | 40.4076 |
| 32 | 188.6270 | 163.7660 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 963 | 997 |
| 2 | 965 | 1039 |
| 4 | 991 | 1115 |
| 8 | 1067 | 1269 |
| 32 | 1715 | 2193 |


### <span id = '2'>Yolo </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 16.4596| 15.2124 |
| 2 | 26.6347| 25.0442 |
| 4 | 43.3695| 43.5017 |
| 8 | 80.9139 | 80.9880 |
| 32 | 293.8080| 310.8810 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1569 | 1775 |
| 2 | 1649 | 1815 |
| 4 | 1709 | 1887 |
| 8 | 1731 | 2031 |
| 32 | 2253 | 2907 |

### <span id = '3'> Resnet50 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 4.2459   |  4.1061 |
| 2 |  6.2627  |  6.5159 |
| 4 | 10.1277  | 11.3327 |
| 8 | 17.8209  | 20.6680 |
| 32 | 65.8582 | 77.8858 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 531  | 503 |
| 2 | 543  | 517 |
| 4 | 583 | 541 |
| 8 | 611 | 589 |
| 32 |  809 | 879 |

### <span id = '4'> Resnet101 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 7.5562 | 7.0837 |
| 2 | 11.6023 | 11.4079 |
| 4 | 18.3650 | 20.0493 |
| 8 | 32.7632 | 36.0648 |
| 32 | 123.2550 | 135.4880 |

- GPU Memory Used (`MB)`

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 701  | 683 |
| 2 | 713  | 697 |
| 4 | 793 | 721 |
| 8 | 819 | 769 |
| 32 | 1043 | 1059 |

###  <span id = '5'> MobileNet V1 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 45.5156  |  1.3947 |
| 2 |  46.5585  |  2.5483 |
| 4 | 48.4242  | 4.3404 |
| 8 |  52.7957 |  8.1513 |
| 32 | 83.2519 | 31.3178 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 329  | 283 |
| 2 | 345  | 289 |
| 4 | 371 | 299 |
| 8 | 393 | 319 |
| 32 |  531 | 433 |

###  <span id = '6'> MobileNet V2</span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 65.6861 | 2.9842 |
| 2 | 66.6814 | 4.7472 |
| 4 | 69.7114 | 7.4163 |
| 8 | 76.1092 | 12.8779 |
| 32 | 124.9810 | 47.2142 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 341 | 293 |
| 2 | 353 | 301 |
| 4 | 385 | 319 |
| 8 | 421 | 351 |
| 32 | 637 | 551 |

## How to run those Benchmark models?

> 1. At first, you should parse the caffe model with [`external converter`](https://github.com/PaddlePaddle/Anakin/blob/b95f31e19993a192e7428b4fcf852b9fe9860e5f/docs/Manual/Converter_en.md).
> 2. Switch to *source_root/benchmark/CNN* directory. Use 'mkdir ./models' to create ./models and put anakin models into this file.
> 3. Use command 'sh run.sh', we will create files in logs to save model log with different batch size. Finally, model latency summary will be displayed on the screen.
> 4. If you want to get more detailed information with op time, you can modify CMakeLists.txt with setting `ENABLE_OP_TIMER` to `YES`, then recompile and run. You will find detailed information in  model log file.
