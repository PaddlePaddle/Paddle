# Cluster Training Benchmark

## Setup

- Platform
  - Kubernetes: v1.6.2
  - Linux Kernel: v3.10.0

- Resource
  - CPU: 10 Cores per Pod
  - Memory: 5GB per Pod

- Docker Image

  We use different base Docker Image to run the benchmark on Kubernetes:
  - PaddlePaddle v2: paddlepaddle/paddle:latest
  - PaddlePaddle Fluid: paddlepaddle/paddle:latest
  - TensorFlow: tensorflow/tensorflow:latest

- Model
  A digits recognize model and MNIST dataset is used in this benchmark.

## Compare the Performance

- Variable
  - Batch Size of training data.
  - PServer count of the training job.

- Invariant
  - The number of trainers.
  - The resource of trainer/pserver Pod.

- Metrics
  - We use `batch/sec` to measure the training performance.

### BatchSize

| BatchSize | 64 | 128 | 256 | 512 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - |
| TensorFlow | - | - | - | - |

### PServer Count

| PServer Count | 10 | 20 | 40 | 80 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - |
| TensorFlow | - | - | - | - |

## Reproduce the benchmark

TODO
