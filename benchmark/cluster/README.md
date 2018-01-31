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
  - PaddlePaddle v2: paddlepaddle/paddle:0.11.0
  - PaddlePaddle Fluid: paddlepaddle/paddle:[commit-id]
  - TensorFlow: tensorflow/tensorflow:1.5.0-rc0

- Model
  vgg16 is used in this benchmark.

## Cases

- Variable
  - Batch Size of training data.
  - PServer count of the training job.
  - The number of trainers.

- Invariant
  - The resource of trainer/pserver Pod.

### Measure the Performance for Different Batch Size

- PServer Count: 40
- Trainer Count: 100
- Metrics: mini-batch / sec

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - |
| TensorFlow | - | - | - | - |

### Measure the Performance for Different PServer Count

- Trainer Count: 60
- Batch Size: 128
- Metrics: mini-batch / sec

| PServer Count | 3 | 6 | 10 | 20 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 589.1 | 592.6 | 656.4 | 655.8 |
| PaddlePaddle v2 | 412.2 | 368.4 | 346.8 | 283.2 |
| TensorFlow | - | - | - | - |

### Measure Parallel Efficiency By Increasing Trainer Count

- PServer Count: 20
- Batch Size: 64
- Metrics:

$S = \div(T1, TN)$

which S is the ratio of T1 over TN, training time of 1 and N trainers.
The parallel efficiency is:

$E = \div(S, N)$

| Trainer Counter | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - | - | - | - | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - | - | - | - | - | - | - | - | - |
| TensorFlow | - | - | - | - | - | - | - | - | - | - | - | - | - |

## Reproduce the benchmark

TODO
