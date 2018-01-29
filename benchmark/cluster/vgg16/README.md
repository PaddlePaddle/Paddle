# Performance for distributed vgg16

## Test Result

### Hardware Infomation

- CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
- cpu MHz		: 2101.000
- cache size	: 20480 KB

### Single Node Single Thread

- PServer Count: 10
- Trainer Count: 20
- Metrics: samples / sec

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 15.44 | 16.32 | 16.74 | 16.79 |
| PaddlePaddle v2 | 15.97 | 17.04 | 17.60 | 17.83 |
| TensorFlow | - | - | - | - |

### different batch size

- PServer Count: 10
- Trainer Count: 20
- Per trainer CPU Core: 1
- Metrics: samples / sec

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 190.20 | 222.15 | 247.40 | 258.18 |
| PaddlePaddle v2 | 170.96 | 233.71 | 256.14 | 329.23 |
| TensorFlow | - | - | - | - |


### Accelerate rate

- Pserver Count: 20
- Batch Size: 128
- Metrics: samples / sec

| Trainer Counter | 20 | 40 | 80 | 100 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 291.06 | 518.80 | 836.26 | 1019.29 |
| PaddlePaddle v2 | 356.28 | - | - | 1041.99 |
| TensorFlow | - | - | - | - |

### different pserver number

- Trainer Count: 100
- Batch Size: 128
- Metrics: mini-batch / sec

| PServer Count | 10 | 20 | 40 | 60 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - |
| TensorFlow | - | - | - | - |


## Steps to run the performance test

1. You must re-compile PaddlePaddle and enable `-DWITH_DISTRIBUTE` to build PaddlePaddle with distributed support.
1. When the build finishes, copy the output `whl` package located under `build/python/dist` to current directory.
1. Run `docker build -t [image:tag] .` to build the docker image and run `docker push [image:tag]` to push the image to reponsitory so kubernetes can find it.
1. Run `kubectl create -f pserver.yaml && kubectl create -f trainer.yaml` to start the job on your kubernetes cluster (you must configure the `kubectl` client before this step).
1. Run `kubectl get po` to get running pods, and run `kubectl logs [podID]` to fetch the pod log of pservers and trainers.

Check the logs for the distributed training progress and analyze the performance.

## Enable verbos logs

Edit `pserver.yaml` and `trainer.yaml` and add an environment variable `GLOG_v=3` to see what happend in detail.
