# Performance for distributed vgg16

## Test Result

### Single node single thread

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | 16.74 | - |
| PaddlePaddle v2 | - | - | 17.60 | - |
| TensorFlow | - | - | - | - |

### different batch size

- PServer Count: 10
- Trainer Count: 20
- Metrics: samples / sec

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | 247.40 | - | - |
| PaddlePaddle v2 | - | - | 256.14 | - |
| TensorFlow | - | - | - | - |

### different pserver number

- Trainer Count: 100
- Batch Size: 64
- Metrics: mini-batch / sec

| PServer Count | 10 | 20 | 40 | 60 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | - | - | - | - |
| PaddlePaddle v2 | - | - | - | - |
| TensorFlow | - | - | - | - |

### Accelerate rate

| Trainer Counter | 20 | 40 | 80 | 100 |
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
