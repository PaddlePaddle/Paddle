# Performance for Distributed vgg16

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

### Different Batch Size

- PServer Count: 10
- Trainer Count: 20
- Per trainer CPU Core: 1
- Metrics: samples / sec

| Batch Size | 32 | 64 | 128 | 256 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 190.20 | 222.15 | 247.40 | 258.18 |
| PaddlePaddle v2 | 170.96 | 233.71 | 256.14 | 329.23 |
| TensorFlow | - | - | - | - |


### Accelerate Rate

- Pserver Count: 20
- Batch Size: 128
- Metrics: samples / sec

| Trainer Count | 20 | 40 | 80 | 100 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid | 263.29 (78.64%) | 518.80 (77.47%) | 836.26 (62.44%) | 1019.29 (60.89%) |
| PaddlePaddle v2 (need more tests) | 326.85 (92.85%) | 534.58 (75.93%) | 853.30 (60.60%) | 1041.99 (59.20%) |
| TensorFlow | - | - | - | - |

### Different Pserver Count

- Trainer Count: 60
- Batch Size: 128
- Metrics: samples/ sec

| PServer Count | 3 | 6 |10 | 20 |
| -- | -- | -- | -- | -- |
| PaddlePaddle Fluid(should fix in next PR) | 589.1 | 592.6 | 656.4 | 655.8 |
| PaddlePaddle v2 | 593.4 | 791.3 | 729.7 | 821.7 |
| TensorFlow | - | - | - | - |

*The performance gap between Fuild and v2 comes from the network interference.*


## Steps to Run the Performance Test

1. You must re-compile PaddlePaddle and enable `-DWITH_DISTRIBUTE` to build PaddlePaddle with distributed support.
1. When the build finishes, copy the output `whl` package located under `build/python/dist` to current directory.
1. Run `docker build -t [image:tag] .` to build the docker image and run `docker push [image:tag]` to push the image to reponsitory so kubernetes can find it.
1. Run `kubectl create -f pserver.yaml && kubectl create -f trainer.yaml` to start the job on your kubernetes cluster (you must configure the `kubectl` client before this step).
1. Run `kubectl get po` to get running pods, and run `kubectl logs [podID]` to fetch the pod log of pservers and trainers.

Check the logs for the distributed training progress and analyze the performance.

## Enable Verbos Logs

Edit `pserver.yaml` and `trainer.yaml` and add an environment variable `GLOG_v=3` and `GLOG_logtostderr=1` to see what happend in detail.
