# Fluid Benchmark

This directory contains several models configurations and tools that used to run
Fluid benchmarks for local and distributed training.


## Run the Benchmark

To start, run the following command to get the full help message:

```bash
python fluid_benchmark.py --help
```

Currently supported `--model` argument include:

* mnist
* resnet
    * you can chose to use different dataset using `--data_set cifar10` or
      `--data_set flowers`.
* vgg
* stacked_dynamic_lstm
* machine_translation

* Run the following command to start a benchmark job locally:
    ```bash
      python fluid_benchmark.py --model mnist --device GPU
    ```
    You can choose to use GPU/CPU training. With GPU training, you can specify
    `--gpus <gpu_num>` to run multi GPU training.
    You can set async mode parameter server. With async mode, you can specify
    `--async_mode` to train model asynchronous.
* Run distributed training with parameter servers:
    * see [run_fluid_benchmark.sh](https://github.com/PaddlePaddle/Paddle/blob/develop/benchmark/fluid/run_fluid_benchmark.sh) as an example.
    * start parameter servers:
        ```bash
        PADDLE_TRAINING_ROLE=PSERVER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=1 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist  --device GPU --update_method pserver
        sleep 15
        ```
    * start trainers:
        ```bash
        PADDLE_TRAINING_ROLE=TRAINER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=1 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist  --device GPU --update_method pserver
        ```
* Run distributed training using NCCL2
    ```bash
    PADDLE_PSERVER_PORT=7164 PADDLE_TRAINER_IPS=192.168.0.2,192.168.0.3  PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist --device GPU --update_method nccl2
    ```

## Prepare the RecordIO file to Achieve Better Performance

Run the following command will generate RecordIO files like "mnist.recordio" under the path
and batch_size you choose, you can use batch_size=1 so that later reader can change the batch_size
at any time using `fluid.batch`.

```bash
python -c 'from recordio_converter import *; prepare_mnist("data", 1)'
```

## Run Distributed Benchmark on Kubernetes Cluster

You may need to build a Docker image before submitting a cluster job onto Kubernetes, or you will
have to start all those processes mannually on each node, which is not recommended.

To build the Docker image, you need to choose a paddle "whl" package to run with, you may either
download it from
http://www.paddlepaddle.org/docs/develop/documentation/zh/build_and_install/pip_install_en.html or
build it by your own. Once you've got the "whl" package, put it under the current directory and run:

```bash
docker build -t [your docker image name]:[your docker image tag] .
```

Then push the image to a Docker registry that your Kubernetes cluster can reach.

We provide a script `kube_gen_job.py` to generate Kubernetes yaml files to submit
distributed benchmark jobs to your cluster. To generate a job yaml, just run:

```bash
python kube_gen_job.py --jobname myjob --pscpu 4 --cpu 8 --gpu 8 --psmemory 20 --memory 40 --pservers 4 --trainers 4 --entry "python fluid_benchmark.py --model mnist --gpus 8 --device GPU --update_method pserver " --disttype pserver
```

Then the yaml files are generated under directory `myjob`, you can run:

```bash
kubectl create -f myjob/
```

The job shall start.


## Notes for Run Fluid Distributed with NCCL2 and RDMA

Before running NCCL2 distributed jobs, please check that whether your node has multiple network
interfaces, try to add the environment variable `export NCCL_SOCKET_IFNAME=eth0` to use your actual
network device.

To run high-performance distributed training, you must prepare your hardware environment to be
able to run RDMA enabled network communication, please check out [this](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/howto/cluster/nccl2_rdma_training.md)
note for details.
