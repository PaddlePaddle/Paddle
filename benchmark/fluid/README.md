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
      python fluid_benchmark.py --model mnist --parallel 1 --device GPU --with_test
    ```
    You can choose to use GPU/CPU training. With GPU training, you can specify
    `--parallel 1` to run multi GPU training.
* Run distributed training with parameter servers:
    * start parameter servers:
        ```bash
        PADDLE_TRAINING_ROLE=PSERVER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=1 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist --parallel 0 --device GPU --update_method pserver
        ```
    * start trainers:
        ```bash
        PADDLE_TRAINING_ROLE=PSERVER PADDLE_PSERVER_PORT=7164 PADDLE_PSERVER_IPS=127.0.0.1 PADDLE_TRAINERS=1 PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist --parallel 0 --device GPU --update_method pserver
        ```
* Run distributed training using NCCL2
    ```bash
    PADDLE_PSERVER_PORT=7164 PADDLE_TRAINER_IPS=192.168.0.2,192.168.0.3  PADDLE_CURRENT_IP=127.0.0.1 PADDLE_TRAINER_ID=0 python fluid_benchmark.py --model mnist --parallel 0 --device GPU --update_method nccl2
    ```

## Run Distributed Benchmark on Kubernetes Cluster

We provide a script `kube_gen_job.py` to generate Kubernetes yaml files to submit
distributed benchmark jobs to your cluster. To generate a job yaml, just run:

```bash
python kube_gen_job.py --jobname myjob --pscpu 4 --cpu 8 --gpu 8 --psmemory 20 --memory 40 --pservers 4 --trainers 4 --entry "python fluid_benchmark.py --model mnist --parallel 1 --device GPU --update_method pserver --with_test" --disttype pserver
```

Then the yaml files are generated under directory `myjob`, you can run:

```bash
kubectl create -f myjob/
```

The job shall start.
