
# PaddlePaddle Client
PaddlePaddle clinet is a command line tool, before startting a cluster train, you need to install it on your latop, you can sumit a cluster train job looks like:

```bash
paddle k8s job <job-name>
    --package-path= ./demo
    --module=demo.train
    --pserver-count=2
    --trainer-count=4
    --output=${OUTPUT_DIR}
    --input=${INPUT_DIR}
```

- job-name: you can specify a name for every job, and the name should be uniq.
- package-path: your package files, including py file and the dependencies.
- module: Include main function, it is the trainer entrance.
- pserver-count: specify the parameter process count.
- trainer-count: specify the trainer process count.
- output: output path on distributed storage.
- input: input path on distributed storage.

For this command, it will setup a master process, 2 pserver processes and 4 triner processes on cluster.

# Master process
- Setup master process

  While user submit a distributed train job through PaddlePaddle client, it will setup a master process on cluster, the master process provids a http service for receiving package files and some cluster parameter.

- Saving package

  Master process will receive the package files and save on the distributed storage, for every job, master will generate a random id called *job-id*, the package file path looks like:
  ```bash
  \_<job-id-0>-pacakge
      \_train.py
      \_settings.py
      \_...
  \_<job-id-1>-package
      \_train.py
  ```

- Setup parameter and trainer

  Master process will set up parameter server process and trainer process, on kubernetes, master will deploy two deployment for parameter server and trainer.


- Runtime environment

  On kubernets, we use PaddlePaddle production image for the standard environment,such as **paddlepaddle/paddle:0.10.0rc2**. For every parameter server process and trainer process, they use the same image.The package files the user upload through PaddlePaddle Client will volume on the pod.

- Data source

  Input data should be saved on the distributed and you have the access.

- Sharding Mass Data

  You can submit a job for sharding data, it looks like:
  ```bash
  paddle k8s job <job-name>
      --pcakage-path=./sharding-data
      --module-name=sharding-data.main
      --input=${INPUT_DIR}
      --output=${OUTPUT_DIR
  ```

# Different cluster management

PaddlePaddle client support plugins for different cluster management client, such as kubernetes looks like `paddle k8s ...`, MPI looks like `paddl mpi...`,but runinig process are the same:

<img src="./submit-job.png" width="500">
