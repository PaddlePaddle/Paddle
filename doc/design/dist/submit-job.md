
# PaddlePaddle Client
PaddlePaddle client is command line tool, you can use a PaddlePaddle client to start a local train and submit a distributed training job to kubernetes cluster.

<img src="./submit-job.png" width="500">

# Running your training locally
Execute `paddle local train` to run your local train.
```bash
paddle local train
  --pcakage-path=./demo
  --module=demo.train
  --input=<input_dir>
  --output=<output_dir>
  --image=<paddle_image>
  --e=NUM_PASS=4
```
- package-path: your trainer code python package
- module: include a main function, trainer entrance.
- input: input directory, for local train, it's a host path.
- output: output directory, for local train, it's a host path.
- image: paddlepaddle production image
- e: environment varible

When you start the local train, the client starts a docker container like:
```bash
  docker run --rm
  -v <input_dir>:/train/input
  -v <output_dir>:/train/output
  -v <package-path>:/train/package
  -e NUM_PASS=4 <paddle_image>
  python /train/package/train.py
```


# Submit distributed training job
You can use `paddle submit job <job-name>` to submit a distributed training job.

```bash
paddle job submit train <job-name>
  --package-path=/train/quick_start
  --module=quick_start.train
  --input=<input_dir>
  --output=<output_dir>
  --trainers=4
  --pservers=2
  --image:<your image>
  -e=NUM_PASS=5
```

- job-name: you should specify a unique job name,
- package-path=your python package files
- module: include the main function, trainer entrance
- input: input directory on distributed file system
- output: output directory on distributed file system
- trainers: trainer process count
- pserver: parameter process count
- image: your trainer docker image, include your trainer files and dependencies.
- command:
- e: environment variable

## Build your docker image
Before submitting a distributed training, you should build your docker image, here
is a simple example for a PaddlePaddle trainer:
```bash
paddle_example
    |-Dcokerfile
    `-quick_start
      |-trainer.py
      `-dataset.py
```
Execute `docker build -t <your repo>/paddle_dist_example .` on directory `paddle_example` and then
push the image use `docker push <your repo>/paddle_dist_example`
`Dockerfile` should include your python package, such as:
```bash
FROM:paddlepaddle/paddle:0.10.0rc2
ADD ./quick_start /train/quick_start
CMD ["python", "/train/quick_stat/train.py"]
```

## Master process
Master process a bootstrap and manager process for a distributed job, it deploys parameter server process, trainer process and dispatch task for the trainers, it is implemented by golang.
1. Setup master process

  While user submits a distributed training job, PaddlePaddle client deploys a master process which is a  Job resource naming `<job-name>-master` on kubernetes.
1. Startup pservers and trainers

  Master process will deploy pserver and trainer on kubernetes, they are also job resource, naming `<job-name>-pserver` and `<job-name>-trainer`. Because of trainer need the IP of pserver, so there should be a dependency for the startup order.
  - Deploy pserver job, and waiting for the status becoming `RUNINIG`.
  - Fetch all pserver's IP as trainer parameters.
  - Deploy trainer job.
1. Dispatch task to trainer

  Detail description is [here](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/dist#master-process)

## Data source
1. Distributed file system

  You can upload your training data to distributed file system, such as GlustereFS,
  PaddlePaddle support a default reader for reading data from distributed file system.
1. HTTP server

  TODO
1. Real-time data

  TODO
