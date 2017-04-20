
# PaddlePaddle Client
PaddlePaddle client is command line interface for running PaddlePaddle distributed training job or start up a local training job.

The relation of PaddlePaddle, kubernetes and docker:

<img src="./submit-job.png" width="500">


# Running Local Training Job
You can execute the command: `paddle train` with flag `--locally` to start up a local train.
```bash
paddle train \
  --locally \
  --job-name=quickstart \
  --package-path=./demo \
  --entry-point="python train.py" \
  --input=<input_dir> \
  --output=<output_dir> \
  --image=<paddle_image> \
  --env=NUM_PASS=4
```
- `job-name`: your local training job name
- `package-path`: your trainer code python package
- `entry-point`: an entry point for startup trainer process
- `input`: input directory, for local train, it's a host path.
- `output`: output directory, for local train, it's a host path.
- `base-image`: paddlepaddle production image
- `env`: environment varible

When users start a local training job, PaddlePaddle client starts a docker container like:
```bash
docker run --rm \
  --name quickstart \
  -v <host input dir>:/input \
  -v <host output dir>:/output \
  -v <package files>:/package \
  -e NUM_PASS=4 \
  -e PYTHONPATH=/package \
  paddlepaddle/paddle:0.10.0rc3 \
  python train.py
```


# Running Distributed Training Job

## Configurate PaddlePaddle client

You should configure PaddlePaddle client by the configuration file firstly, the default path:
`$HOME/.paddle/config`.

```yaml
apiVersion: v1
dockerRegistry:
  domain: domain.com //default is docker.io
  username: <username>
  password: <password>
paddleServer: http://<paddle server domain>:<paddle server port>
```


## Submit a Distributed Training Job
Users will submit a distributed training job with the command: `paddle train` without flag `--locally`.

```bash
paddle train \
  --job-name=cluster-quickstart \
  --package-path=$PWD/quick_start \
  --entry-point="python train.py" \
  --input=<input-dir> \
  --output=<output-dir> \
  --trainers=4 \
  --pservers=2 \
  --base-image:<paddle-image> \
  --use-gpu=true \
  --trainer-gpu-num=1 \
  --env="NUM_PASS=5"
```

- `job-name`: you should specify a unique job name
- `package-path`: python package files on your host
- `entry-point`: an entry point for startup trainer process
- `input`: input directory on distributed file system
- `output`: output directory on distributed file system
- `trainers`: trainer process count
- `pserver`: parameter process count
- `base-image`: your trainer docker image, include your trainer files and dependencies.
- `use-gpu`: whether it is a GPU train
- `trainer-gpu-num`: how much GPU card for one paddle trainer process, it's requirements only if `use-gpu=true`,
- `env`: environment variable

## Runtime Environment On kubernetes

For a distributed training job, there is two docker image called `runtime docker image` and `base docker image`, the `runtime docker image` is actually running in kubernetes.

- Base Docker Image

  Usually, the `base docker image` is PaddlePaddle product docker image including paddle binary files and trainer startup script file. And of course, users can specify any image name hosted on any docker registry which users have the right access.

- Runtime Docker Image

  package the trainer package which user upload and some python dependencies into a `runtime docker image` base on `base docker image`, this is done automatically by Paddle Server.

- Python Dependencies

  users will provide a `requirments.txt` file in packages path, to list python dependencies packages, such as:
  ```txt
  pillow
  protobuf==3.1.0
  ```
  some other details about `requirements` is [here](https://pip.readthedocs.io/en/1.1/requirements.html).

  Here is an example project:
  ```bash
    paddle_example
      |-quick_start
        |-trainer.py
        |-dataset.py
        |-requirments.txt
  ```
  Execute the command: `paddle train --package-path=./paddle_eample/quick_start ...`, PaddlePaddle client will upload the trainer package(quick_start)and setup parameters to [Paddle Server](#paddle-server)

## Paddle Server
Paddle server is running on kubernetes, users will configure the server address in [PaddlePaddle client configuration file](#configurate-paddlepaddle-client)

- Paddle Server

  Paddle server is an HTTP server which receives the trainer package and saves them on GlustereFS.

- Build Runtime Docker Image On Cloud(kubernetes)

  Paddle server deploys a kubernetes Job and builds runtime docker image in Pod, pserver and trainer pod will use this runtime docker image to startup pserver and trainer process.

  There are some benefits for building Docker image on the cloud:
  - Users only need to upload the training package files, does not dependency docker engine, docker registry.
  - If we want to change another image type, such as RKT, the user does not need to care about it.

- Start Up PSrvers and Trainers Job
  - Deploy pserver job, it's a kubernetes StatefulSet.
  - Deploy trainer job, it's a kubernetes Job.
    - Waiting for all pserver pod is running.
    - Fetch all pserver address using kubernetes API and put environment variable.
    - Start up trainer process with `entry-point`.

## PaddlePaddle Client Commands:
To run local training job with flag `--locally` and distributed training job without it.
- `paddle train`: start a training job
- `paddle list`: list all PaddlePaddle jobs in current namespace
- `paddle cancel`: cancel a running job.
- `paddle status`: status of a PaddlePaddle job
- `paddle version`: show PaddlePaddle client and PaddlePaddle server version info.
