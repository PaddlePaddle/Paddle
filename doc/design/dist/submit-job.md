# PaddlePaddle Client

If a user wants to startup a local train, he will startup a PaddlePaddle product Docker image firstly, and then
execute `python train.py` in the Docker container.The details about PaddlePaddle Docker image is [here]()

If a user wants to startup a distributed training job, he will use the PaddlePaddle Client.The PaddlePaddle Client package the trainer files, upload to PaddlePaddle Server and then startup multiple parameter server and trainer processes as user's configuration for a distributed training job.The User can upload the distributed training job with python code or a command line tool.

The relation of PaddlePaddle, kubernetes and docker:
<img src="./submit-job.png" width="500">

## Submit a Distributed Training Job With Python PaddlePaddle Client

Users will Call `paddle.dist.train` and provide distributed training configuration as a parameter.
```python
paddle.dist.train(

    k8s_user="paddle",
    k8s_pssword="paddle-dev",
    job_name="quickstart",
    trainers=8,
    pservers=4,
    input=/quickstart/input,
    output=/quickstart/output,
    base_image="paddlepaddle/paddle:0.10.rc2",
    use_gpu=False)
```

## Submit a Distributed Training Job With a Command Line Tool

### Configurate PaddlePaddle client

Users should configure PaddlePaddle client by the configuration file firstly, the default path:
`$HOME/.paddle/config`.

```yaml
apiVersion: v1
dockerRegistry:
  domain: domain.com //default is docker.io
  username: <username>
  password: <password>
paddleServer: http://<paddle server domain>:<paddle server port>
```

### Submit a Distributed Training Job

Users will execute the command `paddle job submit` and provides distributed training configuration as parameter.
```bash
paddle job submit\
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

### PaddlePaddle Client Commands:
The command line tool also supports these subcommand:
- `paddle train`: start a training job
- `paddle list`: list all PaddlePaddle jobs in current namespace
- `paddle cancel`: cancel a running job.
- `paddle status`: status of a PaddlePaddle job
- `paddle version`: show PaddlePaddle client and PaddlePaddle server version info.
- `paddle upload`: upload training data to distributed storage.
- `paddle download`: download training data from a distributed storage.

# Runtime Environment On kubernetes

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

# Paddle Server
Paddle server is running on kubernetes, users will configure the server address in [PaddlePaddle client configuration file](#configurate-paddlepaddle-client)

- RESTful API

  Paddle server provides a RESTful HTTP server receives the trainer packages, list PaddlePaddle job etc...
  - `POST /v1/packages` upload trainer package and save them on GlustereFS
  - `POST /v1/trainer/job` submit a trainer job
  - `GET /v1/jobs/` list all job
  - `GET /v1/jobs/<job-name>` the status of a job
  - `DELETE /v1/jobs/<job-name>` cancel a job
  - `GET /v1/version` paddle server version

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

# Work Feature
- V1
  - Submit a distributed training job with python code.
  - Support `paddle upload` and `paddle download`
- V1
  - Submit a distributed training job with python code, support `paddle train`, `paddle list` and etc...
