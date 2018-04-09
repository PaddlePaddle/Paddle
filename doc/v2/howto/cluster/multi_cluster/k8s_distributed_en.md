# Kubernetes Distributed Training

We introduced how to create a PaddlePaddle Job with a single node on Kuberentes in the
previous document.
In this article, we will introduce how to craete a PaddlePaddle job with multiple nodes
on Kubernetes cluster.

## Overall Architecture

Before creating a training job, the users need to deploy the Python scripts and
training data which have already been sliced on the precast path in the distributed file
system(We can use the different type of Kuberentes Volumes to mount different distributed
file system). Before start training, The program would copy the training data into the
Container and also save the models at the same path during training. The global architecture
is as follows:

![PaddlePaddle on Kubernetes Architecture](src/k8s-paddle-arch.png)

The above figure describes a distributed training architecture which contains 3 nodes, each 
Pod would mount a folder of the distributed file system to save training data and models
by Kubernetes Volume. Kubernetes created 3 Pod for this training phase and scheduled these on
3 nodes, each Pod has a PaddlePaddle container. After the containers have been created,
PaddlePaddle would start up the communication between PServer and Trainer and read training
data for this training job.

As the description above, we can start up a PaddlePaddle distributed training job on a ready
Kubernetes cluster as the following steps:

1. [Build PaddlePaddle Docker Image](#Build a Docker Image)
1. [Split training data and upload to the distributed file system](#Upload Training Data)
1. [Edit a YAML file and create a Kubernetes Job](#Create a Job)
1. [Check the output](#Check The Output)

We will introduce these steps as follows:

### Build a Docker Image

PaddlePaddle Docker Image needs to support the runtime environment of `Paddle PServer` and
`Paddle Trainer` process and this Docker Image has the two import features:

- Copy the training data into the container.
- Generate the start arguments of `Paddle PServer` and `Paddle Training` process.

Because of the official Docker Image `paddlepaddle/paddle:latest` has already included the
PaddlePaddle executable file, but above features so that we can use the official Docker Image as
a base Image and add some additional scripts to finish the work of building a new image.
You can reference [Dockerfile](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/howto/usage/cluster/src/k8s_train/Dockerfile).


```bash
$ cd doc/howto/usage/k8s/src/k8s_train
$ docker build -t [YOUR_REPO]/paddle:mypaddle .
```

And then upload the new Docker Image to a Docker hub:

```bash
docker push  [YOUR_REPO]/paddle:mypaddle
```

**[NOTE]**, in the above command arguments, `[YOUR_REPO]` representative your Docker repository,
you need to use your repository instead of it. We will use `[YOUR_REPO]/paddle:mypaddle` to
represent the Docker Image which built in this step.

### Prepare Training Data

We can download and split the training job by creating a Kubernetes Job, or custom your image
by editing [k8s_train](./src/k8s_train/README.md).

Before creating a Job, we need to bind a [persistenVolumeClaim](https://kubernetes.io/docs/user-guide/persistent-volumes) by the different type of
the different distributed file system, the generated dataset would be saved on this volume.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: paddle-data
spec:
  template:
    metadata:
      name: pi
    spec:
      hostNetwork: true
      containers:
      - name: paddle-data
        image: paddlepaddle/paddle-tutorial:k8s_data
        imagePullPolicy: Always
        volumeMounts:
        - mountPath: "/mnt"
          name: nfs
        env:
        - name: OUT_DIR
          value: /home/work/mfs/paddle-cluster-job
        - name: SPLIT_COUNT
          value: "3"
      volumes:
        - name: nfs
          persistentVolumeClaim:
            claimName: mfs
      restartPolicy: Never
```

If success, you can see some information like this:

```base
[root@paddle-kubernetes-node0 nfsdir]$ tree -d
.
`-- paddle-cluster-job
    |-- 0
    |   `-- data
    |-- 1
    |   `-- data
    |-- 2
    |   `-- data
    |-- output
    |-- quick_start
```

The `paddle-cluster-job` above is the job name for this training job; we need 3
PaddlePaddle training node and save the split training data on `paddle-cluster-job` path,
the folder `0`, `1` and `2` representative the `training_id` on each node, `quick_start` folder is used to store training data, `output` folder is used to store the models and logs.


### Create a Job

Kubernetes allow users to create an object with YAML files, and we can use a command-line tool
to create it.

The Job YAML file describes that which Docker Image would be used in this training job, how much nodes would be created, what's the startup arguments of `Paddle PServer/Trainer` process and what's the type of Volumes. You can find the details of the YAML filed in
[Kubernetes Job API](http://kubernetes.io/docs/api-reference/batch/v1/definitions/#_v1_job).
The following is an example for this training job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: paddle-cluster-job
spec:
  parallelism: 3
  completions: 3
  template:
    metadata:
      name: paddle-cluster-job
    spec:
      volumes:
      - name: jobpath
        hostPath:
          path: /home/work/mfs
      containers:
      - name: trainer
        image: [YOUR_REPO]/paddle:mypaddle
        command: ["bin/bash",  "-c", "/root/start.sh"]
        env:
        - name: JOB_NAME
          value: paddle-cluster-job
        - name: JOB_PATH
          value: /home/jobpath
        - name: JOB_NAMESPACE
          value: default
        - name: TRAIN_CONFIG_DIR
          value: recommendation
        - name: CONF_PADDLE_NIC
          value: eth0
        - name: CONF_PADDLE_PORT
          value: "7164"
        - name: CONF_PADDLE_PORTS_NUM
          value: "2"
        - name: CONF_PADDLE_PORTS_NUM_SPARSE
          value: "2"
        - name: CONF_PADDLE_GRADIENT_NUM
          value: "3"
        volumeMounts:
        - name: jobpath
          mountPath: /home/jobpath
      restartPolicy: Never
```

In the above YAML file:
- `metadata.name`, The job name.
- `parallelism`, The Kubernetes Job would create `parallelism` Pods at the same time.
- `completions`, The Job would become the success status only the number of successful Pod(the exit code is 0)
  is equal to `completions`.
- `volumeMounts`, the name field `jobpath` is a key, the `mountPath` field represents
  the path in the container, and we can define the `jobpath` in `volumes` filed, use `hostPath`
  to configure the host path we want to mount.
- `env`, the environment variables in the Container, we pass some startup arguments by
  this approach, some details are as following:
  - JOB_PATH：the mount path in the container
  - JOB_NAME：the job name
  - TRAIN_CONFIG_DIR：the job path in the container, we can find the training data path by
    combine with JOB_NAME.
  - CONF_PADDLE_NIC: the argument `--nics` of `Paddle PServer` process, the network
    device name.
  - CONF_PADDLE_PORT: the argument `--port` of `Paddle PServer` process.
  - CONF_PADDLE_PORTS_NUM: the argument `--ports_num` of `Paddle PServer`, the port number
    for dense prameter update. 
  - CONF_PADDLE_PORTS_NUM_SPARSE：the argument `--ports_num_for_sparse` of `Paddle PServer`,
    the port number for sparse parameter update.
  - CONF_PADDLE_GRADIENT_NUM：the number of training node, the argument 
  `--num_gradient_servers` of `Paddle PServer` and `Paddle Trainer`.

You can find some details information at [here]
(http://www.paddlepaddle.org/docs/develop/documentation/zh/howto/usage/cmd_parameter/detail_introduction_cn.html)。

We can use the command-line tool of Kubernetes to create a Job when we finish the YAML file:

```bash
kubectl create -f job.yaml
```

Upon successful creation, Kubernetes would create 3 Pods as PaddlePaddle training node,
, pull the Docker image and begin to train.


### Checkout the Output

At the process of training, we can check the logs and the output models, such as we store
the output on `output` folder. **NOTE**, `node_0`, `node_1` and `node_2` represent the
`trainer_id` of the PaddlePaddle training job rather than the node id of Kubernetes.

```bash
[root@paddle-kubernetes-node0 output]# tree -d
.
├── node_0
│   ├── server.log
│   └── train.log
├── node_1
│   ├── server.log
│   └── train.log
├── node_2
......
├── pass-00002
│   ├── done
│   ├── ___embedding_0__.w0
│   ├── ___embedding_1__.w0
......
```

We can checkout the status of each training Pod by viewing the logs:

```bash
[root@paddle-kubernetes-node0 node_0]# cat train.log
I1116 09:10:17.123121    50 Util.cpp:155] commandline:
 /usr/local/bin/../opt/paddle/bin/paddle_trainer
    --nics=eth0 --port=7164
    --ports_num=2 --comment=paddle_process_by_paddle
    --pservers=192.168.129.66,192.168.223.143,192.168.129.71
    --ports_num_for_sparse=2 --config=./trainer_config.py
    --trainer_count=4 --num_passes=10 --use_gpu=0
    --log_period=50 --dot_period=10 --saving_period=1
    --local=0 --trainer_id=0
    --save_dir=/home/jobpath/paddle-cluster-job/output
I1116 09:10:17.123440    50 Util.cpp:130] Calling runInitFunctions
I1116 09:10:17.123764    50 Util.cpp:143] Call runInitFunctions done.
[WARNING 2016-11-16 09:10:17,227 default_decorators.py:40] please use keyword arguments in paddle config.
[INFO 2016-11-16 09:10:17,239 networks.py:1282] The input order is [movie_id, title, genres, user_id, gender, age, occupation, rating]
[INFO 2016-11-16 09:10:17,239 networks.py:1289] The output order is [__square_error_cost_0__]
I1116 09:10:17.392917    50 Trainer.cpp:170] trainer mode: Normal
I1116 09:10:17.613910    50 PyDataProvider2.cpp:257] loading dataprovider dataprovider::process
I1116 09:10:17.680917    50 PyDataProvider2.cpp:257] loading dataprovider dataprovider::process
I1116 09:10:17.681543    50 GradientMachine.cpp:134] Initing parameters..
I1116 09:10:18.012390    50 GradientMachine.cpp:141] Init parameters done.
I1116 09:10:18.018641    50 ParameterClient2.cpp:122] pserver 0 192.168.129.66:7164
I1116 09:10:18.018950    50 ParameterClient2.cpp:122] pserver 1 192.168.129.66:7165
I1116 09:10:18.019069    50 ParameterClient2.cpp:122] pserver 2 192.168.223.143:7164
I1116 09:10:18.019492    50 ParameterClient2.cpp:122] pserver 3 192.168.223.143:7165
I1116 09:10:18.019716    50 ParameterClient2.cpp:122] pserver 4 192.168.129.71:7164
I1116 09:10:18.019836    50 ParameterClient2.cpp:122] pserver 5 192.168.129.71:7165
```

## Some Additional Details

### Using Environment Variables

Usually we use the environment varialbes to configurate the PaddlePaddle Job which running on
Kubernetes, `start_paddle.py` provides a start up script to convert the environment variable
to the start up argument of PaddlePaddle process:

```bash
API = "/api/v1/namespaces/"
JOBSELECTOR = "labelSelector=job-name="
JOB_PATH = os.getenv("JOB_PATH") + "/" + os.getenv("JOB_NAME")
JOB_PATH_OUTPUT = JOB_PATH + "/output"
JOBNAME = os.getenv("JOB_NAME")
NAMESPACE = os.getenv("JOB_NAMESPACE")
PADDLE_NIC = os.getenv("CONF_PADDLE_NIC")
PADDLE_PORT = os.getenv("CONF_PADDLE_PORT")
PADDLE_PORTS_NUM = os.getenv("CONF_PADDLE_PORTS_NUM")
PADDLE_PORTS_NUM_SPARSE = os.getenv("CONF_PADDLE_PORTS_NUM_SPARSE")
PADDLE_SERVER_NUM = os.getenv("CONF_PADDLE_GRADIENT_NUM")
```

### Communication between Pods

At the begin of `start_paddle.py`, it would initialize and parse the arguments.

```python
parser = argparse.ArgumentParser(prog="start_paddle.py",
                                     description='simple tool for k8s')
    args, train_args_list = parser.parse_known_args()
    train_args = refine_unknown_args(train_args_list)
    train_args_dict = dict(zip(train_args[:-1:2], train_args[1::2]))
    podlist = getPodList()
```

And then query the status of all the other Pods of this Job by the function `getPodList()`, and fetch `triner_id` by the function `getIdMap(podlist)` if all the Pods status is `RUNNING`.

```python
    podlist = getPodList()
    # need to wait until all pods are running
    while not isPodAllRunning(podlist):
        time.sleep(10)
        podlist = getPodList()
    idMap = getIdMap(podlist)
```

**NOTE**: `getPodList()` would fetch all the pod in the current namespace, if some Pods are running, may cause some error. We will use [statfulesets](https://kubernetes.io/docs/concepts/abstractions/controllers/statefulsets) instead of
Kubernetes Pod or Replicaset in the future.

For the implement of `getIdMap(podlist)`, this function would fetch each IP address of
`podlist` and then sort them to generate `trainer_id`.

```python
def getIdMap(podlist):
    '''
    generate tainer_id by ip
    '''
    ips = []
    for pod in podlist["items"]:
        ips.append(pod["status"]["podIP"])
    ips.sort()
    idMap = {}
    for i in range(len(ips)):
        idMap[ips[i]] = i
    return idMap
```

After getting the `idMap`, we can generate the arguments of `Paddle PServer` and `Paddle Trainer`
so that we can start up them by `startPaddle(idMap, train_args_dict)`.

### Create Job

The main goal of `startPaddle` is generating the arguments of `Paddle PServer` and `Paddle Trainer` processes. Such as `Paddle Trainer`, we parse the environment variable and then get
`PADDLE_NIC`, `PADDLE_PORT`, `PADDLE_PORTS_NUM` and etc..., finally find `trainerId` from
`idMap` according to its IP address.

```python
    program = 'paddle train'
    args = " --nics=" + PADDLE_NIC
    args += " --port=" + str(PADDLE_PORT)
    args += " --ports_num=" + str(PADDLE_PORTS_NUM)
    args += " --comment=" + "paddle_process_by_paddle"
    ip_string = ""
    for ip in idMap.keys():
        ip_string += (ip + ",")
    ip_string = ip_string.rstrip(",")
    args += " --pservers=" + ip_string
    args_ext = ""
    for key, value in train_args_dict.items():
        args_ext += (' --' + key + '=' + value)
    localIP = socket.gethostbyname(socket.gethostname())
    trainerId = idMap[localIP]
    args += " " + args_ext + " --trainer_id=" + \
        str(trainerId) + " --save_dir=" + JOB_PATH_OUTPUT
```
