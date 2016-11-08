# Distributed Training

In this article, we explain how to run distributed Paddle training jobs on clusters.  We will create the distributed version of the single-process training example, [recommendation example](https://github.com/baidu/Paddle/tree/develop/demo/recommendation).

All scripts used in this article are in directory [`paddle/scripts/cluster_train`](https://github.com/baidu/Paddle/tree/develop/paddle/scripts/cluster_train).  They launch distributed Paddle training jobs via SSH.  They also work as a reference solution for users running cluster management systems like MPI and Kubernetes.

## Prerequisite

To make it easy for a Python script to run SSH commands, it is recommended to install Python library [fabric](http://www.fabfile.org/):

```bash
pip install fabric
```

Secondly, go through installing scripts to install PaddlePaddle at all nodes to make sure demo can run as local mode. For CUDA enabled training, we assume that CUDA is installed in ```/usr/local/cuda```, otherwise missed cuda runtime libraries error could be reported at cluster runtime. In one word, the local training environment should be well prepared for the simple scripts.

Then you should prepare same ROOT_DIR directory in all nodes. ROOT_DIR is from in cluster_train/conf.py. Assuming that the ROOT_DIR = /home/paddle, you can create ```paddle``` user account as well, at last ```paddle.py``` can ssh connections to all nodes with ```paddle``` user automatically.

At last you can create ssh mutual trust relationship between all nodes for easy ssh login, otherwise ```password``` should be provided at runtime from ```paddle.py```.

## Prepare Job Workspace

```Job workspace``` is defined as one package directory which contains dependency libraries, train data, test data, model config file and all other related file dependencies.

These ```train/test``` data should be prepared before launching cluster job. To  satisfy the requirement that train/test data are placed in different directory from workspace, PADDLE refers train/test data according to index file named as ```train.list/test.list``` which are used in model config file. So the train/test data also contains train.list/test.list two list file. All local training demo already provides scripts to help you create these two files,  and all nodes in cluster job will handle files with same logical code in normal condition.

Generally, you can use same model file from local training for cluster training. What you should have in mind that, the ```batch_size``` set in ```setting``` function in model file means batch size in ```each``` node of cluster job instead of total batch size if synchronization SGD was used.

Following steps are based on demo/recommendation demo in demo directory.

You just go through demo/recommendation tutorial doc until ```Train``` section, and at last you will get train/test data and model configuration file. Finaly, just use demo/recommendation as workspace for cluster training.

At last your workspace should look like as follow:
```
.
|-- common_utils.py
|-- data
|   |-- config.json
|   |-- config_generator.py
|   |-- meta.bin
|   |-- meta_config.json
|   |-- meta_generator.py
|   |-- ml-1m
|   |-- ml_data.sh
|   |-- ratings.dat.test
|   |-- ratings.dat.train
|   |-- split.py
|   |-- test.list
|   `-- train.list
|-- dataprovider.py
|-- evaluate.sh
|-- prediction.py
|-- preprocess.sh
|-- requirements.txt
|-- run.sh
`-- trainer_config.py
```
Not all of these files are needed for cluster training, but it's not necessary to remove useless files.

```trainer_config.py```
Indicates the model config file.

```train.list``` and ```test.list```
File index. It stores all relative or absolute file paths of all train/test data at current node.

```dataprovider.py```
used to read train/test samples. It's same as local training.

```data```
all files in data directory are refered by train.list/test.list which are refered by data provider.


## Prepare Cluster Job Configuration

The options below must be carefully set in cluster_train/conf.py

```HOSTS```  all nodes hostname or ip that will run cluster job. You can also append user and ssh port with hostname, such as root@192.168.100.17:9090.

```ROOT_DIR``` workspace ROOT directory for placing JOB workspace directory

```PADDLE_NIC``` the NIC(Network Interface Card) interface name for cluster communication channel, such as eth0 for ethternet, ib0 for infiniband.

```PADDLE_PORT``` port number for cluster commnunication channel

```PADDLE_PORTS_NUM``` the number of port used for cluster communication channle. if the number of cluster nodes is small(less than 5~6nodes), recommend you set it to larger, such as 2 ~ 8, for better network performance.

```PADDLE_PORTS_NUM_FOR_SPARSE``` the number of port used for sparse updater cluster commnunication channel. if sparse remote update is used, set it like ```PADDLE_PORTS_NUM```

```LD_LIBRARY_PATH``` set addtional LD_LIBRARY_PATH for cluster job. You can use it to set CUDA libraries path.

Default Configuration as follow:

```python
HOSTS = [
        "root@192.168.100.17",
        "root@192.168.100.18",
        ]

'''
workspace configuration
'''

#root dir for workspace
ROOT_DIR = "/home/paddle"

'''
network configuration
'''
#pserver nics
PADDLE_NIC = "eth0"
#pserver port
PADDLE_PORT = 7164
#pserver ports num
PADDLE_PORTS_NUM = 2
#pserver sparse ports num
PADDLE_PORTS_NUM_FOR_SPARSE = 2

#environments setting for all processes in cluster job
LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib64"
```

### Launching Cluster Job
```paddle.py``` provides automatical scripts to start all PaddlePaddle cluster processes in different nodes. By default, all command line options can set as ```paddle.py``` command options and ```paddle.py``` will transparently and automatically set these options to PaddlePaddle lower level processes.

```paddle.py```provides two distinguished command option for easy job launching.

```job_dispatch_package```  set it with local ```workspace```directory, it will be dispatched to all nodes set in conf.py. It could be helpful for frequent hacking workspace files, otherwise frequent mulit-nodes workspace deployment could make your crazy.
```job_workspace```  set it with already deployed workspace directory, ```paddle.py``` will skip dispatch stage to directly launch cluster job with all nodes. It could help to reduce heavy
dispatch latency.

```cluster_train/run.sh``` provides command line sample to run ```demo/recommendation``` cluster job, just modify ```job_dispatch_package``` and ```job_workspace``` with your defined directory, then:
```
sh run.sh
```

The cluster Job will start in several seconds.

### Kill Cluster Job
```paddle.py``` can capture ```Ctrl + C``` SIGINT signal to automatically kill all processes launched by it. So just stop ```paddle.py``` to kill cluster job. You should mannally kill job if program crashed.

### Check Cluster Training Result
Check log in $workspace/log for details, each node owns same log structure.

```paddle_trainer.INFO```
It provides almost all interal output log for training,  same as local training. Check runtime model convergence here.

```paddle_pserver2.INFO```
It provides pserver running log, which could help to diagnose distributed error.

```server.log```
It provides stderr and stdout of pserver process. Check error log if training crashs.

```train.log```
It provides stderr and stdout of trainer process. Check error log if training crashs.

### Check Model Output
After one pass finished, model files will be writed in ```output``` directory in node 0.
```nodefile``` in workspace indicates the node id of current cluster job.
