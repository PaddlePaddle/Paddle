# Cluster Training

We provide this simple scripts to help you to launch cluster training Job to harness PaddlePaddle's distributed trainning. For MPI and other cluster scheduler refer this naive script to implement more robust cluster training platform by yourself.

The following cluster demo is based on RECOMMENDATION local training demo in PaddlePaddle ```demo/recommendation``` directory.  Assuming you enter the cluster_scripts/ directory.

## Pre-requirements

Firstly,

```bash
pip install fabric
```

Secondly, go through installing scripts to install PaddlePaddle at all nodes to make sure demo can run as local mode.

Then you should prepare same ROOT_DIR directory in all nodes. ROOT_DIR is from in cluster_scripts/conf.py. Assuming that the ROOT_DIR = /home/paddle, you can create ```paddle``` user account as well, at last ```paddle.py``` can ssh connections to all nodes with ```paddle``` user automatically.

At last you can create ssh mutual trust relationship between all nodes for easy ssh login, otherwise ```password``` should be provided at runtime from ```paddle.py```.

## Prepare Job Workspace

```Job workspace``` is defined as one package directory which contains dependency libraries, train data, test data, model config file and all other related file dependencies.

These ```train/test``` data should be prepared before launching cluster job. To  satisfy the requirement that train/test data are placed in different directory from workspace, PADDLE refers train/test data according to index file named as ```train.list/test.list``` which are used in model config file. So the train/test data also contains train.list/test.list two list file. All local training demo already provides scripts to help you create these two files,  and all nodes in cluster job will handle files with same logical code in normal condition.

Generally, you can use same model file from local training for cluster training. What you should have in mind that, the ```batch_size``` set in ```setting``` function in model file means batch size in ```each``` node of cluster job instead of total batch size if synchronization SGD was used.

Following steps are based on demo/recommendation demo in demo directory.

You just go through demo/recommendation tutorial doc until ```Train``` section, and at last you will get train/test data and model configuration file. Besides, you can place paddle binaries and related dependencies files in this demo/recommendation directory as well. Finaly, just use demo/recommendation as workspace for cluster training.

At last your workspace should look like as follow:
```
.
|-- conf
|   `-- trainer_config.conf
|-- test
|   |-- dnn_instance_000000
|-- test.list
|-- train
|   |-- dnn_instance_000000
|   |-- dnn_instance_000001
`-- train.list
```
```conf/trainer_config.conf```
Indicates the model config file.

```test``` and ```train```
Train/test data. Different node should owns different parts of all Train data. This simple script did not do this job, so you should prepare it at last. All test data should be placed at node 0 only.

```train.list``` and ```test.list```
File index. It stores all relative or absolute file paths of all train/test data at current node.



## Prepare Cluster Job Configuration

Set serveral options must be carefully set in cluster_scripts/conf.py

```HOSTS```  all nodes hostname or ip that will run cluster job. You can also append user and ssh port with hostname, such as root@192.168.100.17:9090.

```ROOT_DIR``` workspace ROOT directory for placing JOB workspace directory

```PADDLE_NIC``` the NIC(Network Interface Card) interface name for cluster communication channel, such as eth0 for ethternet, ib0 for infiniband.

```PADDLE_PORT``` port number for cluster commnunication channel

```PADDLE_PORTS_NUM``` the number of port used for cluster communication channle. if the number of cluster nodes is small(less than 5~6nodes), recommend you set it to larger, such as 2 ~ 8, for better network performance.

```PADDLE_PORTS_NUM_FOR_SPARSE``` the number of port used for sparse updater cluster commnunication channel. if sparse remote update is used, set it like ```PADDLE_PORTS_NUM```

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
```

### Launching Cluster Job
```paddle.py``` provides automatical scripts to start all PaddlePaddle cluster processes in different nodes. By default, all command line options can set as ```paddle.py``` command options and ```paddle.py``` will transparently and automatically set these options to PaddlePaddle lower level processes.

```paddle.py```provides two distinguished command option for easy job launching.

```job_dispatch_package```  set it with local ```workspace```directory, it will be dispatched to all nodes set in conf.py. It could be helpful for frequent hacking workspace files, otherwise frequent mulit-nodes workspace deployment could make your crazy.
```job_workspace```  set it with already deployed workspace directory, ```paddle.py``` will skip dispatch stage to directly launch cluster job with all nodes. It could help to reduce heavy
dispatch latency.

```cluster_scripts/run.sh``` provides command line sample to run ```demo/recommendation``` cluster job, just modify ```job_dispatch_package``` and ```job_workspace``` with your defined directory, then:
```
sh run.sh
```

The cluster Job will start in several seconds.

### Kill Cluster Job
```paddle.py``` can capture ```Ctrl + C``` SIGINT signal to automatically kill all processes launched by it. So just stop ```paddle.py``` to kill cluster job.

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
