# Fabric

## Prepare a Linux cluster

Run `kubectl -f ssh_servers.yaml` under the directory:  `paddle/scripts/cluster_train_v2/fabric/docker_cluster` will launch a demo cluster. Run `kubectl get po -o wide` to get IP addresses of these nodes.

## Launching Cluster Job
`paddle.py` provides automatical scripts to start all PaddlePaddle cluster processes in different nodes. By default, all command line options can be set as `paddle.py` command options and `paddle.py` will transparently and automatically set these options to PaddlePaddle lower level processes.

`paddle.py`provides two distinguished command option for easy job launching.

- `job_dispatch_package` set it with local `workspace` directory, it will be dispatched to all nodes which is set in `conf.py`. It could be helpful for frequently manipulating workspace files. otherwise, frequent multi-nodes workspace deployment is very annoying.
- `job_workspace`  set it with already deployed workspace directory, `paddle.py` will skip dispatch stage to directly launch cluster job with all nodes. It could help to reduce heavy
dispatch latency.

`cluster_train/run.sh` provides command line sample to run `demo/recommendation` cluster job, just modify `job_dispatch_package` and `job_workspace` with your defined directory, then:
```
sh run.sh
```

The cluster Job will start in several seconds.

## Kill Cluster Job
`paddle.py` can capture `Ctrl + C` SIGINT signal to automatically kill all processes launched by it. So just stop `paddle.py` to kill cluster job. You should manually kill the job if the program crashed.

## Check Cluster Training Result
Check log in $workspace/log for details, each node owns same log structure.

`paddle_trainer.INFO`
It provides almost all internal output log for training,  same as local training. Check runtime model convergence here.

`paddle_pserver2.INFO`
It provides parameter server running log, which could help to diagnose distributed error.

`server.log`
It provides stderr and stdout of parameter server process. Check error log if training crashes.

`train.log`
It provides stderr and stdout of trainer process. Check error log if training crashes.

## Check Model Output
After one pass finished, model files will be written in `output` directory in node 0.
`nodefile` in workspace indicates the node id of current cluster job.
