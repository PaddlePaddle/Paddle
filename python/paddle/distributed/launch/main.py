# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .context import Context


def launch():
    """
    Paddle distribution training entry ``python -m paddle.distributed.launch``.
    
    Usage:
        .. code-block:: bash
            :name: code-block-bash1

            python -m paddle.distributed.launch [-h] [--master MASTER] [--rank RANK]
                   [--log_level LOG_LEVEL] [--nnodes NNODES]
                   [--nproc_per_node NPROC_PER_NODE] [--log_dir LOG_DIR]
                   [--run_mode RUN_MODE] [--job_id JOB_ID] [--devices DEVICES]
                   [--host HOST] [--servers SERVERS] [--trainers TRAINERS]
                   [--trainer_num TRAINER_NUM] [--server_num SERVER_NUM]
                   [--gloo_port GLOO_PORT] [--with_gloo WITH_GLOO]
                   [--max_restart MAX_RESTART] [--elastic_level ELASTIC_LEVEL]
                   [--elastic_timeout ELASTIC_TIMEOUT]
                   training_script ...


    Base Parameters:
        - ``--master``: The master/rendezvous server, support http:// and etcd://, default with http://. e.g., ``--master=127.0.0.1:8080``. Default ``--master=None``.

        - ``--rank``: The rank of the node, can be auto assigned by master. Default ``--rank=-1``.

        - ``--log_level``: The log level to set for logging.setLevel which can be CRITICAL/ERROR/WARNING/INFO/DEBUG/NOTSET, case insensitive. The rank 0 log will not print in the terminal by default, while you can enable it by adding --log_level=debug. Default ``--log_level=INFO``.

        - ``--nnodes``: The number of nodes for a distributed job, it can be a range in elastic mode, e.g., ``--nnodes=2:3``. Default ``--nnodes=1``.

        - ``--nproc_per_node``: The number of processes to launch on a node. In gpu training, it should be less or equal to the gpus number of you system.  e.g., ``--nproc_per_node=8``

        - ``--log_dir``: The path for each process's log. e.g., ``--log_dir=output_dir``. Default ``--log_dir=log``.

        - ``--run_mode``: The run mode of job, can be:collective/ps/ps-heter. e.g., ``--run_mode=ps``. Default ``--run_mode=collective``.

        - ``--job_id``: The job unique id, it affects the log files' name. e.g., ``--job_id=job1``. Default ``--job_id=default``.

        - ``--devices``: The selected accelerate devices on nodes, can be gpu/xpu/npu/mlu etc.. e.g., ``--devices=0,1,2,3`` will launch four training processes each bound to one device.

        - ``training_script``: The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script. e.g., ``traing.py``

        - ``training_script_args``: The args of training_script. e.g., ``--lr=0.1``

    Collective Parameters:
        - ``--ips``: [DEPRECATED] Paddle cluster nodes ips, e.g., ``--ips=192.168.0.16,192.168.0.17``. Default ``--ips=127.0.0.1``.

    Parameter-Server Parameters:
        - ``--servers``: User defined servers ip:port, e.g., ``--servers="192.168.0.16:6170,192.168.0.17:6170"``

        - ``--trainers``: User defined trainers ip:port, e.g., ``--trainers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172"``

        - ``--workers``: [DEPRECATED] The same as trainers.

        - ``--trainer_num``: Number of trainers on each node, can be 0.

        - ``--worker_num``: [DEPRECATED] The same as trainer_num.

        - ``--server_num``: Number of servers on each node, can be 0.

        - ``--heter_workers``: User defined heter workers ip1:port1;ip2:port2, e.g., ``--heter_workers="192.168.0.16:6172;192.168.0.17:6172"``

        - ``--heter_worker_num``: Number of heter_workers in each stage (It recommend to set when in the emulated distributed environment using single node)
        
        - ``--heter_devices``: Type of heter_device in each stage

        - ``--gloo_port``: Gloo http Port. Default ``--gloo_port=6767``.

        - ``--with_gloo``: Using gloo or not. Default ``--with_gloo=0``.

    Elastic Parameters:
        - ``--max_restart``: The maximum restart times for an elastic job. Default ``--max_restart=3``.

        - ``--elastic_level``: The elastic level: -1: disable, 0: failed exit, peers hold, 1: internal restart. Default ``--elastic_level=-1``.

        - ``--elastic_timeout``: Seconds to wait before elastic job begin to train. Default ``--elastic_timeout=30``.


    Returns:
        - ``None``

    Examples 0 (master, ip/port auto detection):
        .. code-block:: bash
            :name: code-block-example-bash0

            # For training on multi node, run the following command in one of the nodes

            python -m paddle.distributed.launch --nnodes 2 train.py

            # Then the following info will be print

            # Copy the following command to other nodes to run.
            # --------------------------------------------------------------------------------
            # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
            # --------------------------------------------------------------------------------

            # Follow the instruction above and paste the command in other nodes can launch a multi nodes training job.

            # There are two ways to launch a job with the same command for multi nodes training
            # 1) using the following command in every nodes, make sure the ip is one of the training node and the port is available on that node
            # python -m paddle.distributed.launch --master 10.0.0.1:38714 --nnodes 2 train.py
            # 2) using the following command in every nodes with a independent etcd service
            # python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2 train.py

            # This functionality works will for both collective and ps mode and even with other arguments.


    Examples 1 (collective, single node):
        .. code-block:: bash
            :name: code-block-example-bash1
            
            # For training on single node using 4 gpus.

            python -m paddle.distributed.launch --devices=0,1,2,3 train.py --lr=0.01
        
    Examples 2 (collective, multi node):
        .. code-block:: bash
            :name: code-block-example-bash2

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 

            # On 192.168.0.16:

            python -m paddle.distributed.launch --devices=0,1,2,3 --master=192.168.0.16:8090 train.py --lr=0.01

            # On 192.168.0.17:
            python -m paddle.distributed.launch --devices=0,1,2,3 --master=192.168.0.16:8090 train.py --lr=0.01
        
    Examples 3 (ps, cpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash3

            # To simulate distributed environment using single node, e.g., 2 servers and 4 workers.
            
            python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01
        
    Examples 4 (ps, cpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash4

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

            # On 192.168.0.16:

            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

            # Or with master, the following command run 2 server and 2 trainer on each node.

            python -m paddle.distributed.launch --master 192.168.0.16:9090 --server_num=2 --trainer_num=2 --nnodes 2 train.py


    Examples 5 (ps, gpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash5

            # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, each worker use single gpu.
            
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01
            
    Examples 6 (ps, gpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash6

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server and 2 workers.

            # On 192.168.0.16:

            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.16:6172,192.168.0.17:6171,192.168.0.17:6172" train.py --lr=0.01

    Examples 7 (ps-heter, cpu + gpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash7

            # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, two workers use gpu, two workers use cpu.
            
            export CUDA_VISIBLE_DEVICES=0,1
            python -m paddle.distributed.launch --server_num=2 --worker_num=2 --heter_worker_num=2 train.py --lr=0.01
            
    Examples 8 (ps-heter, cpu + gpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash8

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 1 server, 1 gpu worker, 1 cpu worker.

            # On 192.168.0.16:

            export CUDA_VISIBLE_DEVICES=0
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

            # On 192.168.0.17:

            export CUDA_VISIBLE_DEVICES=0
            python -m paddle.distributed.launch --servers="192.168.0.16:6170,192.168.0.17:6170" --workers="192.168.0.16:6171,192.168.0.17:6171" --heter_workers="192.168.0.16:6172,192.168.0.17:6172" train.py --lr=0.01

    Examples 9 (elastic):
        .. code-block:: bash
            :name: code-block-example-bash9

            # With the following command, the job will begin to run immediately if 4 nodes are ready,
            # or it will run after elastic_timeout if only 2 or 3 nodes ready
            python -m paddle.distributed.launch --master etcd://10.0.0.1:2379 --nnodes 2:4 train.py
            
            # once the number of nodes changes between 2:4 during training, the strategy holds

    """

    # initialize the context to run
    ctx = Context()

    if ctx.is_legacy_mode():

        # legacy mode
        from paddle.distributed.fleet import launch
        launch.launch()

    else:

        from . import controllers

        # initialize the selected controller
        c = controllers.init(ctx)

        # run the pods
        c.run()

        # manager or just wait pod
        c.finalize()


if __name__ == "__main__":
    launch()
