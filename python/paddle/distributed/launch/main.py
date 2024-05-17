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

import paddle
from paddle.distributed.launch.context import Context

ctx = None


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
        - ``--master``: The master/rendezvous server, support ``http://`` and ``etcd://``, default with ``http://``. e.g., ``--master=127.0.0.1:8080``. Default ``--master=None``.

        - ``--rank``: The rank of the node, can be auto assigned by master. Default ``--rank=-1``.

        - ``--log_level``: The log level to set for logging.setLevel which can be CRITICAL/ERROR/WARNING/INFO/DEBUG/NOTSET, case insensitive. Default ``--log_level=INFO``.

        - ``--nnodes``: The number of nodes for a distributed job, it can be a range in elastic mode, e.g., ``--nnodes=2:3``. Default ``--nnodes=1``.

        - ``--nproc_per_node``: The number of processes to launch on a node. In gpu training, it should be less or equal to the gpus number of you system.  e.g., ``--nproc_per_node=8``

        - ``--log_dir``: The path for each process's log. e.g., ``--log_dir=output_dir``. Default ``--log_dir=log``.

        - ``--run_mode``: The run mode of job, can be:collective/ps/ps-heter/rpc. e.g., ``--run_mode=ps``. Default ``--run_mode=collective``.

        - ``--job_id``: The job unique id, it affects the log files' name. e.g., ``--job_id=job1``. Default ``--job_id=default``.

        - ``--devices``: The selected accelerate devices on nodes, can be gpu/xpu etc.. e.g., ``--devices=0,1,2,3`` will launch four training processes each bound to one device.

        - ``training_script``: The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script. e.g., ``training.py``

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

    IPU Parameters:
        IPU distributed launch only requires and allows three arguments ``--devices``, ``training_script`` and ``training_script_args``.
        The ``--devices`` is the number of IPU devices. e.g., ``--devices=4`` will launch the training program with four IPU devices.
        The ``training_script`` is only allowed to set as ``ipu``.
        The ``training_script_args`` includes arguments required by IPU distributed launch and illustrated as below.
        ``Examples 10`` has provided a example of paddle.distributed.launch with IPUs.

        - ``--hosts``: The hosts for IPU distributed training. Each host is able to include multiple processes.

        - ``--nproc_per_host``: The number of processes launched per host. Each process is able to include multiple replicas.

        - ``--ipus_per_replica``: The number of IPUs requested per replica. Each replica is able to include multiple IPUs.

        - ``--ipu_partition``: The partition name of IPU devices.

        - ``--vipu_server``: The ip of the IPU device manager.

        - ``training_script``: The full path to the IPU distributed training program/script to be launched in parallel. e.g., ``training.py``.

        - ``training_script_args``: The args of the IPU distributed training program/script. e.g., ``--lr=0.1``.

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

    Examples 10 (ipu):
        .. code-block:: bash
            :name: code-block-example-bash10

            # With the following command, the job will begin to run the distributhed program with IPUs
            # Require `devices` as the number of IPUs
            # Require `training_script` to be set as `ipu`
            # Require `training_script_args` as the arguments of IPU distributed training instead of the arguments of the training program/script
            # Please Check the `IPU Parameters` for details
            python -m paddle.distributed.launch --devices 4 ipu --hosts=localhost --nproc_per_host=2 --ipus_per_replica=1 --ipu_partition=pod16 --vipu_server=127.0.0.1 train.py

    Examples 11 (rpc, cpu, single node):
        .. code-block:: bash
            :name: code-block-example-bash11

            # Training on single node with two local servers
            python -m paddle.distributed.launch --master 127.0.0.1:8765 --nnodes 1 --nproc_per_node 2 --rank 0 --run_mode rpc train.py

    Examples 12 (rpc, cpu, multi node):
        .. code-block:: bash
            :name: code-block-example-bash12

            # For training on multiple nodes, e.g., 192.168.0.16, 192.168.0.17 where each node with 2 servers.

            # On 192.168.0.16

            python -m paddle.distributed.launch --master 192.168.0.16:8765 --nnodes 2 --nproc_per_node 2 --rank 0 --run_mode rpc train.py

            # On 192.168.0.17

            python -m paddle.distributed.launch --master 192.168.0.16:8765 --nnodes 2 --nproc_per_node 2 --rank 1 --run_mode rpc train.py

    """

    # initialize the context to run
    global ctx
    ctx = Context()

    if ctx.is_legacy_mode():
        # legacy mode
        from paddle.distributed.fleet import launch

        launch.launch()

    elif ctx.is_auto_tuner_mode():
        import copy
        import json
        import logging
        import os
        import sys
        import time

        from paddle.distributed.auto_tuner.recorder import HistoryRecorder
        from paddle.distributed.auto_tuner.tuner import AutoTuner
        from paddle.distributed.auto_tuner.utils import (
            add_overlap_performance,
            find_error_from_log,
            gen_new_args,
            gen_new_ctx,
            read_completed,
            read_log,
            read_step_time_log,
        )
        from paddle.distributed.launch import controllers

        start_time = time.time()
        # read user defined tuner config json
        if not ctx.args.auto_tuner_json.endswith(".json"):
            raise ValueError("Please use '.json' as the file name suffix.")
        try:
            with open(ctx.args.auto_tuner_json, "r") as f:
                tuner_cfg = json.load(f)
        except:
            raise ValueError("Please check your auto tuner json whether valid.")

        logger = logging.getLogger('auto_tuner')
        logger.setLevel(logging.INFO)
        auto_tuner_log_path = os.path.join(
            os.path.dirname(ctx.args.auto_tuner_json),
            f'{os.path.basename(ctx.args.auto_tuner_json).split(".")[0]}_auto_tuner.log',
        )
        handler = logging.FileHandler(auto_tuner_log_path, mode="w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # copy training script args
        if ctx.args.training_script.endswith('.py'):
            if os.environ.get("WITH_COVERAGE") == "ON":
                entrypoint = [
                    sys.executable,
                    "-u",
                    "-m",
                    "coverage",
                    "run",
                    "--branch",
                    "-p",
                    ctx.args.training_script,
                ]
            else:
                entrypoint = [sys.executable, "-u", ctx.args.training_script]
        elif ctx.args.training_script.endswith('.pyxes'):
            entrypoint = [sys.executable, ctx.args.training_script]
        else:
            entrypoint = [ctx.args.training_script]
        entrypoint.extend(ctx.args.training_script_args)
        raw_args = copy.deepcopy(ctx.args.training_script_args)

        # get nodes and gpus from args
        if not ctx.args.devices:
            gpus_per_node = 8
        else:
            gpus_per_node = len(ctx.args.devices.split(","))
        nnodes = ctx.args.nnodes
        if isinstance(nnodes, str):
            nnodes = int(nnodes.split(":")[0])
        else:
            nnodes = int(nnodes)
        tuner_cfg["nodes"] = nnodes
        tuner_cfg["gpus_per_node"] = gpus_per_node
        tuner_cfg["num_gpus"] = gpus_per_node * tuner_cfg["nodes"]
        if not tuner_cfg.get("search_algo", None):
            tuner_cfg["search_algo"] = {"name": "grid"}
        mode = tuner_cfg.get("mode", None)

        history_file_path = os.path.join(
            os.path.dirname(ctx.args.auto_tuner_json),
            f'{os.path.basename(ctx.args.auto_tuner_json).split(".")[0]}_history.csv',
        )
        sorted_ips = []
        ip = None
        if nnodes > 1:
            from paddle.distributed.launch.utils.etcd_client import ETCDClient

            assert "etcd://" in ctx.args.master
            master_ip, port = ctx.args.master.strip("etcd://").split(':')
            client = ETCDClient(host=master_ip, port=port)
            client.delete("best_cfg")
            client.delete_prefix("auto_tuner")

            import socket

            try:
                hostname = socket.gethostname()
                ip = socket.gethostbyname(socket.getfqdn(hostname))
            except:
                ip = '127.0.0.1'
            assert ip != '127.0.0.1'
            if tuner_cfg["search_algo"].get("estimated_num_gpus", None):
                # get all machine ips and sort them
                # to avoid etcd deleting key and adding key at the same time
                time.sleep(5)
                path = f"auto_tuner/ip/{ip}"
                while not client.put(path, f"{ip}".encode('latin-1')):
                    time.sleep(1)

                ips = list(client.get_prefix("auto_tuner/ip/"))
                size = len(ips)
                while size != nnodes:
                    time.sleep(1)
                    client.put(path, f"{ip}".encode('latin-1'))
                    ips = list(client.get_prefix("auto_tuner/ip/"))
                    size = len(ips)
                sorted_ips = sorted([i[0].decode() for i in ips])
                logger.info(
                    f"The total count of nodes is {len(sorted_ips)} and sorted ips are {sorted_ips}."
                )

        # get max time per task run
        max_time_per_task = tuner_cfg.get("max_time_per_task", 1800)
        tuner_cfg["max_time_per_task"] = max_time_per_task
        ctx.max_time_per_task = max_time_per_task

        # warmup
        warmup_time = (
            max_time_per_task
            if "warmup_time" not in tuner_cfg
            else tuner_cfg.get("warmup_time")
        )

        # max_search_time
        max_search_time = tuner_cfg.get("max_search_time", None)

        # buffer and memory
        buffer = tuner_cfg.get("buffer", None)
        max_mem_usage = tuner_cfg.get("max_mem_usage", None)

        is_first_task = True
        # build history recorder
        recorder = HistoryRecorder(tuner_cfg)

        job_id = 0
        error_task_nums = 0
        ctx.args.max_restart = -1
        raw_ctx = copy.deepcopy(ctx)

        # gbs search
        if (
            tuner_cfg.get('model_cfg', {}).get('global_batch_size', 'auto')
            == "auto"
        ):
            # adjust micron batch size until out of memory to get best global batch size
            gbs_tuner_cfg = copy.deepcopy(tuner_cfg)
            gbs_tuner_cfg["search_algo"] = "gbs"
            gbs_tuner = AutoTuner(gbs_tuner_cfg)

            gbs_cur_cfg = gbs_tuner.search_once()
            best_gbs = None

            # every task has own job id
            job_id += 1
            task_job_id = "gbs_tuner_" + str(job_id)
            ctx.args.job_id = task_job_id

            while gbs_cur_cfg:
                ctx = copy.deepcopy(raw_ctx)
                log_dir = "Job{}_GBSSearch/GBS{}_DP{}_MP{}_PP{}_Sharding_degree_{}_stage_{}_MBS{}_Recompute_{}_granularity_{}".format(
                    job_id,
                    gbs_cur_cfg["global_batch_size"],
                    gbs_cur_cfg["dp_degree"],
                    gbs_cur_cfg["mp_degree"],
                    gbs_cur_cfg["pp_degree"],
                    gbs_cur_cfg["sharding_degree"],
                    gbs_cur_cfg["sharding_stage"],
                    gbs_cur_cfg["micro_batch_size"],
                    gbs_cur_cfg["use_recompute"],
                    gbs_cur_cfg["recompute_granularity"],
                )
                ctx.args.log_dir = log_dir

                # generate script args of task
                gbs_new_args = gen_new_args(
                    raw_args, gbs_cur_cfg, gbs_tuner_cfg
                )
                ctx.args.training_script_args = gbs_new_args

                # launch task
                ctx.logger.info(
                    f"Launch task from auto tuner: job_id {task_job_id}, log_dir {log_dir}, config {gbs_cur_cfg}"
                )
                logger.info(
                    f"Launch task from auto tuner: job_id {task_job_id}, log_dir {log_dir}, config {gbs_cur_cfg}"
                )
                c = controllers.init(ctx)
                c.run()

                # process generated result
                # TODO differentiate out of memory and no loss(maybe over time)
                # TODO integrate memory and metric read
                metric, mem, err = read_log(
                    path=ctx.args.log_dir,
                    metric_file="workerlog.0",
                    target_metric=tuner_cfg["metric_cfg"]["name"],
                    memory_file=f"{ctx.args.job_id}.gpu.log",
                )

                if err & (1 << 0):
                    ctx.logger.warning(
                        f"Read metric failed for parameters: {log_dir}"
                    )
                    logger.warning(
                        f"Read metric failed for parameters: {log_dir}"
                    )
                    # for pruner use
                    gbs_cur_cfg['time'] = -1
                    gbs_cur_cfg[tuner_cfg['metric_cfg']['name']] = None
                    gbs_cur_cfg["max_mem_usage"] = mem

                if err & (1 << 1):
                    ctx.logger.warning(
                        f"Out of memory for parameters: {log_dir}"
                    )
                    logger.warning(f"Out of memory for parameters: {log_dir}")
                    # for pruner use
                    gbs_cur_cfg['time'] = -1
                    gbs_cur_cfg[tuner_cfg['metric_cfg']['name']] = None
                    gbs_cur_cfg["max_mem_usage"] = "OOM"

                # not err & (1 << 1): do not record memory usage when out of memory
                if err & (1 << 2) and not err & (1 << 1):
                    ctx.logger.warning(
                        f"Read memory usage failed for parameters: {log_dir}"
                    )
                    logger.warning(
                        f"Read memory usage failed for parameters: {log_dir}"
                    )
                    gbs_cur_cfg["max_mem_usage"] = None

                if not err:
                    # for pruner use
                    gbs_cur_cfg['time'] = metric
                    gbs_cur_cfg[tuner_cfg['metric_cfg']['name']] = metric
                    gbs_cur_cfg["max_mem_usage"] = mem

                if err & (1 << 0) or err & (1 << 1):
                    # no metric or out of memory, end gbs search
                    break

                # store and update args for next round
                gbs_cur_cfg["job_id"] = job_id
                best_gbs = gbs_cur_cfg["global_batch_size"]
                recorder.add_cfg(**gbs_cur_cfg)
                c.finalize(exit=False)
                recorder.store_history("./tuner_gbs_history.csv")

                # new cfgs for next round
                gbs_new_cfg = gbs_tuner.search_once()
                gbs_cur_cfg = copy.deepcopy(gbs_new_cfg)
                gbs_tuner.add_cfg(gbs_cur_cfg)

                # per task launch interval
                time.sleep(3)
            # prevent no valid global batch size found
            if best_gbs is None:
                raise ValueError(
                    f"No valid global batch size found, check memory or valid search time. cur_tuner_cfg{gbs_tuner_cfg}"
                )
            # set best global batch size to tuner cfg
            tuner_cfg["model_cfg"]["global_batch_size"] = best_gbs

            recorder.store_history("./tuner_gbs_history.csv")
            recorder.clean_history()

            end_time = time.time()
            ctx.logger.info(
                f"AutoTuner for GBS search ends in {end_time - start_time}s."
            )
            logger.info(
                f"AutoTuner for GBS search ends in {end_time - start_time}s."
            )

        # build AutoTuner to get new config
        auto_tuner = AutoTuner(tuner_cfg)
        logger.info(
            f"Launch {len(auto_tuner.algo.all_tasks)} tasks by auto tuner: "
        )
        resume_csv_file_path = tuner_cfg.get(
            "resume_csv_file_path", history_file_path
        )
        auto_tuner.resume_form_history(resume_csv_file_path)
        cur_cfg = auto_tuner.search_once()
        auto_tuner.add_cfg(cur_cfg)
        error_msg = (
            "No config can search. Please check if there are any situations "
            + "where GBS is unable to divide dp degree or shading degree, "
            + "or if there are related configurations of the model such as "
            + "hidden_size cannot be evenly divided by mp degree, "
            + "num_ Layers cannot divide pp degree."
        )

        assert cur_cfg is not None, error_msg
        while cur_cfg:
            task_start_time = time.time()
            ctx = copy.deepcopy(raw_ctx)
            if is_first_task:
                ctx.max_time_per_task = warmup_time
            is_first_task = False
            # auto tuner supports dp, mp, pp, micro batch size, sharding, recompute by default and every task has own log dir
            global_batch_size = (
                cur_cfg["global_batch_size"]
                if "global_batch_size" in cur_cfg
                else tuner_cfg["model_cfg"]["global_batch_size"]
            )
            acc_steps = (
                global_batch_size
                // cur_cfg["dp_degree"]
                // cur_cfg["sharding_degree"]
                // cur_cfg["micro_batch_size"]
            )
            cur_cfg["acc_steps"] = acc_steps
            cur_cfg["global_batch_size"] = global_batch_size

            # every task has own job id
            job_id += 1
            task_job_id = "auto_tuner_" + str(job_id)
            ctx.args.job_id = task_job_id
            log_dir = "Job{}_GBS{}_DP{}_MP{}_PP{}_VPP{}_Sharding{}_Stage{}_MBS{}_Recompute_{}_Granularity_{}_AccStep{}".format(
                job_id,
                global_batch_size,
                cur_cfg["dp_degree"],
                cur_cfg["mp_degree"],
                cur_cfg["pp_degree"],
                cur_cfg["vpp_degree"],
                cur_cfg["sharding_degree"],
                cur_cfg["sharding_stage"],
                cur_cfg["micro_batch_size"],
                cur_cfg["use_recompute"],
                cur_cfg["recompute_granularity"],
                cur_cfg["acc_steps"],
            )
            if "sharding_overlap" in cur_cfg:
                log_dir = log_dir + f"_Overlap_{cur_cfg['sharding_overlap']}"
            if "refined_recompute" in tuner_cfg:
                for key in tuner_cfg["refined_recompute"]:
                    dir_name = "".join(i.capitalize() for i in key.split("_"))
                    dir_name += str(cur_cfg[key])
                    log_dir = log_dir + "_" + dir_name

            if "custom_search_dim" in tuner_cfg:
                for key in tuner_cfg["custom_search_dim"]:
                    dir_name = "".join(i.capitalize() for i in key.split("_"))
                    dir_name += str(cur_cfg[key])
                    log_dir = log_dir + "_" + dir_name

            ctx.args.log_dir = os.path.join(
                os.path.dirname(ctx.args.auto_tuner_json), log_dir
            )

            # generate the script arguments and launch configuration JSON/YAML for the task.
            cur_cfg["log_dir_name"] = log_dir
            new_args = gen_new_args(raw_args, cur_cfg, tuner_cfg)
            ctx.args.training_script_args = new_args
            cur_cfg.pop("log_dir_name")

            # launch task
            ctx.logger.info(
                f"Launch task: job_id {task_job_id}, log_dir {log_dir}"
            )
            logger.info(f"Launch task: job_id {task_job_id}, log_dir {log_dir}")

            cur_resume_cfg = auto_tuner.get_cfg_from_resume(cur_cfg)
            if cur_resume_cfg:
                cur_cfg = cur_resume_cfg
                cur_cfg['job_id'] = job_id
                auto_tuner.history_cfgs.pop(-1)
                auto_tuner.add_cfg(cur_cfg)
                if (
                    recorder.additional_metric_key is None
                    and "additional_metric_key" in cur_cfg
                ):
                    recorder.additional_metric_key = cur_cfg[
                        "additional_metric_key"
                    ]
                recorder.add_cfg(**cur_cfg)
                cur_best_cfgs, err = recorder.get_best(
                    metric=tuner_cfg['metric_cfg']['name'],
                    direction=tuner_cfg['metric_cfg']['OptimizationDirection'],
                    buffer=buffer,
                    max_mem_usage=max_mem_usage,
                )
                if not err:
                    to_json_str = json.dumps(cur_best_cfgs)
                    ctx.logger.info(f"Current best config: {to_json_str}")
                    logger.info(f"Current best config: {to_json_str}")
                else:
                    ctx.logger.info(
                        "Get best config failed. Currently no config can be run."
                    )
                    logger.info(
                        "Get best config failed. Currently no config can be run."
                    )
                if (
                    "sharding_overlap" in cur_cfg
                    and cur_cfg["sharding_overlap"]
                ):
                    add_overlap_performance(
                        cur_cfg, tuner_cfg, recorder.history
                    )

                if cur_cfg["error_info"]:
                    error_task_nums += 1
                error_info = cur_cfg["error_info"]
                task_nums = len(auto_tuner.algo.all_tasks)
                cur_task_id = auto_tuner.algo.idx
                ctx.logger.info(
                    "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                        cur_task_id,
                        task_nums,
                        cur_task_id - job_id,
                        error_task_nums,
                        error_info,
                        round(
                            (task_nums - cur_task_id) * max_time_per_task / 60,
                            2,
                        ),
                    )
                )
                logger.info(
                    "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                        cur_task_id,
                        task_nums,
                        cur_task_id - job_id,
                        error_task_nums,
                        error_info,
                        round(
                            (task_nums - cur_task_id) * max_time_per_task / 60,
                            2,
                        ),
                    )
                )
                recorder.store_history(history_file_path)
                # generate a new config
                new_cfg = auto_tuner.search_once()
                cur_cfg = copy.deepcopy(new_cfg)
                auto_tuner.add_cfg(cur_cfg)
                continue

            # in single dp estimation scene, just some nodes not all nodes run
            ctx = gen_new_ctx(ctx, cur_cfg, tuner_cfg)
            actual_nnodes = (
                int(ctx.args.nnodes.split(":")[0])
                if not isinstance(ctx.args.nnodes, int)
                else ctx.args.nnodes
            )
            if sorted_ips:
                actual_exec_ips = sorted_ips[:actual_nnodes]
                if ip not in actual_exec_ips:
                    cur_cfg = client.get(f"auto_tuner/{log_dir}")[0]
                    wait_start_time = time.time()
                    while not cur_cfg:
                        wait_end_time = time.time()
                        if (
                            wait_end_time - wait_start_time
                            > tuner_cfg["max_time_per_task"] + 30
                        ):
                            raise ValueError(f"Wait {log_dir} failed")
                        time.sleep(3)
                        cur_cfg = client.get(f"auto_tuner/{log_dir}")[0]
                    logger.info(
                        f"Receive that task {log_dir} has ended by etcd."
                    )
                    ctx.logger.info(
                        f"Receive that task {log_dir} has ended by etcd."
                    )
                    cur_cfg = json.loads(cur_cfg.decode())
                    auto_tuner.history_cfgs.pop(-1)
                    auto_tuner.add_cfg(cur_cfg)
                    if (
                        recorder.additional_metric_key is None
                        and "additional_metric_key" in cur_cfg
                    ):
                        recorder.additional_metric_key = cur_cfg[
                            "additional_metric_key"
                        ]
                    recorder.add_cfg(**cur_cfg)
                    cur_best_cfgs, err = recorder.get_best(
                        metric=tuner_cfg['metric_cfg']['name'],
                        direction=tuner_cfg['metric_cfg'][
                            'OptimizationDirection'
                        ],
                        buffer=buffer,
                        max_mem_usage=max_mem_usage,
                    )
                    if not err:
                        to_json_str = json.dumps(cur_best_cfgs)
                        ctx.logger.info(f"Current best config: {to_json_str}")
                        logger.info(f"Current best config: {to_json_str}")
                    else:
                        ctx.logger.info(
                            "Get best config failed. Currently no config can be run."
                        )
                        logger.info(
                            "Get best config failed. Currently no config can be run."
                        )
                    if (
                        "sharding_overlap" in cur_cfg
                        and cur_cfg["sharding_overlap"]
                    ):
                        add_overlap_performance(
                            cur_cfg, tuner_cfg, recorder.history
                        )
                    has_error = cur_cfg["has_error"]
                    if has_error:
                        error_task_nums += 1
                    error_info = cur_cfg["error_info"]
                    task_nums = len(auto_tuner.algo.all_tasks)
                    cur_task_id = auto_tuner.algo.idx
                    ctx.logger.info(
                        "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                            cur_task_id,
                            task_nums,
                            cur_task_id - job_id,
                            error_task_nums,
                            error_info,
                            round(
                                (task_nums - cur_task_id)
                                * max_time_per_task
                                / 60,
                                2,
                            ),
                        )
                    )
                    logger.info(
                        "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                            cur_task_id,
                            task_nums,
                            cur_task_id - job_id,
                            error_task_nums,
                            error_info,
                            round(
                                (task_nums - cur_task_id)
                                * max_time_per_task
                                / 60,
                                2,
                            ),
                        )
                    )
                    recorder.store_history(history_file_path)
                    # generate a new config
                    new_cfg = auto_tuner.search_once()
                    cur_cfg = copy.deepcopy(new_cfg)
                    auto_tuner.add_cfg(cur_cfg)
                    continue

            # for single dp estimation and not run sharding overlap
            if tuner_cfg["search_algo"]["name"] != "grid":
                # estimated_num_gpus means need single dp estimation
                bypass_optimizer_flag = "0"
                if (
                    "estimated_num_gpus" in tuner_cfg["search_algo"]
                    and cur_cfg["sharding_degree"] == 1
                ):
                    bypass_optimizer_flag = "1"
                ctx.set_envs(
                    {
                        "FLAGS_shard_bypass_dygraph_optimizer": bypass_optimizer_flag
                    }
                )
            c = controllers.init(ctx)
            c.run()

            task_end_time = time.time()
            cur_cfg["exec_time"] = round(task_end_time - task_start_time, 2)
            ctx.logger.info(
                "Task: job_id {}, log_dir {} ended in {}s".format(
                    task_job_id, log_dir, cur_cfg["exec_time"]
                )
            )
            logger.info(
                "Task: job_id {}, log_dir {} ended in {}s".format(
                    task_job_id, log_dir, cur_cfg["exec_time"]
                )
            )
            # process generated result

            metric, mem, err = read_log(
                path=ctx.args.log_dir,
                metric_file="workerlog.0",
                target_metric=tuner_cfg["metric_cfg"]["name"],
                memory_file=f"{ctx.args.job_id}.gpu.log",
            )
            # sync sigint
            timeout_flag = True
            OOM_flag = err & (1 << 1)
            if actual_nnodes > 1:
                path = f"auto_tuner/{job_id}/{ip}"
                completed = read_completed(ctx.args.log_dir)
                if OOM_flag:
                    while not client.put(path, "OOM".encode('latin-1')):
                        time.sleep(1)
                    ctx.logger.info(f"Put OOM to {path}")
                    logger.info(f"Put OOM to {path}")
                elif completed:
                    while not client.put(path, "OK".encode('latin-1')):
                        time.sleep(1)
                    ctx.logger.info(f"Put OK to {path}")
                    logger.info(f"Put OK to {path}")
                elif hasattr(c, 'sigint') and c.sigint == 14:
                    while not client.put(path, "OK".encode('latin-1')):
                        time.sleep(1)
                    ctx.logger.info(f"Put OK to {path}")
                    logger.info(f"Put OK to {path}")
                elif not hasattr(c, 'sigint') and c.pod.exit_code == 0:
                    while not client.put(path, "OK".encode('latin-1')):
                        time.sleep(1)
                    ctx.logger.info(f"Put OK to {path}")
                    logger.info(f"Put OK to {path}")
                else:
                    while not client.put(path, "Error".encode('latin-1')):
                        time.sleep(1)
                    ctx.logger.info(f"Put Error to {path}")
                    logger.info(f"Put Error to {path}")

                result = list(client.get_prefix(f"auto_tuner/{job_id}/"))
                size = len(result)
                while size != actual_nnodes:
                    time.sleep(1)
                    result = list(client.get_prefix(f"auto_tuner/{job_id}/"))
                    size = len(result)

                status = [i[0].decode() for i in result]
                ctx.logger.info(f"Status of auto_tuner/{job_id}/: {status}")
                logger.info(f"Status of auto_tuner/{job_id}/: {status}")

                if "OOM" in status:
                    timeout_flag = False
                    OOM_flag = True
                elif "OK" not in status:
                    timeout_flag = False

            has_error = False
            if err & (1 << 0):
                ctx.logger.warning(f"Read metric of {log_dir} failed.")
                logger.warning(f"Read metric of {log_dir} failed.")
                # for pruner use
                cur_cfg['time'] = -1
                cur_cfg[tuner_cfg['metric_cfg']['name']] = None
                cur_cfg["max_mem_usage"] = mem if not OOM_flag else "OOM"
                has_error = True

            if err & (1 << 1):
                ctx.logger.warning(f"{log_dir} OOM.")
                logger.warning(f"{log_dir} OOM.")
                # for pruner use
                cur_cfg['time'] = -1
                cur_cfg[tuner_cfg['metric_cfg']['name']] = None
                cur_cfg["max_mem_usage"] = "OOM"
                has_error = True

            # not err & (1 << 1): do not record memory usage when out of memory
            if err & (1 << 2) and not err & (1 << 1):
                ctx.logger.warning(f"Read memory usage of {log_dir} failed.")
                logger.warning(f"Read memory usage of {log_dir} failed.")
                cur_cfg["max_mem_usage"] = None if not OOM_flag else "OOM"

            if not has_error and timeout_flag:
                # for pruner use
                cur_cfg['time'] = metric
                cur_cfg[tuner_cfg['metric_cfg']['name']] = metric
                cur_cfg["max_mem_usage"] = mem if not OOM_flag else "OOM"

            if not has_error and not timeout_flag:
                cur_cfg['time'] = -1
                cur_cfg[tuner_cfg['metric_cfg']['name']] = None
                cur_cfg["max_mem_usage"] = None if not OOM_flag else "OOM"

            if tuner_cfg['metric_cfg']['name'] not in cur_cfg:
                cur_cfg[tuner_cfg['metric_cfg']['name']] = None

            path = f"auto_tuner/mem/{job_id}/{ip}"
            if nnodes > 1:
                while not client.put(
                    path, str(cur_cfg["max_mem_usage"]).encode('latin-1')
                ):
                    time.sleep(1)
                result = list(client.get_prefix(f"auto_tuner/mem/{job_id}"))
                size = len(result)
                while size != nnodes:
                    time.sleep(1)
                    result = list(
                        client.get_prefix(f"auto_tuner/mem/{job_id}/")
                    )
                    size = len(result)
                mem_allnodes = [i[0].decode() for i in result]

                for mem in mem_allnodes:
                    if mem is None:
                        continue
                    if mem == "OOM":
                        cur_cfg["max_mem_usage"] = mem
                        break
                    cur_cfg["max_mem_usage"] = max(
                        int(mem), int(cur_cfg["max_mem_usage"])
                    )

            # if need accurate peak memory
            if os.environ.get("FLAGS_log_memory_stats", False):
                max_peak_memory = None
                from paddle.distributed.auto_tuner.utils import (
                    read_allocated_memory_log,
                )

                for root, dirs, files in os.walk(ctx.args.log_dir):
                    for file in files:
                        if not file.startswith("workerlog"):
                            continue
                        peak_memory = read_allocated_memory_log(
                            ctx.args.log_dir, file
                        )
                        if peak_memory is not None and max_peak_memory is None:
                            max_peak_memory = peak_memory
                        elif peak_memory and max_peak_memory:
                            if peak_memory > max_peak_memory:
                                max_peak_memory = peak_memory
                cur_cfg["max_peak_memory"] = max_peak_memory

            cur_cfg['job_id'] = job_id

            # multi dp conversion
            if (
                "conversion" in tuner_cfg["search_algo"]
                and "step_time" in tuner_cfg["search_algo"]["conversion"]
                and "sharding_overlap" not in cur_cfg
            ):
                single_dp_performance = cur_cfg[tuner_cfg['metric_cfg']['name']]
                step_time_metric = tuner_cfg["search_algo"]["conversion"][
                    "step_time"
                ]
                step_time = read_step_time_log(
                    path=ctx.args.log_dir,
                    file="workerlog.0",
                    target_metric=step_time_metric,
                )

                # set default
                comm_bw = tuner_cfg["search_algo"]["conversion"].get(
                    "comm_bw", [100]
                )
                model_size_b = int(
                    tuner_cfg["search_algo"]["conversion"].get(
                        "model_size_b", 7
                    )
                )
                amp = tuner_cfg["search_algo"]["conversion"].get("amp", False)
                num_gpus = int(cur_cfg["num_gpus"])
                seq_length = int(
                    tuner_cfg["model_cfg"].get("max_seq_length", 2048)
                )
                cur_cfg[f"unified_{tuner_cfg['metric_cfg']['name']}"] = (
                    round(single_dp_performance / num_gpus, 2)
                    if single_dp_performance
                    and tuner_cfg["search_algo"]["conversion"].get(
                        "need_unify", False
                    )
                    else single_dp_performance
                )
                for bw in comm_bw:
                    if amp:
                        comm_time = model_size_b * (4 + 2) / bw
                    else:
                        comm_time = model_size_b * 4 / bw
                    multi_dp_performance = (
                        round(
                            step_time
                            / (step_time + comm_time)
                            * single_dp_performance,
                            5,
                        )
                        if single_dp_performance and step_time
                        else None
                    )
                    cur_cfg[
                        f"bw_{bw}_{tuner_cfg['metric_cfg']['name']}"
                    ] = multi_dp_performance
                    cur_cfg[
                        f"unified_bw_{bw}_{tuner_cfg['metric_cfg']['name']}"
                    ] = (
                        round(multi_dp_performance / num_gpus, 2)
                        if multi_dp_performance
                        and tuner_cfg["search_algo"]["conversion"].get(
                            "need_unify", False
                        )
                        else multi_dp_performance
                    )
                    if recorder.additional_metric_key is None:
                        recorder.additional_metric_key = (
                            f"unified_bw_{bw}_{tuner_cfg['metric_cfg']['name']}"
                        )
                        cur_cfg[
                            "additional_metric_key"
                        ] = recorder.additional_metric_key

            error_info = None
            cur_cfg["has_error"] = has_error
            if has_error:
                error_info = []
                error_task_nums += 1
                if OOM_flag:
                    error_info.append("Out of memory")
                else:
                    if actual_nnodes > 1:
                        path = f"auto_tuner/error/{job_id}/{ip}"
                        single_error_info = find_error_from_log(
                            ctx.args.log_dir
                        )
                        if len(single_error_info) > 0:
                            while not client.put(
                                path,
                                single_error_info.encode('latin-1', 'ignore'),
                            ):
                                time.sleep(1)
                            ctx.logger.info(
                                f"Put Error info: {single_error_info} to {path}"
                            )
                            logger.info(
                                f"Put Error info: {single_error_info} to {path}"
                            )
                        else:
                            while not client.put(path, "OK".encode('latin-1')):
                                time.sleep(1)
                            ctx.logger.info(f"Put OK to {path}")
                            logger.info(f"Put OK to {path}")

                        result = list(
                            client.get_prefix(f"auto_tuner/error/{job_id}/")
                        )
                        size = len(result)
                        while size != actual_nnodes:
                            time.sleep(1)
                            result = list(
                                client.get_prefix(f"auto_tuner/error/{job_id}/")
                            )
                            size = len(result)

                        status = [
                            i[0].decode()
                            for i in result
                            if "OK" not in i[0].decode('utf-8', 'ignore')
                        ]
                        error_info = list(set(status))
                        ctx.logger.info(
                            f"Status of auto_tuner/error/{job_id}/: {error_info}"
                        )
                        logger.info(
                            f"Status of auto_tuner/error/{job_id}/: {error_info}"
                        )
                    else:
                        error_info.append(find_error_from_log(ctx.args.log_dir))
            cur_cfg["error_info"] = error_info
            task_nums = len(auto_tuner.algo.all_tasks)
            cur_task_id = auto_tuner.algo.idx
            ctx.logger.info(
                "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                    cur_task_id,
                    task_nums,
                    cur_task_id - job_id,
                    error_task_nums,
                    error_info,
                    round(
                        (task_nums - cur_task_id) * max_time_per_task / 60,
                        2,
                    ),
                )
            )
            logger.info(
                "Auto Tuner Schedule: [{}/{}], Pruned nums {}, Error nums {}, Error info {}, Remaining time {} min".format(
                    cur_task_id,
                    task_nums,
                    cur_task_id - job_id,
                    error_task_nums,
                    error_info,
                    round(
                        (task_nums - cur_task_id) * max_time_per_task / 60,
                        2,
                    ),
                )
            )

            # sync for single dp
            if sorted_ips:
                master_ip = sorted_ips[0]
                if ip == master_ip:
                    while not client.put(
                        f"auto_tuner/{log_dir}",
                        json.dumps(cur_cfg).encode('latin-1'),
                    ):
                        time.sleep(1)
                    logger.info(f"{ip} put auto_tuner/{log_dir} successfully.")
            recorder.add_cfg(**cur_cfg)
            cur_best_cfgs, err = recorder.get_best(
                metric=tuner_cfg['metric_cfg']['name'],
                direction=tuner_cfg['metric_cfg']['OptimizationDirection'],
                buffer=buffer,
                max_mem_usage=max_mem_usage,
            )
            if not err:
                to_json_str = json.dumps(cur_best_cfgs)
                ctx.logger.info(f"Current best config: {to_json_str}")
                logger.info(f"Current best config: {to_json_str}")
            else:
                ctx.logger.info("Get best config failed, no config can be run.")
                logger.info("Get best config failed, no config can be run.")

            # record history
            if "sharding_overlap" in cur_cfg and cur_cfg["sharding_overlap"]:
                add_overlap_performance(cur_cfg, tuner_cfg, recorder.history)

            recorder.store_history(history_file_path)
            c.finalize(exit=False)

            # generate a new config
            new_cfg = auto_tuner.search_once()
            cur_cfg = copy.deepcopy(new_cfg)
            auto_tuner.add_cfg(cur_cfg)

            # per task launch interval
            self_pid = str(os.getpid())
            if paddle.device.is_compiled_with_custom_device('npu'):
                processes = os.popen(
                    "fuser -v /dev/davinci* |awk '{for(i=1;i<=NF;i++) print $i;}'"
                ).readlines()
            else:
                processes = os.popen(
                    "fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++) print $i;}'"
                ).readlines()
            for process in processes:
                pid = str(process.strip())
                if pid != self_pid:
                    os.system("kill -9 " + pid)
            time.sleep(3)
            end_time = time.time()

            # keep cluster exit consistency
            path = f"auto_tuner/exit/{job_id}/{ip}"
            if max_search_time and (end_time - start_time) > int(
                max_search_time
            ):
                if nnodes > 1:
                    while not client.put(path, "error".encode('latin-1')):
                        time.sleep(1)
                else:
                    break
            else:
                if nnodes > 1:
                    while not client.put(path, "ok".encode('latin-1')):
                        time.sleep(1)

            if nnodes > 1:
                result = list(client.get_prefix(f"auto_tuner/exit/{job_id}"))
                size = len(result)
                while size != nnodes:
                    time.sleep(1)
                    result = list(
                        client.get_prefix(f"auto_tuner/exit/{job_id}/")
                    )
                    size = len(result)
                status = [i[0].decode() for i in result]

                if "error" in status:
                    break

        recorder.store_history(history_file_path)

        # get best config to run
        best_cfg = None
        ctx = copy.deepcopy(raw_ctx)
        if nnodes > 1:
            collective_master_ip = os.environ.get("COLLECTIVE_MASTER_IP", None)
            assert collective_master_ip is not None
            if ip == collective_master_ip:
                best_cfg, err = recorder.get_best(
                    metric=tuner_cfg['metric_cfg']['name'],
                    direction=tuner_cfg['metric_cfg']['OptimizationDirection'],
                    buffer=buffer,
                    max_mem_usage=max_mem_usage,
                )
                if err:
                    raise ValueError(
                        "Get best config failed. Currently there are no appropriate configs."
                    )
                data = json.dumps(best_cfg)
                while not client.put("best_cfg", data):
                    time.sleep(1)
                    continue
            else:
                for i in range(10):
                    try:
                        data = client.get("best_cfg")[0].decode()
                        best_cfg = json.loads(data)
                    except Exception as e:
                        ctx.logger.warning(e)
                        logger.warning(e)
                        time.sleep(2)
                    if best_cfg:
                        break
                assert best_cfg
        else:
            best_cfg, err = recorder.get_best(
                metric=tuner_cfg['metric_cfg']['name'],
                direction=tuner_cfg['metric_cfg']['OptimizationDirection'],
                buffer=buffer,
                max_mem_usage=max_mem_usage,
            )
            if err:
                raise ValueError(
                    "Get best config failed. Currently there are no appropriate configs."
                )
        assert best_cfg and best_cfg["time"] != -1

        end_time = time.time()
        ctx.logger.info(f"AutoTuner ended in {end_time - start_time}s.")
        logger.info(f"AutoTuner ended in {end_time - start_time}s.")
        # launch best cfg
        # estimation search need not run best cfg
        if not tuner_cfg.get("run_best", True) or tuner_cfg["search_algo"].get(
            "estimated_num_gpus", None
        ):
            sys.exit()
        new_args = gen_new_args(raw_args, best_cfg, tuner_cfg, run_best=True)
        ctx.run_best = True
        ctx.args.training_script_args = new_args
        ctx.args.job_id = "best_cfg"
        to_json_str = json.dumps(best_cfg)
        ctx.logger.info(f"Launch best cfg: {to_json_str}")
        logger.info(f"Launch best cfg: {to_json_str}")

        if tuner_cfg.get("best_cfg_dir", None):
            ctx.args.log_dir = tuner_cfg["best_cfg_dir"]
        else:
            ctx.args.log_dir = os.path.join(
                os.path.dirname(ctx.args.auto_tuner_json), "best_cfg"
            )
        # run best cfg
        c = controllers.init(ctx)
        c.run()
        c.finalize(exit=True)

    else:
        from paddle.distributed.launch import controllers

        # initialize the selected controller
        c = controllers.init(ctx)

        # run the pods
        c.run()

        # manager or just wait pod
        c.finalize()


if __name__ == "__main__":
    launch()
