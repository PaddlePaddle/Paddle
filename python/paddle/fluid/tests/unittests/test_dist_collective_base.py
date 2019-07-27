#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import time

import unittest
import os
import sys
import signal
import subprocess
import six
import argparse
import pickle
import numpy as np
import time
import traceback
import paddle.fluid as fluid
from paddle.fluid import compiler

# dump loss value into files in each trainer
# when process ends, load pick files, and compare loss with local training

RUN_STEP = 5
DEFAULT_BATCH_SIZE = 4


class TestDistRunnerBase(object):
    def get_model(self,
                  batch_size=DEFAULT_BATCH_SIZE,
                  lr=0.1,
                  single_device=False,
                  use_dgc=False):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    @staticmethod
    def get_transpiler(trainer_id,
                       main_program,
                       pserver_endpoints,
                       trainers,
                       sync_mode,
                       dc_asgd=False,
                       current_endpoint=None,
                       nccl_comm_num=1):
        # NOTE: import fluid until runtime, or else forking processes will cause error.
        config = fluid.DistributeTranspilerConfig()
        config.enable_dc_asgd = dc_asgd
        config.sync_mode = sync_mode
        if nccl_comm_num > 1:
            config.nccl_comm_num = nccl_comm_num
        # config.runtime_split_send_recv = True
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers,
            current_endpoint=current_endpoint)
        return t

    def run_trainer(self, args):
        self.lr = args.lr
        if args.nccl2_reduce_layer_local_run:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size, single_device=True)
        elif args.use_dgc:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size, use_dgc=args.use_dgc)
        else:
            test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
                self.get_model(batch_size=args.batch_size)

        if args.update_method == "nccl2":
            # transpile for nccl2
            config = fluid.DistributeTranspilerConfig()
            config.mode = "nccl2"
            config.nccl_comm_num = args.nccl_comm_num
            if args.use_hallreduce:
                config.use_hierarchical_allreduce = True
                config.hierarchical_allreduce_inter_nranks = args.hallreduce_inter_nranks
            nccl2_t = fluid.DistributeTranspiler(config=config)
            nccl2_t.transpile(
                args.trainer_id,
                program=fluid.default_main_program(),
                startup_program=fluid.default_startup_program(),
                trainers=args.endpoints,
                current_endpoint=args.current_endpoint)
            trainer_prog = fluid.default_main_program()
        else:
            trainer_prog = fluid.default_main_program()
        if args.use_cuda:
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = fluid.CUDAPlace(device_id)
        else:
            place = fluid.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.allow_op_delay = False

        build_stra = fluid.BuildStrategy()
        # FIXME force disable enable_inplace and memory_optimize
        build_stra.enable_inplace = False
        build_stra.memory_optimize = False

        if args.enable_backward_deps:
            build_stra.enable_backward_optimizer_op_deps = True

        if args.use_reduce:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce

        pass_builder = None
        if args.batch_merge_repeat > 1:
            pass_builder = build_stra._finalize_strategy_and_create_passes()
            mypass = pass_builder.insert_pass(0, "multi_batch_merge_pass")
            mypass.set("num_repeats", args.batch_merge_repeat)

        if args.update_method == "nccl2" or args.update_method == "nccl2_reduce_layer":
            build_stra.num_trainers = len(args.endpoints.split(","))
            build_stra.trainer_id = args.trainer_id
        else:
            # case args.update_method == "nccl2_reduce_layer":
            build_stra.num_trainers = 1
            build_stra.trainer_id = 0

        binary = compiler.CompiledProgram(trainer_prog).with_data_parallel(
            loss_name=avg_cost.name,
            build_strategy=build_stra,
            exec_strategy=exec_strategy)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        out_losses = []
        for i in six.moves.xrange(RUN_STEP):
            loss, = exe.run(binary,
                            fetch_list=[avg_cost.name],
                            feed=feeder.feed(get_data()))
            out_losses.append(loss[0])

        result_dict = {}
        result_dict["loss"] = out_losses
        with open(self.pick_filename, "wb") as fout:
            pickle.dump(result_dict, fout)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run dist test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--update_method',
        type=str,
        default="local",
        choices=["pserver", "nccl2", "local", "nccl2_reduce_layer"])
    parser.add_argument('--pick_filename', type=str, required=True)
    parser.add_argument('--trainer_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--nccl_comm_num', type=int, required=False, default=1)
    parser.add_argument('--enable_backward_deps', action='store_true')
    parser.add_argument('--use_hallreduce', action='store_true')
    parser.add_argument(
        '--hallreduce_inter_nranks', type=int, required=False, default=2)
    parser.add_argument(
        '--current_endpoint', type=str, required=False, default="")
    parser.add_argument('--sync_mode', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_dgc', action='store_true')
    parser.add_argument('--use_reduce', action='store_true')
    parser.add_argument('--dc_asgd', action='store_true')
    parser.add_argument(
        '--use_reader_alloc', action='store_true', required=False)
    parser.add_argument('--batch_size', required=False, type=int, default=2)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument(
        '--batch_merge_repeat', required=False, type=int, default=1)
    parser.add_argument(
        '--nccl2_reduce_layer_local_run',
        required=False,
        type=bool,
        default=False)

    args = parser.parse_args()

    model = test_class()
    model.pick_filename = args.pick_filename
    model.run_trainer(args)


import paddle.compat as cpt
import socket
from contextlib import closing


class TestDistCollectiveBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def _after_setup_config(self):
        if self._enforce_place == "CPU":
            self.__use_cuda = False
            self._use_dgc = False
        elif self._enforce_place == "GPU":
            self.__use_cuda = True
        else:
            if fluid.core.is_compiled_with_cuda():
                self.__use_cuda = True
            else:
                self.__use_cuda = False
                self._use_dgc = False

        if self._use_reduce:
            assert not self._use_dgc

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable
        self._sync_mode = True
        self._enforce_place = None
        self._use_reduce = False
        self._dc_asgd = False  # must use with async mode
        self._use_reader_alloc = True
        self._nccl2_mode = False
        self._mp_mode = False
        # FIXME(typhoonzero): I added this stupid argument to enable
        # testing allreduce layers, which users can call layers.allreduce
        # to accumulate tensors at anywhere. Find a better way to do this
        # test, reduce check this argument everywhere.
        self._nccl2_reduce_layer = False
        self._lr = 0.001
        self._use_dgc = False
        self._dygraph = False
        self._nccl_comm_num = 1
        self._enable_backward_deps = False
        self._use_hallreduce = False
        self._setup_config()
        self._after_setup_config()
        self.pick_filename = ""

    def _find_free_port(self):
        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def _run_local(self,
                   model,
                   envs,
                   check_error_log=False,
                   batch_size=DEFAULT_BATCH_SIZE,
                   batch_merge_repeat=1):

        self.pick_filename = "local_run.pkl"
        cmd = "%s %s --role trainer --lr %f --pick_filename %s" % \
              (self._python_interp, model,
               self._lr, self.pick_filename)
        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size
        if batch_merge_repeat > 1:
            cmd += " --batch_merge_repeat %d" % batch_merge_repeat
        if self._nccl2_reduce_layer:
            cmd += " --nccl2_reduce_layer_local_run 1"

        if self.__use_cuda:
            cmd += " --use_cuda"
            env_local = {
                "CUDA_VISIBLE_DEVICES": "0",
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0"
            }
        else:
            env_local = {'CPU_NUM': '1'}

        env_local.update(envs)
        #print("local_cmd: {}, env: {}".format(cmd, env_local))

        if check_error_log:
            err_log = open("trainer.err.log", "wb")
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=err_log,
                env=env_local)
        else:
            local_proc = subprocess.Popen(
                cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_local)

        local_out, local_err = local_proc.communicate()

        if check_error_log:
            err_log.close()

        return self.pick_filename

    def _get_nccl2_trainer_cmd(self, model, ep, update_method, pickle_filename,
                               trainer_id, trainer_num):
        self.pick_filename = pickle_filename
        env = {}
        tr_cmd = "%s -u %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method %s --lr %f --pick_filename %s"
        tr_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   trainer_id, ep, update_method, self._lr, pickle_filename)

        if self._use_reduce:
            tr_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr_cmd += " --use_reader_alloc"
        if self.__use_cuda:
            tr_cmd += " --use_cuda"
            env.update({
                "CUDA_VISIBLE_DEVICES": "{}".format(trainer_id),
                "PADDLE_TRAINERS_NUM": "{}".format(trainer_num),
                "PADDLE_TRAINER_ID": "{}".format(trainer_id)
            })
        else:
            env.update({'CPU_NUM': '1'})

        if self._use_dgc:
            tr_cmd += " --use_dgc"

        if self._mp_mode:
            env = {"FLAGS_selected_gpus": "{}".format(trainer_id)}

        if self._nccl_comm_num > 1:
            tr_cmd += " --nccl_comm_num {}".format(self._nccl_comm_num)

        if self._use_hallreduce:
            tr_cmd += " --use_hallreduce --hallreduce_inter_nranks 2"

        if self._enable_backward_deps:
            tr_cmd += " --enable_backward_deps"

        return tr_cmd, env

    def _run_collective(self, model, envs, check_error_log):
        if self._use_hallreduce:
            self._ps_endpoints = ""
            for i in range(0, 4):
                self._ps_endpoints += "127.0.0.1:%s," % (self._find_free_port())
            self._ps_endpoints = self._ps_endpoints[:-1]

        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")
        update_method = "nccl2"

        trainer_num = len(worker_endpoints)
        pickle_filenames = ["trainer%d.pkl" % x for x in range(trainer_num)]

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, pickle_filenames[i],
                i, trainer_num)
            tr_env.update(envs)
            #print("use_hallreduce:{} tr_cmd:{}, env: {}".format(
            #self._use_hallreduce, tr_cmd, tr_env))

            tr_pipe = open("tr{}_err.log".format(i), "wb")

            tr_proc = subprocess.Popen(
                tr_cmd.strip().split(" "),
                stdout=subprocess.PIPE,
                stderr=tr_pipe,
                env=tr_env)

            procs.append(tr_proc)
            pipes.append(tr_pipe)

        outs = []
        for i in range(0, trainer_num):
            tr_out, tr_err = procs[i].communicate()
            outs.append(tr_out)
            pipes[i].close()

        return pickle_filenames

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        # TODO(typhoonzero): should auto adapt GPU count on the machine.
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_rpc_deadline": "30000",  # 5sec to fail fast
            "FLAGS_cudnn_deterministic": "1",
            "FLAGS_eager_delete_tensor_gb": "0.0",  # add gc for all test case
            "http_proxy": "",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "1"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "10"
            required_envs["GLOG_logtostderr"] = "1"

        local_picklefile = self._run_local(model_file, required_envs,
                                           check_error_log)
        cluster_picklefiles = self._run_collective(model_file, required_envs,
                                                   check_error_log)

        with open(local_picklefile, "rb") as fin:
            local_result_dict = pickle.load(fin)

        cluster_trainer_result_dicts = []

        for pickle_file in cluster_picklefiles:
            with open(pickle_file, "rb") as fin:
                cluster_trainer_result_dicts.append(pickle.load(fin))

        for step_id in range(RUN_STEP):
            local_loss = local_result_dict["loss"][step_id]
            cluster_losses = [
                x["loss"][step_id] for x in cluster_trainer_result_dicts
            ]
            dist_loss = np.mean(cluster_losses)
            print("=======", local_loss, ":", dist_loss, "=======")
            self.assertAlmostEqual(local_loss, dist_loss, delta=delta)
