#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid

RUN_STEP = 10
DEFAULT_BATCH_SIZE = 2


class TestDistRunnerBase(object):
    def get_model(self, batch_size=DEFAULT_BATCH_SIZE, lr=0.1):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    @staticmethod
    def get_transpiler(trainer_id,
                       main_program,
                       pserver_endpoints,
                       trainers,
                       sync_mode,
                       dc_asgd=False):
        # NOTE: import fluid until runtime, or else forking processes will cause error.
        config = fluid.DistributeTranspilerConfig()
        config.enable_dc_asgd = dc_asgd
        t = fluid.DistributeTranspiler(config=config)
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=sync_mode)
        return t

    def run_pserver(self, args):
        self.lr = args.lr
        self.get_model(batch_size=args.batch_size)
        # NOTE: pserver should not call memory optimize
        t = self.get_transpiler(args.trainer_id,
                                fluid.default_main_program(), args.endpoints,
                                args.trainers, args.sync_mode, args.dc_asgd)
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(args.current_endpoint,
                                             pserver_prog)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self, args):
        self.lr = args.lr
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
            self.get_model(batch_size=args.batch_size)

        if args.mem_opt:
            fluid.memory_optimize(fluid.default_main_program(), skip_grads=True)
        if args.update_method == "pserver":
            t = self.get_transpiler(args.trainer_id,
                                    fluid.default_main_program(),
                                    args.endpoints, args.trainers,
                                    args.sync_mode, args.dc_asgd)
            trainer_prog = t.get_trainer_program()
        elif args.update_method == "nccl2":
            # transpile for nccl2
            config = fluid.DistributeTranspilerConfig()
            config.mode = "nccl2"
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
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1
        strategy.allow_op_delay = False

        build_stra = fluid.BuildStrategy()

        if args.use_reduce:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        else:
            build_stra.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.AllReduce

        if args.batch_merge_repeat > 1:
            pass_builder = build_stra._finalize_strategy_and_create_passes()
            mypass = pass_builder.insert_pass(
                len(pass_builder.all_passes()) - 2, "multi_batch_merge_pass")
            mypass.set_int("num_repeats", args.batch_merge_repeat)

        if args.update_method == "nccl2":
            num_trainers = len(args.endpoints.split(","))
            trainer_id = args.trainer_id
        else:
            num_trainers = 1
            trainer_id = 0

        exe = fluid.ParallelExecutor(
            args.use_cuda,
            loss_name=avg_cost.name,
            exec_strategy=strategy,
            build_strategy=build_stra,
            num_trainers=num_trainers,
            trainer_id=trainer_id)

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
        for _ in six.moves.xrange(RUN_STEP):
            loss, = exe.run(fetch_list=[avg_cost.name],
                            feed=feeder.feed(get_data()))
            out_losses.append(loss[0])
        if six.PY2:
            print(pickle.dumps(out_losses))
        else:
            sys.stdout.buffer.write(pickle.dumps(out_losses))


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run dist test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument(
        '--update_method',
        type=str,
        default="local",
        choices=["pserver", "nccl2", "local"])
    parser.add_argument('--trainer_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument(
        '--current_endpoint', type=str, required=False, default="")
    parser.add_argument('--sync_mode', action='store_true')
    parser.add_argument('--mem_opt', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_reduce', action='store_true')
    parser.add_argument('--dc_asgd', action='store_true')
    parser.add_argument(
        '--use_reader_alloc', action='store_true', required=False)
    parser.add_argument('--batch_size', required=False, type=int, default=2)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument(
        '--batch_merge_repeat', required=False, type=int, default=1)

    args = parser.parse_args()

    model = test_class()
    if args.role == "pserver" and args.update_method == "pserver":
        model.run_pserver(args)
    else:
        model.run_trainer(args)


import paddle.compat as cpt
import socket
from contextlib import closing


class TestDistBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def _after_setup_config(self):
        if self._enforce_place == "CPU":
            self.__use_cuda = False
        elif self._enforce_place == "GPU":
            self.__use_cuda = True
        else:
            if fluid.core.is_compiled_with_cuda():
                self.__use_cuda = True
            else:
                self.__use_cuda = False

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable
        self._sync_mode = True
        self._enforce_place = None
        self._mem_opt = False
        self._use_reduce = False
        self._dc_asgd = False  # must use with async mode
        self._use_reader_alloc = True
        self._nccl2_mode = False
        self._lr = 0.001
        self._setup_config()
        self._after_setup_config()

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

    def start_pserver(self, model_file, check_error_log, required_envs):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps_cmd = "%s %s --role pserver --endpoints %s --trainer_id 0 --current_endpoint %s --trainers %d --update_method pserver"
        ps0_cmd = ps_cmd % \
                  (self._python_interp, model_file, self._ps_endpoints, ps0_ep,
                   self._trainers)
        ps1_cmd = ps_cmd % \
                  (self._python_interp, model_file, self._ps_endpoints, ps1_ep,
                   self._trainers)

        if self._sync_mode:
            ps0_cmd += " --sync_mode"
            ps1_cmd += " --sync_mode"
        if self._mem_opt:
            ps0_cmd += " --mem_opt"
            ps1_cmd += " --mem_opt"

        print(ps0_cmd)
        print(ps1_cmd)
        ps0_pipe = open("/tmp/ps0_err.log", "wb")
        ps1_pipe = open("/tmp/ps1_err.log", "wb")

        ps0_proc = subprocess.Popen(
            ps0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps0_pipe,
            env=required_envs)
        ps1_proc = subprocess.Popen(
            ps1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=ps1_pipe,
            env=required_envs)

        return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _run_local(self,
                   model,
                   envs,
                   check_error_log=False,
                   batch_size=DEFAULT_BATCH_SIZE,
                   batch_merge_repeat=1):

        cmd = "%s %s --role trainer --lr %f" % (self._python_interp, model,
                                                self._lr)
        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size
        if batch_merge_repeat > 1:
            cmd += " --batch_merge_repeat %d" % batch_merge_repeat

        if self.__use_cuda:
            cmd += " --use_cuda"
            env_local = {"CUDA_VISIBLE_DEVICES": "0"}
        else:
            env_local = {'CPU_NUM': '1'}

        env_local.update(envs)
        print("local_cmd: {}, env: {}".format(cmd, env_local))

        if check_error_log:
            err_log = open("/tmp/trainer.err.log", "wb")
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

        sys.stderr.write('local_stderr: %s\n' % local_err)
        sys.stderr.write('local_stdout: %s\n' % pickle.loads(local_out))

        return pickle.loads(local_out)

    def _run_cluster(self, model, envs, check_error_log):
        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self.start_pserver(model,
                                                          check_error_log, envs)

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")

        tr_cmd = "%s %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --trainers %d --update_method pserver --lr %f"
        tr0_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   0, ps0_ep, self._trainers, self._lr)
        tr1_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   1, ps1_ep, self._trainers, self._lr)

        if self._sync_mode:
            tr0_cmd += " --sync_mode"
            tr1_cmd += " --sync_mode"
        if self._mem_opt:
            tr0_cmd += " --mem_opt"
            tr1_cmd += " --mem_opt"
        if self._use_reduce:
            tr0_cmd += " --use_reduce"
            tr1_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr0_cmd += " --use_reader_alloc"
            tr1_cmd += " --use_reader_alloc"
        if self.__use_cuda:
            tr0_cmd += " --use_cuda"
            tr1_cmd += " --use_cuda"
            env0 = {"CUDA_VISIBLE_DEVICES": "0"}
            env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        else:
            env0 = {'CPU_NUM': '1'}
            env1 = {'CPU_NUM': '1'}

        env0.update(envs)
        env1.update(envs)

        print("tr0_cmd: {}, env: {}".format(tr0_cmd, env0))
        print("tr1_cmd: {}, env: {}".format(tr1_cmd, env1))
        tr0_pipe = open("/tmp/tr0_err.log", "wb")
        tr1_pipe = open("/tmp/tr1_err.log", "wb")

        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        # Wait until trainer process terminate
        while True:
            stat0 = tr0_proc.poll()
            time.sleep(0.1)
            if stat0 is not None:
                break
        while True:
            stat1 = tr1_proc.poll()
            time.sleep(0.1)
            if stat1 is not None:
                break

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()

        ps0.terminate()
        ps1.terminate()

        # print server log
        with open("/tmp/ps0_err.log", "r") as fn:
            sys.stderr.write("ps0 stderr: %s\n" % fn.read())
        with open("/tmp/ps1_err.log", "r") as fn:
            sys.stderr.write("ps1 stderr: %s\n" % fn.read())

        # print log
        if stat0 == 0:
            sys.stderr.write('trainer 0 stdout: %s\n' % pickle.loads(tr0_out))
        with open("/tmp/tr0_err.log", "r") as fn:
            sys.stderr.write('trainer 0 stderr: %s\n' % fn.read())
        if stat1 == 0:
            sys.stderr.write('trainer 1 stdout: %s\n' % pickle.loads(tr1_out))
        with open("/tmp/tr1_err.log", "r") as fn:
            sys.stderr.write('trainer 1 stderr: %s\n' % fn.read())

        return pickle.loads(tr0_out), pickle.loads(tr1_out)

    def _run_cluster_nccl2(self, model, envs, check_error_log):
        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")
        w0_ep, w1_ep = worker_endpoints

        tr_cmd = "%s %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --update_method nccl2 --lr %f"
        tr0_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   0, w0_ep, self._lr)
        tr1_cmd = tr_cmd % \
                  (self._python_interp, model, self._ps_endpoints,
                   1, w1_ep, self._lr)

        if self._mem_opt:
            tr0_cmd += " --mem_opt"
            tr1_cmd += " --mem_opt"
        if self._use_reduce:
            tr0_cmd += " --use_reduce"
            tr1_cmd += " --use_reduce"
        if self._use_reader_alloc:
            tr0_cmd += " --use_reader_alloc"
            tr1_cmd += " --use_reader_alloc"
        if self.__use_cuda:
            tr0_cmd += " --use_cuda"
            tr1_cmd += " --use_cuda"
            env0 = {"CUDA_VISIBLE_DEVICES": "0"}
            env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        else:
            env0 = {'CPU_NUM': '1'}
            env1 = {'CPU_NUM': '1'}

        env0.update(envs)
        env1.update(envs)

        print("tr0_cmd:{}, env: {}".format(tr0_cmd, env0))
        print("tr1_cmd:{}, env: {}".format(tr1_cmd, env1))
        tr0_pipe = open("/tmp/tr0_err.log", "wb")
        tr1_pipe = open("/tmp/tr1_err.log", "wb")

        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        tr0_out, tr0_err = tr0_proc.communicate()
        tr1_out, tr1_err = tr1_proc.communicate()

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()

        # print log
        sys.stderr.write('trainer 0 stderr: %s\n' % tr0_err)
        sys.stderr.write('trainer 1 stderr: %s\n' % tr1_err)
        sys.stderr.write('trainer 0 stdout: %s\n' % tr0_out)
        sys.stderr.write('trainer 1 stdout: %s\n' % tr1_out)

        return pickle.loads(tr0_out), pickle.loads(tr1_out)

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
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "FLAGS_cudnn_deterministic": "1",
            "http_proxy": "",
            "NCCL_P2P_DISABLE": "1"
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        local_losses\
            = self._run_local(model_file, required_envs,
                                       check_error_log)
        if self._nccl2_mode:
            tr0_losses, tr1_losses = self._run_cluster_nccl2(
                model_file, required_envs, check_error_log)
        else:
            tr0_losses, tr1_losses = self._run_cluster(
                model_file, required_envs, check_error_log)

        for step_id in range(RUN_STEP):
            local_loss = local_losses[step_id]
            tr0_loss = tr0_losses[step_id]
            tr1_loss = tr1_losses[step_id]
            dist_loss = (np.array([tr0_loss]) + np.array([tr1_loss])) / 2
            print("=======", local_loss, ":", dist_loss[0], "=======")
            self.assertAlmostEqual(local_loss, dist_loss[0], delta=delta)
