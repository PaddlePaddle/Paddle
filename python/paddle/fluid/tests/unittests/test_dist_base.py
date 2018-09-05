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
import six
import signal
import subprocess
import argparse


class TestDistRunnerBase(object):
    def get_model(self, batch_size=2):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def get_transpiler(self, trainer_id, main_program, pserver_endpoints,
                       trainers, sync_mode):
        # NOTE: import fluid until runtime, or else forking processes will cause error.
        import paddle
        import paddle.fluid as fluid
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers,
            sync_mode=sync_mode)
        return t

    def run_pserver(self, args):
        import paddle
        import paddle.fluid as fluid
        self.get_model(batch_size=2)
        t = self.get_transpiler(args.trainer_id,
                                fluid.default_main_program(), args.endpoints,
                                args.trainers, args.sync_mode)
        pserver_prog = t.get_pserver_program(args.current_endpoint)
        startup_prog = t.get_startup_program(args.current_endpoint,
                                             pserver_prog)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self, place, args):
        import paddle
        import paddle.fluid as fluid
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
        self.get_model(batch_size=2)
        if args.is_dist:
            t = self.get_transpiler(args.trainer_id,
                                    fluid.default_main_program(),
                                    args.endpoints, args.trainers,
                                    args.sync_mode)
            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

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

        exe = fluid.ParallelExecutor(
            True,
            loss_name=avg_cost.name,
            exec_strategy=strategy,
            build_strategy=build_stra)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = test_reader()

        data = next(reader_generator)
        first_loss, = exe.run(fetch_list=[avg_cost.name],
                              feed=feeder.feed(data))
        print(first_loss)

        for i in six.moves.xrange(5):
            data = next(reader_generator)
            loss, = exe.run(fetch_list=[avg_cost.name], feed=feeder.feed(data))

        data = next(reader_generator)
        last_loss, = exe.run(fetch_list=[avg_cost.name], feed=feeder.feed(data))
        print(last_loss)


def runtime_main(test_class):
    import paddle
    import paddle.fluid as fluid
    import paddle.fluid.core as core

    parser = argparse.ArgumentParser(description='Run dist test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument('--is_dist', action='store_true')
    parser.add_argument('--trainer_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument(
        '--current_endpoint', type=str, required=False, default="")
    parser.add_argument('--sync_mode', action='store_true')
    parser.add_argument('--mem_opt', action='store_true')
    parser.add_argument('--use_reduce', action='store_true')

    args = parser.parse_args()

    model = test_class()
    if args.role == "pserver":
        model.run_pserver(args)
    else:
        p = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        model.run_trainer(p, args)


import paddle.compat as cpt


class TestDistBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._ps_endpoints = "127.0.0.1:9123,127.0.0.1:9124"
        self._python_interp = "python"
        self._sync_mode = True
        self._mem_opt = False
        self._use_reduce = False
        self._setup_config()

    def start_pserver(self, model_file, check_error_log):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps_cmd = "%s %s --role pserver --endpoints %s --trainer_id 0 --current_endpoint %s --trainers %d --is_dist"
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

        ps0_pipe = subprocess.PIPE
        ps1_pipe = subprocess.PIPE
        if check_error_log:
            print("ps0_cmd:", ps0_cmd)
            print("ps1_cmd:", ps1_cmd)
            ps0_pipe = open("/tmp/ps0_err.log", "wb")
            ps1_pipe = open("/tmp/ps1_err.log", "wb")

        ps0_proc = subprocess.Popen(
            ps0_cmd.split(" "), stdout=subprocess.PIPE, stderr=ps0_pipe)
        ps1_proc = subprocess.Popen(
            ps1_cmd.split(" "), stdout=subprocess.PIPE, stderr=ps1_pipe)

        if not check_error_log:
            return ps0_proc, ps1_proc, None, None
        else:
            return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _wait_ps_ready(self, pid):
        retry_times = 50
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(3)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error as e:
                sys.stderr.write('waiting for pserver: %s, left retry %d\n' %
                                 (e, retry_times))
                retry_times -= 1

    def check_with_place(self, model_file, delta=1e-3, check_error_log=False):
        # *ATTENTION* THIS TEST NEEDS AT LEAST 2GPUS TO RUN
        required_envs = {
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_cudnn_deterministic": "1"
        }

        if check_error_log:
            required_envs["GLOG_v"] = "7"
            required_envs["GLOG_logtostderr"] = "1"

        # Run local to get a base line
        env_local = {"CUDA_VISIBLE_DEVICES": "0"}
        env_local.update(required_envs)
        local_cmd = "%s %s --role trainer" % (self._python_interp, model_file)
        if not check_error_log:
            local_proc = subprocess.Popen(
                local_cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_local)
        else:
            print("trainer cmd:", local_cmd)
            err_log = open("/tmp/trainer.err.log", "wb")
            local_proc = subprocess.Popen(
                local_cmd.split(" "),
                stdout=subprocess.PIPE,
                stderr=err_log,
                env=env_local)

        local_proc.wait()
        out, err = local_proc.communicate()
        local_ret = cpt.to_text(out)
        sys.stderr.write('local_loss: %s\n' % local_ret)
        sys.stderr.write('local_stderr: %s\n' % err)

        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self.start_pserver(model_file,
                                                          check_error_log)
        self._wait_ps_ready(ps0.pid)
        self._wait_ps_ready(ps1.pid)

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        tr_cmd = "%s %s --role trainer --endpoints %s --trainer_id %d --current_endpoint %s --trainers %d --is_dist"
        tr0_cmd = tr_cmd % \
            (self._python_interp, model_file, self._ps_endpoints,
             0, ps0_ep, self._trainers)
        tr1_cmd = tr_cmd % \
            (self._python_interp, model_file, self._ps_endpoints,
             1, ps1_ep, self._trainers)

        if self._sync_mode:
            tr0_cmd += " --sync_mode"
            tr1_cmd += " --sync_mode"
        if self._mem_opt:
            tr0_cmd += " --mem_opt"
            tr1_cmd += " --mem_opt"
        if self._use_reduce:
            tr0_cmd += " --use_reduce"
            tr1_cmd += " --use_reduce"

        env0 = {"CUDA_VISIBLE_DEVICES": "0"}
        env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        env0.update(required_envs)
        env1.update(required_envs)
        FNULL = open(os.devnull, 'w')

        tr0_pipe = subprocess.PIPE
        tr1_pipe = subprocess.PIPE
        if check_error_log:
            print("tr0_cmd:", tr0_cmd)
            print("tr1_cmd:", tr1_cmd)
            tr0_pipe = open("/tmp/tr0_err.log", "wb")
            tr1_pipe = open("/tmp/tr1_err.log", "wb")

        tr0_proc = subprocess.Popen(
            tr0_cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=env0)
        tr1_proc = subprocess.Popen(
            tr1_cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=env1)

        tr0_proc.wait()
        tr1_proc.wait()
        out, err = tr0_proc.communicate()
        sys.stderr.write('dist_stderr: %s\n' % err)
        loss_data0 = cpt.to_text(out)
        sys.stderr.write('dist_loss: %s\n' % loss_data0)
        lines = loss_data0.split("\n")
        dist_first_loss = eval(lines[0].replace(" ", ","))[0]
        dist_last_loss = eval(lines[1].replace(" ", ","))[0]

        local_lines = local_ret.split("\n")
        local_first_loss = eval(local_lines[0])[0]
        local_last_loss = eval(local_lines[1])[0]

        # close trainer file
        if check_error_log:
            tr0_pipe.close()
            tr1_pipe.close()

            ps0_pipe.close()
            ps1_pipe.close()
        # FIXME: use terminate() instead of sigkill.
        os.kill(ps0.pid, signal.SIGKILL)
        os.kill(ps1.pid, signal.SIGKILL)
        ps0.terminate()
        ps1.terminate()
        ps0.wait()
        ps1.wait()
        FNULL.close()

        self.assertAlmostEqual(local_first_loss, dist_first_loss, delta=delta)
        self.assertAlmostEqual(local_last_loss, dist_last_loss, delta=delta)
