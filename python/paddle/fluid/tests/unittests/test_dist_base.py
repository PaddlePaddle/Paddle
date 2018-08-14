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
import time

import unittest
import os
import sys
import six
import signal
import subprocess
import six


class TestDistRunnerBase(object):
    def get_model(self, batch_size=2):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def get_transpiler(self, trainer_id, main_program, pserver_endpoints,
                       trainers):
        # NOTE: import fluid until runtime, or else forking processes will cause error.
        import paddle
        import paddle.fluid as fluid
        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id=trainer_id,
            program=main_program,
            pservers=pserver_endpoints,
            trainers=trainers)
        return t

    def run_pserver(self, pserver_endpoints, trainers, current_endpoint,
                    trainer_id):
        import paddle
        import paddle.fluid as fluid
        self.get_model(batch_size=2)
        t = self.get_transpiler(trainer_id,
                                fluid.default_main_program(), pserver_endpoints,
                                trainers)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self, place, endpoints, trainer_id, trainers, is_dist=True):
        import paddle
        import paddle.fluid as fluid
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
        self.get_model(batch_size=2)
        if is_dist:
            t = self.get_transpiler(trainer_id,
                                    fluid.default_main_program(), endpoints,
                                    trainers)
            trainer_prog = t.get_trainer_program()
        else:
            trainer_prog = fluid.default_main_program()

        startup_exe = fluid.Executor(place)
        startup_exe.run(fluid.default_startup_program())

        strategy = fluid.ExecutionStrategy()
        strategy.num_threads = 1
        strategy.allow_op_delay = False
        exe = fluid.ParallelExecutor(
            True, loss_name=avg_cost.name, exec_strategy=strategy)

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

    if len(sys.argv) != 7:
        print(
            "Usage: python dist_se_resnext.py [pserver/trainer] [endpoints] [trainer_id] [current_endpoint] [trainers] [is_dist]"
        )
    role = sys.argv[1]
    endpoints = sys.argv[2]
    trainer_id = int(sys.argv[3])
    current_endpoint = sys.argv[4]
    trainers = int(sys.argv[5])
    is_dist = True if sys.argv[6] == "TRUE" else False

    model = test_class()
    if role == "pserver":
        model.run_pserver(endpoints, trainers, current_endpoint, trainer_id)
    else:
        p = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        model.run_trainer(p, endpoints, trainer_id, trainers, is_dist)


import paddle.fluid.compat as cpt


class TestDistBase(unittest.TestCase):
    def setUp(self):
        self._trainers = 2
        self._pservers = 2
        self._ps_endpoints = "127.0.0.1:9123,127.0.0.1:9124"
        self._python_interp = "python"

    def start_pserver(self, model_file):
        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        ps0_cmd = "%s %s pserver %s 0 %s %d TRUE" % \
            (self._python_interp, model_file, self._ps_endpoints, ps0_ep,
             self._trainers)
        ps1_cmd = "%s %s pserver %s 0 %s %d TRUE" % \
            (self._python_interp, model_file, self._ps_endpoints, ps1_ep,
             self._trainers)

        ps0_proc = subprocess.Popen(
            ps0_cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ps1_proc = subprocess.Popen(
            ps1_cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ps0_proc, ps1_proc

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

    def check_with_place(self, model_file, delta=1e-3):
        # *ATTENTION* THIS TEST NEEDS AT LEAST 2GPUS TO RUN
        required_envs = {
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_cudnn_deterministic": "1"
        }
        # Run local to get a base line
        env_local = {"CUDA_VISIBLE_DEVICES": "0"}
        env_local.update(required_envs)
        local_cmd = "%s %s trainer %s 0 %s %d FLASE" % \
            (self._python_interp, model_file,
             "127.0.0.1:1234", "127.0.0.1:1234", 1)
        local_proc = subprocess.Popen(
            local_cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env_local)
        local_proc.wait()
        out, err = local_proc.communicate()
        local_ret = cpt.to_literal_str(out)
        sys.stderr.write('local_loss: %s\n' % local_ret)
        sys.stderr.write('local_stderr: %s\n' % err)

        # Run dist train to compare with local results
        ps0, ps1 = self.start_pserver(model_file)
        self._wait_ps_ready(ps0.pid)
        self._wait_ps_ready(ps1.pid)

        ps0_ep, ps1_ep = self._ps_endpoints.split(",")
        tr0_cmd = "%s %s trainer %s 0 %s %d TRUE" % \
            (self._python_interp, model_file, self._ps_endpoints, ps0_ep,
             self._trainers)
        tr1_cmd = "%s %s trainer %s 1 %s %d TRUE" % \
            (self._python_interp, model_file, self._ps_endpoints, ps1_ep,
             self._trainers)

        env0 = {"CUDA_VISIBLE_DEVICES": "0"}
        env1 = {"CUDA_VISIBLE_DEVICES": "1"}
        env0.update(required_envs)
        env1.update(required_envs)
        FNULL = open(os.devnull, 'w')

        tr0_proc = subprocess.Popen(
            tr0_cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env0)
        tr1_proc = subprocess.Popen(
            tr1_cmd.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env1)

        tr0_proc.wait()
        tr1_proc.wait()
        out, err = tr0_proc.communicate()
        sys.stderr.write('dist_stderr: %s\n' % err)
        loss_data0 = cpt.to_literal_str(out)
        sys.stderr.write('dist_loss: %s\n' % loss_data0)
        lines = loss_data0.split("\n")
        dist_first_loss = eval(lines[0].replace(" ", ","))[0]
        dist_last_loss = eval(lines[1].replace(" ", ","))[0]

        local_lines = local_ret.split("\n")
        local_first_loss = eval(local_lines[0])[0]
        local_last_loss = eval(local_lines[1])[0]

        # FIXME: use terminate() instead of sigkill.
        os.kill(ps0.pid, signal.SIGKILL)
        os.kill(ps1.pid, signal.SIGKILL)
        FNULL.close()

        self.assertAlmostEqual(local_first_loss, dist_first_loss, delta=delta)
        self.assertAlmostEqual(local_last_loss, dist_last_loss, delta=delta)
