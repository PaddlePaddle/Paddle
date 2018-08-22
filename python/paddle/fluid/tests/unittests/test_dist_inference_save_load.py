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
import unittest
from test_dist_base import TestDistBase

import os
import sys
import signal
import subprocess
import six
import paddle.compat as cpt


class TestDistInferenceSaveAndLoad2x2(TestDistBase):
    def _setup_config(self):
        self._sync_mode = True

    @staticmethod
    def _save_model(dirname, feeded_var_names, target_vars):
        import paddle.fluid as fluid

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        fluid.io.save_inference_model(dirname, feeded_var_names, target_vars,
                                      exe)

    def run_pserver(self,
                    pserver_endpoints,
                    trainers,
                    current_endpoint,
                    trainer_id,
                    sync_mode=True):
        import paddle
        import paddle.fluid as fluid
        self.get_model(batch_size=2)
        t = self.get_transpiler(trainer_id,
                                fluid.default_main_program(), pserver_endpoints,
                                trainers, sync_mode)
        pserver_prog = t.get_pserver_program(current_endpoint)
        startup_prog = t.get_startup_program(current_endpoint, pserver_prog)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        exe.run(pserver_prog)

    def run_trainer(self,
                    place,
                    endpoints,
                    trainer_id,
                    trainers,
                    is_dist=True,
                    sync_mode=True):
        import paddle
        import paddle.fluid as fluid
        test_program, avg_cost, train_reader, test_reader, batch_acc, predict = \
            self.get_model(batch_size=2)
        if is_dist:
            t = self.get_transpiler(trainer_id,
                                    fluid.default_main_program(), endpoints,
                                    trainers, sync_mode)
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

        dirname = "/tmp/simnet.infer.model"
        if trainer_id == 0:
            self._save_model(dirname, [], [predict])

    def check_with_save_inference(self,
                                  model_file,
                                  delta=1e-3,
                                  check_error_log=False):
        # *ATTENTION* THIS TEST NEEDS AT LEAST 2GPUS TO RUN
        required_envs = {
            "PATH": os.getenv("PATH"),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_cudnn_deterministic": "1"
        }

        if check_error_log:
            required_envs["GLOG_v"] = "7"
            required_envs["GLOG_logtostderr"] = "1"

        # Run local to get a base line
        env_local = {"CUDA_VISIBLE_DEVICES": "0"}
        env_local.update(required_envs)
        sync_mode_str = "TRUE" if self._sync_mode else "FALSE"
        local_cmd = "%s %s trainer %s 0 %s %d FLASE %s" % \
                    (self._python_interp, model_file,
                     "127.0.0.1:1234", "127.0.0.1:1234", 1, sync_mode_str)
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
        tr0_cmd = "%s %s trainer %s 0 %s %d TRUE %s" % \
                  (self._python_interp, model_file, self._ps_endpoints, ps0_ep,
                   self._trainers, sync_mode_str)
        tr1_cmd = "%s %s trainer %s 1 %s %d TRUE %s" % \
                  (self._python_interp, model_file, self._ps_endpoints, ps1_ep,
                   self._trainers, sync_mode_str)

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
        FNULL.close()

        self.assertAlmostEqual(local_first_loss, dist_first_loss, delta=delta)
        self.assertAlmostEqual(local_last_loss, dist_last_loss, delta=delta)

    @unittest.skip(reason="Not Ready, Debugging")
    def test_dist_save_inference_model(self):
        self.check_with_save_inference("dist_simnet_bow.py", delta=1e-7)


if __name__ == "__main__":
    unittest.main()
