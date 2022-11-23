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

import sys

sys.path.append("..")
import unittest
from test_dist_base import TestDistBase
import paddle.fluid as fluid

import os
import subprocess
import pickle

DEFAULT_BATCH_SIZE = 2

flag_name = os.path.splitext(__file__)[0]

print("file: {}".format(flag_name))


class TestParallelDygraphMnistMLU(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False
        self._cncl_mode = True
        self._dygraph = True
        self._enforce_place = "MLU"

    def _get_required_envs(self, check_error_log=False, need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "LD_PRELOAD": os.getenv("LD_PRELOAD", ""),
            "FLAGS_fraction_of_gpu_memory_to_use": "0.15",
            "FLAGS_eager_delete_tensor_gb": "0.0",
            "FLAGS_call_stack_level": "2",
            "GLOG_v": "2",
            "PADDLE_WITH_GLOO": '0',
            "BACKEND": "cncl"
        }

        if check_error_log:
            required_envs["GLOG_v"] = "5"
            required_envs["GLOG_logtostderr"] = "1"
            required_envs["GLOO_LOG_LEVEL"] = "TRACE"

        required_envs.update(need_envs)
        return required_envs

    def _run_local(self,
                   model,
                   envs,
                   check_error_log=False,
                   batch_size=DEFAULT_BATCH_SIZE,
                   batch_merge_repeat=1,
                   log_name="",
                   devices="1"):

        cmd = self._python_interp

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            cmd += " -m coverage run --branch -p"

        cmd += " %s --role trainer --update_method local --lr %f" % (model,
                                                                     self._lr)

        if batch_size != DEFAULT_BATCH_SIZE:
            cmd += " --batch_size %d" % batch_size
        if batch_merge_repeat > 1:
            cmd += " --batch_merge_repeat %d" % batch_merge_repeat
        if self._nccl2_reduce_layer:
            cmd += " --nccl2_reduce_layer_local_run 1"

        if self._use_mlu:
            cmd += " --use_mlu"
            env_local = {
                "FLAGS_selected_mlus": devices,
                "PADDLE_TRAINERS_NUM": "1",
                "PADDLE_TRAINER_ID": "0"
            }
        else:
            env_local = {'CPU_NUM': '1'}

        # not use dgc in single card
        if len(devices) > 1 and self._use_dgc:
            cmd += " --use_dgc"

        if self._accumulate_gradient:
            cmd += " --accumulate_gradient"

        if self._find_unused_parameters:
            cmd += " --find_unused_parameters"

        env_local.update(envs)
        print("local_cmd: {}, env: {}".format(cmd, env_local))

        if check_error_log:
            path = "/tmp/local_err_%d.log" % os.getpid()
            err_log = open(path, "w")
            local_proc = subprocess.Popen(cmd.split(" "),
                                          stdout=subprocess.PIPE,
                                          stderr=err_log,
                                          env=env_local)
        else:
            local_proc = subprocess.Popen(cmd.split(" "),
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          env=env_local)

        local_out, local_err = local_proc.communicate()

        if check_error_log:
            err_log.close()
            sys.stderr.write(
                '\n--run_local-- trainer 0 stderr file saved in: %s\n' % (path))

        sys.stderr.write('local_stderr: %s\n' % local_err)
        sys.stderr.write('local_stdout: %s\n' % pickle.loads(local_out))

        return pickle.loads(local_out)

    def _run_cluster_nccl2(self, model, envs, update_method, check_error_log,
                           log_name):
        # NOTE: we reuse ps_endpoints as nccl2 worker endpoints
        worker_endpoints = self._ps_endpoints.split(",")

        trainer_num = len(worker_endpoints)

        procs = []
        pipes = []
        for i in range(0, trainer_num):
            tr_cmd, tr_env = self._get_nccl2_trainer_cmd(
                model, worker_endpoints[i], update_method, i, trainer_num)
            tr_env.update(envs)
            print("use_hallreduce:{} \ntr{}_cmd:{}, env: {}".format(
                self._use_hallreduce, i, tr_cmd, tr_env))

            tr_pipe = open("/tmp/tr%d_err_%d.log" % (i, os.getpid()), "w")

            sys.stderr.write(
                "\n{} going to start process {} with nccl2\n".format(
                    type(self).__name__, i))
            tr_proc = subprocess.Popen(tr_cmd.strip().split(" "),
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
            sys.stderr.write('trainer {} stderr: {}\n'.format(i, tr_err))
            sys.stderr.write(
                'trainer {} glog file saved in: /tmp/tr{}_err_{}.log \n'.format(
                    i, i, os.getpid()))

        if check_error_log:
            print("outs[0]:", pickle.loads(outs[0]))
            print("outs[1]:", pickle.loads(outs[1]))

        return pickle.loads(outs[0]), pickle.loads(outs[1])

    def test_mnist(self):
        if fluid.core.is_compiled_with_mlu():
            self.check_with_place(
                os.path.abspath("parallel_dygraph_sync_batch_norm.py"),
                delta=1e-5,
                check_error_log=True,
                log_name=flag_name)


if __name__ == "__main__":
    unittest.main()
