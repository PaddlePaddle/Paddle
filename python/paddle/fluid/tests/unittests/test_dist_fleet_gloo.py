# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
import tempfile
import unittest
import subprocess
import time
import paddle.fluid as fluid
#import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import paddle.distributed.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory
from test_dist_fleet_base import TestFleetBase

#from dist_simnet_bow import train_network


class TestDistGloo_2x2(TestFleetBase):

    def _setup_config(self):
        self._mode = "sync"
        self._reader = "pyreader"
        self._path = "./tmp4"
        if (os.path.exists(self._path)):
            shutil.rmtree(self._path)
        # if not os.path.exists(self._path):
        #      os.mkdir(self._path)

    def _start_pserver(self, cmd, required_envs):
        #env.update(required_envs)
        ps0_cmd = cmd
        ps1_cmd = cmd

        ps0_pipe = open(tempfile.gettempdir() + "/ps0_err.log", "wb+")
        ps1_pipe = open(tempfile.gettempdir() + "/ps1_err.log", "wb+")

        required_envs["POD_IP"] = "127.0.0.1"
        required_envs["PADDLE_PSERVER_ID"] = "0"
        required_envs["PADDLE_PORT"] = "36011"
        ps0_proc = subprocess.Popen(ps0_cmd.strip().split(" "),
                                    stdout=subprocess.PIPE,
                                    stderr=ps0_pipe,
                                    env=required_envs)
        print("PADDLE_PSERVER_ID=0:")
        print(required_envs)
        required_envs["PADDLE_PSERVER_ID"] = "1"
        required_envs["PADDLE_PORT"] = "36012"
        ps1_proc = subprocess.Popen(ps1_cmd.strip().split(" "),
                                    stdout=subprocess.PIPE,
                                    stderr=ps1_pipe,
                                    env=required_envs)
        print("PADDLE_PSERVER_ID=1:")
        print(required_envs)
        return ps0_proc, ps1_proc, ps0_pipe, ps1_pipe

    def _start_trainer(self, cmd, required_envs):
        #env.update(required_envs)

        tr0_cmd = cmd
        tr1_cmd = cmd

        tr0_pipe = open(tempfile.gettempdir() + "/tr0_err.log", "wb+")
        tr1_pipe = open(tempfile.gettempdir() + "/tr1_err.log", "wb+")
        required_envs["PADDLE_TRAINER_ID"] = "0"
        tr0_proc = subprocess.Popen(tr0_cmd.strip().split(" "),
                                    stdout=subprocess.PIPE,
                                    stderr=tr0_pipe,
                                    env=required_envs)
        print("PADDLE_TRAINER_ID=0:")
        print(required_envs)
        required_envs["PADDLE_TRAINER_ID"] = "1"
        tr1_proc = subprocess.Popen(tr1_cmd.strip().split(" "),
                                    stdout=subprocess.PIPE,
                                    stderr=tr1_pipe,
                                    env=required_envs)
        print("PADDLE_TRAINER_ID=1:")
        print(required_envs)
        return tr0_proc, tr1_proc, tr0_pipe, tr1_pipe

    def _run_cluster(self, model, envs):
        env = {'GRAD_CLIP': str(self._grad_clip_mode)}
        python_path = self._python_interp
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            python_path += " -m coverage run --branch -p"
        env.update(envs)

        tr_cmd = "{0} {1}".format(python_path, model)

        ps_cmd = "{0} {1}".format(python_path, model)

        # Run dist train to compare with local results
        env["TRAINING_ROLE"] = "PSERVER"
        ps0, ps1, ps0_pipe, ps1_pipe = self._start_pserver(ps_cmd, env)
        print(ps_cmd)
        env["TRAINING_ROLE"] = "TRAINER"
        tr0, tr1, tr0_pipe, tr1_pipe = self._start_trainer(tr_cmd, env)

        # Wait until trainer process terminate
        while True:
            stat0 = tr0.poll()
            time.sleep(0.1)
            if stat0 is not None:
                break

        while True:
            stat1 = tr1.poll()
            time.sleep(0.1)
            if stat1 is not None:
                break

        tr0_out, tr0_err = tr0.communicate()
        tr1_out, tr1_err = tr1.communicate()

        tr0_ret = tr0.returncode
        tr1_ret = tr0.returncode

        self.assertEqual(tr0_ret, 0, "something wrong in tr0, please check")
        self.assertEqual(tr1_ret, 0, "something wrong in tr1, please check")

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()

        ps0.terminate()
        ps1.terminate()

        return 0, 0

    def check_with_place(self,
                         model_file,
                         delta=1e-3,
                         check_error_log=False,
                         need_envs={}):
        required_envs = {
            "PATH": os.getenv("PATH", ""),
            "PYTHONPATH": os.getenv("PYTHONPATH", ""),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", ""),
            "FLAGS_rpc_deadline": "5000",  # 5sec to fail fast
            "http_proxy": "",
            "CPU_NUM": "2",
            #PSERVER
            "PADDLE_PSERVERS_IP_PORT_LIST": "127.0.0.1:36011,127.0.0.1:36012",
            #"PADDLE_PSERVER_PORT_ARRAY":"(36011 36012)",
            "PADDLE_PSERVER_NUMS": "2",
            "PADDLE_TRAINER_ID": "0",
            #TRAINER
            "PADDLE_TRAINER_ENDPOINTS": "127.0.0.1:36013,127.0.0.1:36014",
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_PSERVER_ID": "0",
            #GLOO FLAG
            "PADDLE_WITH_GLOO": "1",
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)

    def test_dist_train(self):
        print("path is not delete", os.path.exists("./tmp4"))
        self.check_with_place("dist_fleet_debug_gloo.py",
                              delta=1e-5,
                              check_error_log=True)


if __name__ == "__main__":
    unittest.main()
