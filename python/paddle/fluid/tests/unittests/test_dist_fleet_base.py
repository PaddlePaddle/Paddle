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

import argparse
import os
import pickle
import subprocess
import sys
import time
import traceback
import math
import collections
import socket
from contextlib import closing

import six
import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig

RUN_STEP = 5
LEARNING_RATE = 0.01


class FleetDistRunnerBase(object):
    def run_pserver(self, args):
        if args.role.upper() != "PSERVER":
            raise ValueError("args role must be PSERVER")

        role = role_maker.UserDefinedRoleMaker(
            current_id=args.current_id,
            role=role_maker.Role.SERVER,
            worker_num=args.trainers,
            server_endpoints=args.endpoints.split(","))

        fleet.init(role)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = args.sync_mode

        avg_cost = self.net()

        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        fleet.init_server()
        fleet.run_server()

    def run_trainer(self, args):
        if args.role.upper() != "TRAINER":
            raise ValueError("args role must be TRAINER")

        role = role_maker.UserDefinedRoleMaker(
            current_id=args.current_id,
            role=role_maker.Role.WORKER,
            worker_num=args.trainers,
            server_endpoints=args.endpoints.split(","))

        fleet.init(role)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = args.sync_mode

        avg_cost = self.net()

        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        self.do_training(fleet)
        out = self.do_training(fleet)

    def net(self, batch_size=4, lr=0.01):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def do_training(self, fleet):
        raise NotImplementedError(
            "do_training should be implemented by child classes.")


class TestFleetBase(unittest.TestCase):
    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def setUp(self):
        self._sync_mode = True
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()
        self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
            self._find_free_port(), self._find_free_port())
        self._python_interp = sys.executable
        self._setup_config()

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

    def _start_pserver(self, cmd, required_envs):
        ps0_cmd, ps1_cmd = cmd.format(0), cmd.format(1)

        ps0_pipe = open("/tmp/ps0_err.log", "wb+")
        ps1_pipe = open("/tmp/ps1_err.log", "wb+")

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

    def _start_trainer(self, cmd, required_envs):
        tr0_cmd, tr1_cmd = cmd.format(0), cmd.format(1)

        tr0_pipe = open("/tmp/tr0_err.log", "wb+")
        tr1_pipe = open("/tmp/tr1_err.log", "wb+")

        tr0_proc = subprocess.Popen(
            tr0_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr0_pipe,
            env=required_envs)
        tr1_proc = subprocess.Popen(
            tr1_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=tr1_pipe,
            env=required_envs)

        return tr0_proc, tr1_proc, tr0_pipe, tr1_pipe

    def _run_cluster(self, model, envs):
        env = {'CPU_NUM': '1'}
        env.update(envs)

        tr_cmd = "{0} {1} --role trainer --endpoints {2} --current_id {{}} --trainers {3}".format(
            self._python_interp, model, self._ps_endpoints, self._trainers)

        ps_cmd = "{0} {1} --role pserver --endpoints {2} --current_id {{}} --trainers {3}".format(
            self._python_interp, model, self._ps_endpoints, self._trainers)

        if self._sync_mode:
            tr_cmd += " --sync_mode"
            ps_cmd += " --sync_mode"

        # Run dist train to compare with local results
        ps0, ps1, ps0_pipe, ps1_pipe = self._start_pserver(ps_cmd, env)
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

        # close trainer file
        tr0_pipe.close()
        tr1_pipe.close()
        ps0_pipe.close()
        ps1_pipe.close()

        ps0.terminate()
        ps1.terminate()
        '''
        with open("/tmp/tr0_out.log", "wb+") as wn:
            wn.write(tr0_out)
        with open("/tmp/tr1_out.log", "wb+") as wn:
            wn.write(tr1_out)
        # print server log
        '''

        # print server log
        '''
        with open("/tmp/ps0_err.log", "r") as fn:
            sys.stderr.write("ps0 stderr: %s\n" % fn.read())
        with open("/tmp/ps1_err.log", "r") as fn:
            sys.stderr.write("ps1 stderr: %s\n" % fn.read())
        '''

        # print log
        '''
        with open("/tmp/tr0_err.log", "r") as fn:
            sys.stderr.write('trainer 0 stderr: %s\n' % fn.read())
        with open("/tmp/tr1_err.log", "r") as fn:
            sys.stderr.write('trainer 1 stderr: %s\n' % fn.read())
        '''

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
            "http_proxy": ""
        }

        required_envs.update(need_envs)

        if check_error_log:
            required_envs["GLOG_v"] = "3"
            required_envs["GLOG_logtostderr"] = "1"

        tr0_losses, tr1_losses = self._run_cluster(model_file, required_envs)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(description='Run Fleet test.')
    parser.add_argument(
        '--role', type=str, required=True, choices=['pserver', 'trainer'])
    parser.add_argument('--endpoints', type=str, required=False, default="")
    parser.add_argument('--current_id', type=int, required=False, default=0)
    parser.add_argument('--trainers', type=int, required=False, default=1)
    parser.add_argument('--sync_mode', action='store_true')

    args = parser.parse_args()

    model = test_class()
    if args.role == "pserver":
        model.run_pserver(args)
    else:
        model.run_trainer(args)
