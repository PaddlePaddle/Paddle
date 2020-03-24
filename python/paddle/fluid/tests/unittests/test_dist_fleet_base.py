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
"""
    high level unit test for distribute fleet.
"""
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
import tempfile

import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

__all__ = ['FleetDistRunnerBase', 'TestFleetBase', 'runtime_main']

RUN_STEP = 5
LEARNING_RATE = 0.01
DIST_UT_PORT = 0


class FleetDistRunnerBase(object):
    """
        run_pserver,run_trainer : after init role, using transpiler split program
        net : implment by child class, the network of model
        do training : exe run program
    """

    def build_role(self, args):
        if args.role.upper() == "PSERVER":
            role = role_maker.UserDefinedRoleMaker(
                current_id=args.current_id,
                role=role_maker.Role.SERVER,
                worker_num=args.trainers,
                server_endpoints=args.endpoints.split(","))
        else:
            role = role_maker.UserDefinedRoleMaker(
                current_id=args.current_id,
                role=role_maker.Role.WORKER,
                worker_num=args.trainers,
                server_endpoints=args.endpoints.split(","))
        return role

    def build_strategy(self, args):
        self.strategy = None
        if args.mode == "async":
            self.strategy = StrategyFactory.create_async_strategy()
        elif args.mode == "sync":
            self.strategy = StrategyFactory.create_sync_strategy()
        elif args.mode == "half_async":
            self.strategy = StrategyFactory.create_half_async_strategy()
        elif args.mode == "geo":
            self.strategy = StrategyFactory.create_geo_strategy(
                args.geo_sgd_need_push_nums)
        self.dump_param = os.getenv("dump_param", "").split(",")
        self.dump_fields = os.getenv("dump_fields", "").split(",")
        self.dump_fields_path = os.getenv("dump_fields_path", "")
        debug = int(os.getenv("Debug", "0"))
        if debug:
            self.strategy.set_debug_opt({
                "dump_param": self.dump_param,
                "dump_fields": self.dump_fields,
                "dump_fields_path": self.dump_fields_path
            })

        return self.strategy

    def build_optimizer(self, avg_cost, strategy):
        use_grad_clip = int(os.getenv('GRAD_CLIP', 0))
        if use_grad_clip:
            # 1: clip_by_value; 2: clip_by_norm; 3:clip_by_global_norm
            if use_grad_clip == 1:
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByValue(2.0))
            elif use_grad_clip == 2:
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByNorm(2.0))
            elif use_grad_clip == 3:
                fluid.clip.set_gradient_clip(
                    clip=fluid.clip.GradientClipByGlobalNorm(2.0))

        optimizer = fluid.optimizer.SGD(LEARNING_RATE)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

    def run_pserver(self, args):
        fleet.init(self.build_role(args))
        strategy = self.build_strategy(args)
        avg_cost = self.net(args)
        self.build_optimizer(avg_cost, strategy)

        fleet.init_server()
        fleet.run_server()

    def run_dataset_trainer(self, args):
        fleet.init(self.build_role(args))
        strategy = self.build_strategy(args)
        avg_cost = self.net(args)
        self.build_optimizer(avg_cost, strategy)
        out = self.do_dataset_training(fleet)

    def run_pyreader_trainer(self, args):
        fleet.init(self.build_role(args))
        strategy = self.build_strategy(args)
        avg_cost = self.net(args)
        self.build_optimizer(avg_cost, strategy)
        out = self.do_pyreader_training(fleet)

    def net(self, args, batch_size=4, lr=0.01):
        raise NotImplementedError(
            "get_model should be implemented by child classes.")

    def do_dataset_training(self, fleet):
        raise NotImplementedError(
            "do_dataset_training should be implemented by child classes.")

    def do_pyreader_training(self, fleet):
        raise NotImplementedError(
            "do_pyreader_training should be implemented by child classes.")


class TestFleetBase(unittest.TestCase):
    """
        start_pserver,start_trainer : add start cmd to test
        run_cluster : using multi process to test distribute program
    """

    def _setup_config(self):
        raise NotImplementedError("tests should have _setup_config implemented")

    def setUp(self):
        self._mode = "sync"
        self._reader = "pyreader"
        self._trainers = 2
        self._pservers = 2
        self._port_set = set()

        global DIST_UT_PORT
        if DIST_UT_PORT == 0 and os.getenv("PADDLE_DIST_UT_PORT"):
            DIST_UT_PORT = int(os.getenv("PADDLE_DIST_UT_PORT"))

        if DIST_UT_PORT:
            print("set begin_port:", DIST_UT_PORT)
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                DIST_UT_PORT, DIST_UT_PORT + 1)
            DIST_UT_PORT += 2
        else:
            self._ps_endpoints = "127.0.0.1:%s,127.0.0.1:%s" % (
                self._find_free_port(), self._find_free_port())

        self._python_interp = sys.executable
        self._geo_sgd_need_push_nums = 5
        self._grad_clip_mode = 0
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

        ps0_pipe = open(tempfile.gettempdir() + "/ps0_err.log", "wb+")
        ps1_pipe = open(tempfile.gettempdir() + "/ps1_err.log", "wb+")

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

        tr0_pipe = open(tempfile.gettempdir() + "/tr0_err.log", "wb+")
        tr1_pipe = open(tempfile.gettempdir() + "/tr1_err.log", "wb+")

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
        env = {'GRAD_CLIP': str(self._grad_clip_mode)}
        env.update(envs)

        python_path = self._python_interp

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            envs['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            python_path += " -m coverage run --branch -p"

        tr_cmd = "{0} {1} --role trainer --endpoints {2} --current_id {{}} --trainers {3} --mode {4} --geo_sgd_need_push_nums {5} --reader {6}".format(
            python_path, model, self._ps_endpoints, self._trainers, self._mode,
            self._geo_sgd_need_push_nums, self._reader)

        ps_cmd = "{0} {1} --role pserver --endpoints {2} --current_id {{}} --trainers {3} --mode {4} --geo_sgd_need_push_nums {5} --reader {6}".format(
            python_path, model, self._ps_endpoints, self._trainers, self._mode,
            self._geo_sgd_need_push_nums, self._reader)

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
    parser.add_argument('--mode', type=str, required=False, default='geo')
    parser.add_argument(
        '--geo_sgd_need_push_nums', type=int, required=False, default=2)
    parser.add_argument('--reader', type=str, required=False, default='dataset')
    args = parser.parse_args()

    model = test_class()
    if args.role == "pserver":
        model.run_pserver(args)
    else:
        if args.reader == "dataset":
            model.run_dataset_trainer(args)
        else:
            model.run_pyreader_trainer(args)
