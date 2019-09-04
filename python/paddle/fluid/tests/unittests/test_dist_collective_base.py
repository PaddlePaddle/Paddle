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

from __future__ import print_function
from datetime import datetime

import unittest
import os
import sys
import subprocess
import six
import argparse
import pickle
import numpy as np
import paddle.fluid as fluid

from paddle.fluid.transpiler.collective import GradAllReduce

DEFAULT_BATCH_SIZE = 2
RUN_STEPS = 5


def print2pipe(value):
    if six.PY2:
        print(pickle.dumps(value))
    else:
        sys.stdout.buffer.write(pickle.dumps(value))


def elog(ref, message, to_pipe=False):
    localtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_str = '[%s] [%s] %s' % (localtime, type(ref).__name__, message)
    if to_pipe:
        if six.PY2:
            sys.stderr.write(pickle.dumps(log_str))
        else:
            sys.stderr.buffer.write(pickle.dumps(log_str))
    else:
        print(log_str, file=sys.stderr)


class DistCollectiveRunner(object):
    ##################################
    ##### user specified methods #####

    @classmethod
    def add_arguments(cls, parser):
        pass

    def build_local_net(self):
        raise NotImplementedError(
            'local_net should be implemented by child classes.')

    def build_parall_net(self):
        raise NotImplementedError(
            'parall_net should be implemented by child classes.')

    def yield_sample(self, np_random):
        raise NotImplementedError(
            'data_generator should be implemented by child classes')

    def create_optimizer(self):
        return fluid.optimizer.SGD(learning_rate=0.001)

    def dist_optimize(self, optimizer, loss):
        args = self.args
        optimizer.minimize(loss)
        transpiler = GradAllReduce()
        transpiler.transpile(
            rank=args.rank,
            endpoints=args.endpoints,
            current_endpoint=args.current_endpoint,
            wait_port=True)

    ##### user specified methods #####
    ##################################

    def __init__(self, args):
        self.args = args

    def elog(self, message, to_pipe=False):
        elog(self, message, to_pipe)

    def build_net(self):
        args = self.args
        if args.nranks <= 1:
            elog(self, 'build local network')
            data, loss = self.build_local_net()
        else:
            elog(self, 'build parallel network')
            data, loss = self.build_parall_net()
        return data, loss

    def optimize(self, loss):
        args = self.args
        optimizer = self.create_optimizer()
        if args.nranks <= 1:
            optimizer.minimize(loss)
        else:
            self.dist_optimize(optimizer, loss)

    def get_rank_batch(self):
        args = self.args

        def generate_global_batch():
            if not hasattr(self, 'seed'):
                self.seed = args.batch_size * args.nranks
            np.random.seed(self.seed)
            self.seed += 1

            global_batch_size = args.batch_size * args.nranks
            return [
                next(self.yield_sample(np.random))
                for i in range(global_batch_size)
            ]

        rank_batch = []
        global_batch = generate_global_batch()
        for i, sample in enumerate(global_batch):
            if i // args.batch_size == args.rank:
                rank_batch.append(sample)

        return rank_batch

    def run(self):
        main_prog = fluid.Program()
        start_prog = fluid.Program()
        with fluid.program_guard(main_prog, start_prog):
            data, loss = self.build_net()
            self.optimize(loss)

        place = fluid.CUDAPlace(self.args.device_id)
        exe = fluid.Executor(place)
        exe.run(start_prog)
        elog(self, 'finish running startup program.')

        feeder = fluid.DataFeeder(data, place)

        elog(self, 'start to train')
        out_losses = []
        for i in range(RUN_STEPS):
            losses = exe.run(main_prog,
                             fetch_list=[loss],
                             feed=feeder.feed(self.get_rank_batch()))
            out_losses.append(losses[0][0])
            elog(self, "step %d loss: %f" % (i, losses[0][0]))

        elog(self, 'finish training')

        print2pipe(out_losses)


def runtime_main(test_class):
    parser = argparse.ArgumentParser(
        description='Run distributed classification test.')
    parser.add_argument('--batch_size', type=int, required=True)
    test_class.add_arguments(parser)
    args = parser.parse_args()

    args.rank = int(os.getenv('PADDLE_TRAINER_ID', '0'))
    args.current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    args.nranks = int(os.getenv('PADDLE_TRAINERS_NUM', '1'))
    args.endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS', '').split(',')
    args.device_id = int(os.getenv('FLAGS_selected_gpus', '0'))

    trainer = test_class(args)
    trainer.run()


import socket
from contextlib import closing


class TestDistCollectiveBase(unittest.TestCase):
    ##################################
    ##### user specified methods #####

    # override configurations in setUp
    def update_config(self):
        pass

    def append_common_cmd(self):
        return ''

    def append_local_cmd(self):
        return ''

    def append_parall_cmd(self):
        return ''

    ##### user specified methods #####
    ##################################

    def setUp(self):
        self.nranks = 2
        self.batch_size = DEFAULT_BATCH_SIZE
        self.update_config()

        self.global_batch_size = self.batch_size * self.nranks
        self.endpoints = [
            '127.0.0.1:%d' % self.find_free_port() for i in range(self.nranks)
        ]

    def find_free_port(self):
        while True:
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                elog(self, 'socket port: %s' % s.getsockname()[1])
                port = s.getsockname()[1]
                return port

    def run_local(self, train_script, update_env):
        env = {}
        cmd = '%s -u %s --batch_size %d' % (sys.executable, train_script,
                                            self.global_batch_size)
        if self.append_common_cmd():
            cmd += ' ' + self.append_common_cmd().strip()
        if self.append_local_cmd():
            cmd += ' ' + self.append_local_cmd().strip()

        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            env['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            cmd += ' -m coverage run --branch -p'
        env.update(update_env)

        elog(self, 'local_cmd: %s' % cmd)
        elog(self, 'local_env: %s' % env)

        ferr = open('/tmp/local.log', 'w')
        proc = subprocess.Popen(
            cmd.split(' '), stdout=subprocess.PIPE, stderr=ferr, env=env)

        out, err = proc.communicate()
        ferr.close()

        elog(self, 'local_stdout: %s' % pickle.loads(out))

        return pickle.loads(out)

    def get_parall_env(self, rank):
        env = {
            'FLAGS_selected_gpus': str(rank),
            'PADDLE_TRAINER_ID': str(rank),
            'PADDLE_CURRENT_ENDPOINT': self.endpoints[rank],
            'PADDLE_TRAINERS_NUM': str(self.nranks),
            'PADDLE_TRAINER_ENDPOINTS': ','.join(self.endpoints),
        }
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            env['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
        return env

    def run_parall(self, train_script, update_env):
        cmd = '%s -u %s --batch_size %d' % (sys.executable, train_script,
                                            self.batch_size)
        if self.append_common_cmd():
            cmd += ' ' + self.append_common_cmd().strip()
        if self.append_parall_cmd():
            cmd += ' ' + self.append_parall_cmd().strip()
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd += ' -m coverage run --branch -p'

        procs = []
        ferrs = []
        for rank in range(self.nranks):
            env = self.get_parall_env(rank)
            env.update(update_env)
            elog(self, '[r%d] parall_cmd: %s' % (rank, cmd))
            elog(self, '[r%d] parall_env: %s' % (rank, env))

            ferr = open('/tmp/parall_tr%d.log' % rank, 'w')
            proc = subprocess.Popen(
                cmd.strip().split(' '),
                stdout=subprocess.PIPE,
                stderr=ferr,
                env=env)
            procs.append(proc)
            ferrs.append(ferr)

        outs = []
        for rank in range(self.nranks):
            out, err = procs[rank].communicate()
            ferrs[rank].close()

            outs.append(out)

        return [pickle.loads(outs[i]) for i in range(self.nranks)]

    def compare_parall_to_local(self, train_script, delta=1e-3, update_envs={}):
        required_envs = {
            'PATH': os.getenv('PATH', ''),
            'PYTHONPATH': os.getenv('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''),
            'FLAGS_fraction_of_gpu_memory_to_use': '0.15',
            'FLAGS_rpc_deadline': '5000',  # 5s to fail fast
            'FLAGS_cudnn_deterministic': '1',
            'NCCL_P2P_DISABLE': '1',
            'NCCL_SHM_DISABLE': '1'
        }
        required_envs.update(update_envs)

        local_losses = self.run_local(train_script, required_envs)
        parall_losses = self.run_parall(train_script, required_envs)

        elog(self, '======= local_loss : parall_loss =======')
        for i in range(RUN_STEPS):
            local_loss = local_losses[i]
            parall_loss = sum(
                [parall_losses[j][i] for j in range(self.nranks)]) / self.nranks
            elog(self, '======= %s : %s =======' % (local_loss, parall_loss))
            self.assertAlmostEqual(local_loss, parall_loss, delta=delta)
