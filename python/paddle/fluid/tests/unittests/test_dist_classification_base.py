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

from paddle.fluid.transpiler.collective import \
    GradAllReduce, DistributedClassificationOptimizer

DEFAULT_BATCH_SIZE = 2
DEFAULT_FEATURE_SIZE = 4
DEFAULT_CLASS_NUM = 4
DEFAULT_LR = 0.001

RUN_STEPS = 5


def stdprint(value):
    if six.PY2:
        print(pickle.dumps(value))
    else:
        sys.stdout.buffer.write(pickle.dumps(value))


def log(ref, message, print2pipe=False):
    localtime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_str = '[%s] [%s] %s' % (localtime, type(ref).__name__, message)
    if print2pipe:
        if six.PY2:
            sys.stderr.write(pickle.dumps(log_str))
        else:
            sys.stderr.buffer.write(pickle.dumps(log_str))
    else:
        sys.stderr.write(log_str + "\n")


class DistClassificationRunner(object):
    def __init__(self, args):
        args.rank = int(os.getenv('PADDLE_TRAINER_ID', '0'))
        args.current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
        args.nranks = int(os.getenv('PADDLE_TRAINERS_NUM', '1'))
        args.endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS', '').split(',')
        args.device_id = int(os.getenv('FLAGS_selected_gpus', '0'))
        self.args = args

    def log(self, message, print2pipe=False):
        log(self, message, print2pipe)

    def local_classify_subnet(self, feature, label):
        raise NotImplementedError(
            'get_local_model should be implemented by child classes.')

    def parall_classify_subnet(self, feature, label):
        raise NotImplementedError(
            'get_parall_model should be implemented by child classes.')

    def build_net(self):
        args = self.args
        main_prog = fluid.Program()
        start_prog = fluid.Program()
        optimizer = fluid.optimizer.SGD(learning_rate=args.lr)
        with fluid.program_guard(main_prog, start_prog):
            feature = fluid.layers.data(
                name='feature', shape=[args.feature_size], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            if args.nranks <= 1:
                log(self, 'build local network')
                loss = self.local_classify_subnet(feature, label)
                optimizer.minimize(loss)
            else:
                log(self, 'build parallel network')
                loss = self.parall_classify_subnet(feature, label)
                # TODO why need batch size?
                optimizer_wrapper = DistributedClassificationOptimizer(
                    optimizer, args.batch_size)
                optimizer_wrapper.minimize(loss)
                self.transpile(main_prog, start_prog)

        return [feature, label], loss, start_prog

    def gen_rank_batch(self):
        args = self.args

        def generate_global_batch():
            if not hasattr(self, 'seed'):
                self.seed = args.batch_size * args.nranks
            np.random.seed(self.seed)
            self.seed += 1

            global_batch_size = args.batch_size * args.nranks
            return [[
                np.random.rand(args.feature_size),
                np.random.randint(args.class_num)
            ] for i in range(global_batch_size)]

        rank_batch = []
        global_batch = generate_global_batch()
        for i, sample in enumerate(global_batch):
            if i // args.batch_size == args.rank:
                rank_batch.append(sample)

        log(self, rank_batch)

        return rank_batch

    def transpile(self, main_prog, start_prog):
        args = self.args
        transpiler = GradAllReduce()
        transpiler.transpile(
            startup_program=start_prog,
            main_program=main_prog,
            rank=args.rank,
            endpoints=args.endpoints,
            current_endpoint=args.current_endpoint,
            wait_port=True)

    def run(self):
        feed_vars, loss, start_prog = self.build_net()
        main_prog = loss.block.program

        place = fluid.CUDAPlace(self.args.device_id)
        exe = fluid.Executor(place)
        exe.run(start_prog)
        log(self, 'finish running startup program.')

        feeder = fluid.DataFeeder(feed_vars, place)

        log(self, 'start to train')
        out_losses = []
        for i in range(RUN_STEPS):
            losses = exe.run(main_prog,
                             fetch_list=[loss],
                             feed=feeder.feed(self.gen_rank_batch()))
            out_losses.append(losses[0][0])
            log(self, "step %d loss: %f" % (i, losses[0][0]))

        log(self, 'finish training')

        stdprint(out_losses)

    @classmethod
    def add_arguments(cls, parser):
        pass


def runtime_main(test_class):
    parser = argparse.ArgumentParser(
        description='Run distributed classification test.')
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument(
        '--feature_size', type=int, default=DEFAULT_FEATURE_SIZE)
    parser.add_argument('--class_num', type=int, default=DEFAULT_CLASS_NUM)
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    test_class.add_arguments(parser)
    args = parser.parse_args()

    trainer = test_class(args)
    trainer.run()


import socket
from contextlib import closing


class TestDistClassificationBase(unittest.TestCase):
    # override configurations in setUp
    def setup_config(self):
        raise NotImplementedError('tests should have setup_config implemented')

    def setUp(self):
        self.nranks = 2
        self.batch_size = DEFAULT_BATCH_SIZE
        self.setup_config()

        self.global_batch_size = self.batch_size * self.nranks
        self.endpoints = [
            '127.0.0.1:%d' % self.find_free_port() for i in range(self.nranks)
        ]

    def find_free_port(self):
        while True:
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(('', 0))
                log(self, 'socket port: %s' % s.getsockname()[1])
                port = s.getsockname()[1]
                return port

    def run_local(self, train_script, user_env):
        env = {}
        cmd = '%s -u %s --batch_size %d' % (sys.executable, train_script,
                                            self.global_batch_size)
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            env['COVERAGE_FILE'] = os.getenv('COVERAGE_FILE', '')
            cmd += ' -m coverage run --branch -p'
        env.update(user_env)

        log(self, 'local_cmd: %s' % cmd)
        log(self, 'local_env: %s' % env)

        ferr = open('/tmp/local.log', 'w')
        proc = subprocess.Popen(
            cmd.split(' '),
            stdout=subprocess.PIPE,
            #stderr=subprocess.PIPE,
            stderr=ferr,
            env=env)

        out, err = proc.communicate()
        ferr.close()

        log(self, 'local_stdout: %s' % pickle.loads(out))
        #log(self, 'local_stderr: %s' % pickle.loads(err))

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

    def run_parall(self, train_script, user_env):
        cmd = '%s -u %s --batch_size %d' % (sys.executable, train_script,
                                            self.batch_size)
        if os.getenv('WITH_COVERAGE', 'OFF') == 'ON':
            cmd += ' -m coverage run --branch -p'

        procs = []
        ferrs = []
        for rank in range(self.nranks):
            env = self.get_parall_env(rank)
            env.update(user_env)
            log(self, '[r%d] parall_cmd: %s' % (rank, cmd))
            log(self, '[r%d] parall_env: %s' % (rank, env))

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
            #log(self, '[r%d] parall_stderr: %s' % (rank, pickle.loads(err)))

        return [pickle.loads(outs[i]) for i in range(self.nranks)]

    def compare_parall_to_local(self, train_script, delta=1e-3, user_envs={}):
        required_envs = {
            'PATH': os.getenv('PATH', ''),
            'PYTHONPATH': os.getenv('PYTHONPATH', ''),
            'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH', ''),
            'FLAGS_fraction_of_gpu_memory_to_use': '0.15',
            'FLAGS_rpc_deadline': '30000',  # 5s to fail fast
            'FLAGS_cudnn_deterministic': '1',
            'NCCL_P2P_DISABLE': '1',
            'NCCL_SHM_DISABLE': '1'
        }
        required_envs.update(user_envs)

        local_losses = self.run_local(train_script, required_envs)
        parall_losses = self.run_parall(train_script, required_envs)

        for i in range(RUN_STEPS):
            local_loss = local_losses[i]
            parall_loss = sum(
                [parall_losses[j][i] for j in range(self.nranks)]) / self.nranks
            log(self, '======= local_loss : parall_loss =======')
            log(self, '======= %s : %s =======' % (local_loss, parall_loss))
            self.assertAlmostEqual(local_loss, parall_loss, delta=delta)
