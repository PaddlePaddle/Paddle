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

import unittest
import subprocess
import sys, os
import json
import shutil

import random

from os import listdir
from os.path import isfile, join

pyname = 'train.py'
colpyfile = '''# train.py for unitest
import os
env = os.environ.copy()
assert "PADDLE_MASTER" in env
assert "PADDLE_GLOBAL_SIZE" in env
assert "PADDLE_LOCAL_SIZE" in env
assert "PADDLE_GLOBAL_RANK" in env
assert "PADDLE_LOCAL_RANK" in env
'''

pspyfile = '''# train.py for unitest
import os
env = os.environ.copy()
assert "PADDLE_PSERVERS_IP_PORT_LIST" in env
assert "PADDLE_TRAINER_ENDPOINTS" in env
assert "PADDLE_ROLE" in env
#assert "PADDLE_RANK" in env
'''


def write_file(name, ct):
    with open(name, "w") as f:
        f.write(ct)


def get_files(pth, prefix):
    return [
        f for f in listdir(pth) if isfile(join(pth, f)) and f.startswith(prefix)
    ]


class Collective_Test(unittest.TestCase):
    def setUp(self):
        write_file(pyname, colpyfile)

    def pdrun(self, args, env=None):
        cmd = [sys.executable.split('/')[-1], "-m", "paddle.distributed.launch"]
        if args:
            cmd.extend(args.split(" "))
        cmd.extend([pyname])
        env = os.environ.copy()
        # virtual devies for testing
        env.update({'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
        proc = subprocess.Popen(cmd, env=env)
        return proc

    def test_collective_1(self):
        args = "--job_id test1"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

    def test_collective_2(self):
        if os.path.exists('./log'):
            shutil.rmtree('./log')

        args = "--job_id test2 --devices 0,1,2"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

        c = get_files('log', 'test2')
        self.assertTrue(len(c) == 4)

    def test_collective_3(self):
        if os.path.exists('./log'):
            shutil.rmtree('./log')

        port = random.randrange(6000, 8000)
        args = "--job_id test3 --devices 0,1 --master 127.0.0.1:{} --np 2".format(
            port)
        p1 = self.pdrun(args)
        p2 = self.pdrun(args)
        p1.wait()
        p2.wait()
        self.assertTrue(p1.poll() == 0)
        self.assertTrue(p2.poll() == 0)

        c = get_files('log', 'test3')
        self.assertTrue(len(c) == 6)


class PS_Test(unittest.TestCase):
    def setUp(self):
        write_file(pyname, pspyfile)

    def pdrun(self, args, env=None):
        cmd = [sys.executable.split('/')[-1], "-m", "paddle.distributed.launch"]
        if args:
            cmd.extend(args.split(" "))
        cmd.extend([pyname])
        proc = subprocess.Popen(cmd, env)
        return proc

    def test_ps_1(self):
        args = "--run_mode ps"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

    def test_ps_2(self):
        if os.path.exists('./log'):
            shutil.rmtree('./log')

        args = "--job_id ps2 --server_num=2 --trainer_num=2"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

        c = get_files('log', 'ps2')
        self.assertTrue(len(c) == 5)

    def test_ps_3(self):
        if os.path.exists('./log'):
            shutil.rmtree('./log')

        port = random.randrange(6000, 8000)
        args = "--job_id ps3 --master 127.0.0.1:{} --np 2 --server_num=1 --trainer_num=1".format(
            port)
        p1 = self.pdrun(args)
        p2 = self.pdrun(args)
        p1.wait()
        p2.wait()
        self.assertTrue(p1.poll() == 0)
        self.assertTrue(p2.poll() == 0)

        c = get_files('log', 'ps3')
        self.assertTrue(len(c) == 6)

    def test_ps_4(self):
        if os.path.exists('./log'):
            shutil.rmtree('./log')

        args = "--job_id ps4 --servers 127.0.0.1:8900,127.0.0.1:8901 --trainers 127.0.0.1:8902,127.0.0.1:8903"
        p1 = self.pdrun(args)
        p1.wait()
        self.assertTrue(p1.poll() == 0)

        c = get_files('log', 'ps4')
        self.assertTrue(len(c) == 5)


if __name__ == '__main__':
    unittest.main()
