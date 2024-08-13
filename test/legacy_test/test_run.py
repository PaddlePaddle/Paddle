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

import os
import random
import subprocess
import sys
import tempfile
import unittest
from os import listdir
from os.path import isfile, join

pyname = 'train.py'
colpyfile = '''# train.py for unittest
import os
env = os.environ.copy()
if "PADDLE_AUTO_PARALLEL_CONFIG" not in env:
    assert "PADDLE_MASTER" in env
    assert "PADDLE_GLOBAL_RANK" in env
    assert "PADDLE_LOCAL_RANK" in env
assert "PADDLE_GLOBAL_SIZE" in env
assert "PADDLE_LOCAL_SIZE" in env
'''

pspyfile = '''# train.py for unittest
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
        f
        for f in listdir(pth)
        if isfile(join(pth, f))
        and not f.endswith('gpu.log')
        and not f.startswith('envlog')
        and not f.startswith('backup_env')
    ]


class Collective_Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, pyname)
        write_file(self.path, colpyfile)

    def tearDown(self):
        self.temp_dir.cleanup()

    def pdrun(self, args, env=None):
        cmd = [sys.executable.split('/')[-1], "-m", "paddle.distributed.launch"]
        if args:
            cmd.extend(args.split(" "))
        cmd.extend([self.path])
        env = os.environ.copy()
        # virtual devies for testing
        env.update({'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})
        proc = subprocess.Popen(cmd, env=env)
        return proc

    def test_collective_1(self):
        log_dir = tempfile.TemporaryDirectory()
        args = f"--job_id test1 --log_dir {log_dir.name}"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)
        log_dir.cleanup()

    def test_collective_2(self):
        log_dir = tempfile.TemporaryDirectory()
        args = f"--job_id test2 --devices 0,1,2 --log_dir {log_dir.name}"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

        c = get_files(log_dir.name, 'test2')
        self.assertTrue(len(c) == 4)
        log_dir.cleanup()

    def test_collective_3(self):
        log_dir = tempfile.TemporaryDirectory()
        port = random.randrange(6000, 8000)
        args = "--job_id test3 --devices 0,1 --log_dir {} --master 127.0.0.1:{} --nnodes 2"
        p1 = self.pdrun(args.format(log_dir.name + "/1", port))
        p2 = self.pdrun(args.format(log_dir.name + "/2", port))
        p1.wait()
        p2.wait()
        self.assertTrue(p1.poll() == 0)
        self.assertTrue(p2.poll() == 0)

        c1 = get_files(log_dir.name + "/1", 'test3')
        c2 = get_files(log_dir.name + "/2", 'test3')
        print(c1)
        self.assertTrue(len(c1) == 3)
        self.assertTrue(len(c2) == 3)
        log_dir.cleanup()

    def test_collective_4(self):
        log_dir = tempfile.TemporaryDirectory()
        config_dir = tempfile.TemporaryDirectory()
        config_path = os.path.join(config_dir.name, 'auto_parallel_config.json')
        with open(config_path, 'w') as wobj:
            wobj.write(
                '{"tuner_save_path":"parallel_strategy.pkl","tuner_load_path":"parallel_strategy.pkl","tuner_run_mode":"tuner_and_run"}'
            )
        port = random.randrange(6000, 8000)
        args = "--job_id test4 --devices 0,1 --log_dir {} --auto_parallel_config {}"
        p1 = self.pdrun(args.format(log_dir.name + "/1", config_path))
        p1.wait()
        self.assertTrue(p1.poll() == 0)

        c1 = get_files(log_dir.name + "/1", 'test4')
        print(c1)
        self.assertTrue(len(c1) == 4)
        log_dir.cleanup()
        config_dir.cleanup()


class PS_Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, pyname)
        write_file(self.path, pspyfile)

    def tearDown(self):
        self.temp_dir.cleanup()

    def pdrun(self, args, env=None):
        cmd = [sys.executable.split('/')[-1], "-m", "paddle.distributed.launch"]
        if args:
            cmd.extend(args.split(" "))
        cmd.extend([self.path])
        proc = subprocess.Popen(cmd, env)
        return proc

    def test_ps_1(self):
        log_dir = tempfile.TemporaryDirectory()
        args = f"--run_mode ps --log_dir {log_dir.name}"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)
        log_dir.cleanup()

    def test_ps_2(self):
        log_dir = tempfile.TemporaryDirectory()
        args = f"--job_id ps2 --server_num=2 --trainer_num=2 --log_dir {log_dir.name}"
        p = self.pdrun(args)
        p.wait()
        self.assertTrue(p.poll() == 0)

        c = get_files(log_dir.name, 'ps2')
        self.assertTrue(len(c) == 5)
        log_dir.cleanup()

    def test_ps_3(self):
        log_dir = tempfile.TemporaryDirectory()
        port = random.randrange(6000, 8000)
        args = "--job_id ps3 --log_dir {} --master 127.0.0.1:{} --nnodes 2 --server_num=1 --trainer_num=1"
        p1 = self.pdrun(args.format(log_dir.name + "/1", port))
        p2 = self.pdrun(args.format(log_dir.name + "/2", port))
        p1.wait()
        p2.wait()
        self.assertTrue(p1.poll() == 0)
        self.assertTrue(p2.poll() == 0)

        c1 = get_files(log_dir.name + "/1", 'ps3')
        c2 = get_files(log_dir.name + "/2", 'ps3')
        print(c1)
        self.assertTrue(len(c1) == 3)
        self.assertTrue(len(c2) == 3)
        log_dir.cleanup()

    def test_ps_4(self):
        log_dir = tempfile.TemporaryDirectory()
        args = f"--job_id ps4 --log_dir {log_dir.name} --servers 127.0.0.1:8900,127.0.0.1:8901 --trainers 127.0.0.1:8902,127.0.0.1:8903"
        p1 = self.pdrun(args)
        p1.wait()
        self.assertTrue(p1.poll() == 0)

        c = get_files(log_dir.name, 'ps4')
        print(c)
        self.assertTrue(len(c) == 5)
        log_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
