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
import shlex
<<<<<<< HEAD
import shutil
import sys
import tempfile
import unittest

import paddle
=======
import sys
import shutil
import unittest
import paddle
import tempfile
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def get_test_file():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(dirname, 'distributed_fused_lamb_test_base.py')


def remove_file_if_exists(file_name):
    if not os.path.exists(file_name):
        return
    if os.path.isfile(file_name):
        os.remove(file_name)
    else:
        shutil.rmtree(file_name)


<<<<<<< HEAD
def run_test(
    clip_after_allreduce=True,
    max_global_norm=-1.0,
    gradient_merge_steps=1,
    use_master_acc_grad=True,
):
=======
def run_test(clip_after_allreduce=True,
             max_global_norm=-1.0,
             gradient_merge_steps=1,
             use_master_acc_grad=True):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    temp_dir = tempfile.TemporaryDirectory()
    if not paddle.is_compiled_with_cuda():
        return
    if os.name == 'nt':
        return
    args = locals()
    log_dir = os.path.join(temp_dir.name, 'log_{}'.format(os.getpid()))
    cmd = [
        sys.executable,
        '-u',
        '-m',
        'paddle.distributed.launch',
        '--log_dir',
        log_dir,
        get_test_file(),
    ]

    cmd = ' '.join([shlex.quote(c) for c in cmd])

    os.environ['CLIP_AFTER_ALLREDUCE'] = str(clip_after_allreduce)
    os.environ['MAX_GLOBAL_NORM'] = str(max_global_norm)
    os.environ['GRADIENT_MERGE_STEPS'] = str(gradient_merge_steps)
    os.environ['USE_MASTER_ACC_GRAD'] = str(1 if use_master_acc_grad else 0)

    touch_file_env = 'SUCCESS_TOUCH_FILE'
    touch_file_name = os.path.join(
        temp_dir.name,
<<<<<<< HEAD
        'distributed_fused_lamb_touch_file_{}'.format(os.getpid()),
    )
    os.environ[touch_file_env] = touch_file_name
    try:
        assert os.system(cmd) == 0 and os.path.exists(
            touch_file_name
        ), 'Test failed when {}'.format(args)
=======
        'distributed_fused_lamb_touch_file_{}'.format(os.getpid()))
    os.environ[touch_file_env] = touch_file_name
    try:
        assert os.system(cmd) == 0 and os.path.exists(
            touch_file_name), 'Test failed when {}'.format(args)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    finally:
        temp_dir.cleanup()


class TestDistributedFusedLambWithClip(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_1(self):
        run_test(clip_after_allreduce=True, max_global_norm=0.01)

    def test_2(self):
        run_test(clip_after_allreduce=False, max_global_norm=0.01)


if __name__ == '__main__':
    unittest.main()
