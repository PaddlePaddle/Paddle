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
import shutil
import sys
import tempfile
import unittest

import paddle


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


def run_test(
    clip_after_allreduce=True,
    max_global_norm=-1.0,
    gradient_merge_steps=1,
    use_master_acc_grad=True,
    need_env={},
):
    temp_dir = tempfile.TemporaryDirectory()
    if not paddle.is_compiled_with_cuda():
        return
    if os.name == 'nt':
        return
    args = locals()
    log_dir = os.path.join(temp_dir.name, f'log_{os.getpid()}')
    cmd = [
        sys.executable,
        '-u',
        '-m',
        'paddle.distributed.launch',
        '--devices',
        '0,1',
        '--log_dir',
        log_dir,
        get_test_file(),
    ]

    cmd = ' '.join([shlex.quote(c) for c in cmd])

    os.environ['CLIP_AFTER_ALLREDUCE'] = str(clip_after_allreduce)
    os.environ['MAX_GLOBAL_NORM'] = str(max_global_norm)
    os.environ['GRADIENT_MERGE_STEPS'] = str(gradient_merge_steps)
    os.environ['USE_MASTER_ACC_GRAD'] = str(1 if use_master_acc_grad else 0)
    os.environ["FLAGS_dynamic_static_unified_comm"] = "1"
    os.environ.update(need_env)

    touch_file_env = 'SUCCESS_TOUCH_FILE'
    touch_file_name = os.path.join(
        temp_dir.name,
        f'distributed_fused_lamb_touch_file_{os.getpid()}',
    )
    os.environ[touch_file_env] = touch_file_name
    try:
        assert os.system(cmd) == 0 and os.path.exists(
            touch_file_name
        ), f'Test failed when {args}'
    finally:
        temp_dir.cleanup()


class TestDistributedFusedLambWithClip(unittest.TestCase):
    def test_1(self):
        run_test(clip_after_allreduce=True, max_global_norm=0.01)

    def test_2(self):
        run_test(clip_after_allreduce=False, max_global_norm=0.01)

    def test_1_new_comm(self):
        run_test(
            clip_after_allreduce=True,
            max_global_norm=0.01,
            need_env={"FLAGS_dynamic_static_unified_comm": "true"},
        )

    def test_2_new_comm(self):
        run_test(
            clip_after_allreduce=False,
            max_global_norm=0.01,
            need_env={"FLAGS_dynamic_static_unified_comm": "true"},
        )


if __name__ == '__main__':
    unittest.main()
