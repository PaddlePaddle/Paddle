#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import framework
from paddle.framework.io_utils import is_pir_fetch_var

LARGE_PARAM = 2**26


class TestStaticSaveLoadLargeParameters(unittest.TestCase):

    def test_large_parameters_static_save(self):
        # enable static graph mode
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="static_save_load_large_x",
                shape=[None, 10],
                dtype='float32',
            )
            z = paddle.static.nn.fc(x, LARGE_PARAM, bias_attr=False)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()

            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    # make sure all the parameter or optimizer var have been update
                    self.assertTrue(np.sum(np.abs(t)) != 0)
                    base_map[var.name] = t
            temp_dir = tempfile.TemporaryDirectory()
            path = os.path.join(
                temp_dir.name, "test_static_save_load_large_param"
            )
            path = os.path.join(path, "static_save")
            protocol = 4
            paddle.static.save(prog, path, pickle_protocol=protocol)

            load_prog1 = paddle.static.Program()
            paddle.static.load(load_prog1, path)

            for var in load_prog1.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)

            load_prog2 = paddle.static.Program()
            program_state = paddle.static.load_program_state(path)
            paddle.static.set_program_state(load_prog2, program_state)
            for var in load_prog2.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    if is_pir_fetch_var(var):
                        continue
                    new_t = np.array(
                        base.global_scope().find_var(var.name).get_tensor()
                    )
                    base_t = base_map[var.name]
                    np.testing.assert_array_equal(new_t, base_t)
            temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
