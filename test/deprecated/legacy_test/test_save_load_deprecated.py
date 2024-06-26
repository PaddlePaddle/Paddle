# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import pickle
import tempfile
import unittest
from io import BytesIO

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base, nn
from paddle.jit.api import to_static
from paddle.jit.translated_layer import INFER_PARAMS_INFO_SUFFIX
from paddle.nn import Linear
from paddle.static import InputSpec

IMAGE_SIZE = 784
CLASS_NUM = 10

SEED = 10


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


class LinearNetReturnHidden(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self._linear_1 = Linear(in_size, out_size)
        self._linear_2 = Linear(in_size, out_size)

    @to_static
    def forward(self, x):
        y = self._linear_1(x)
        z = self._linear_2(y)
        loss = paddle.mean(z)
        return y, loss


class TestSaveLoadProgram(unittest.TestCase):
    def test_save_load_program(self):
        paddle.enable_static()
        temp_dir = tempfile.TemporaryDirectory()

        with new_program_scope():
            layer = LinearNet()
            data = paddle.static.data(
                name='x_static_save', shape=(None, IMAGE_SIZE), dtype='float32'
            )
            y_static = layer(data)
            main_program = paddle.static.default_main_program()
            startup_program = paddle.static.default_startup_program()
            origin_main = main_program.desc.serialize_to_string()
            origin_startup = startup_program.desc.serialize_to_string()
            path1 = os.path.join(
                temp_dir.name,
                "test_paddle_save_load_program/main_program.pdmodel",
            )
            path2 = os.path.join(
                temp_dir.name,
                "test_paddle_save_load_program/startup_program.pdmodel",
            )
            paddle.save(main_program, path1)
            paddle.save(startup_program, path2)

        with new_program_scope():
            load_main = paddle.load(path1).desc.serialize_to_string()
            load_startup = paddle.load(path2).desc.serialize_to_string()
            self.assertTrue(origin_main == load_main)
            self.assertTrue(origin_startup == load_startup)
        temp_dir.cleanup()


class TestJitPruneModelAndLoad(unittest.TestCase):
    def setUp(self):
        self.linear_size = 4
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "jit_prune_model_and_load/model"
        )
        # enable dygraph mode
        base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save(self):
        train_layer = LinearNetReturnHidden(8, 8)
        train_layer = to_static(
            train_layer,
            input_spec=[InputSpec([None, 8], name='x')],
            full_graph=True,
        )
        adam = paddle.optimizer.Adam(
            learning_rate=0.1, parameters=train_layer.parameters()
        )
        x = paddle.to_tensor(np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            hidden, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        output_spec = train_layer.forward.outputs[:1]
        paddle.jit.save(
            layer=train_layer,
            path=self.model_path,
            input_spec=[x],
            output_spec=output_spec,
        )

        return train_layer

    # pir has no need to save extra var info, param always saved with program,
    # and trainable info saved in program's op attr
    def test_load_var_not_in_extra_var_info(self):
        self.train_and_save()

        # chage extra var info
        var_info_path = self.model_path + INFER_PARAMS_INFO_SUFFIX
        with open(var_info_path, 'rb') as f:
            extra_var_info = pickle.load(f)
            extra_var_info.clear()
        with open(var_info_path, 'wb') as f:
            pickle.dump(extra_var_info, f, protocol=2)

        with self.assertRaises(RuntimeError):
            paddle.jit.load(self.model_path)


class TestSaveLoadToMemory(unittest.TestCase):
    def test_static_save_to_memory(self):
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32'
            )
            z = paddle.static.nn.fc(x, 10, bias_attr=False)
            z = paddle.static.nn.fc(z, 128, bias_attr=False)
            loss = paddle.mean(z)
            place = (
                base.CPUPlace()
                if not paddle.base.core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            prog = paddle.static.default_main_program()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())

            state_dict = prog.state_dict()
            keys = list(state_dict.keys())
            tensor = state_dict[keys[0]]

            byio = BytesIO()
            byio2 = BytesIO()
            paddle.save(prog, byio2)
            paddle.save(tensor, byio)
            paddle.save(state_dict, byio)
            byio.seek(0)
            byio2.seek(0)

            prog_load = paddle.load(byio2)
            self.assertTrue(
                prog.desc.serialize_to_string()
                == prog_load.desc.serialize_to_string()
            )

            tensor_load = paddle.load(byio, return_numpy=True)
            np.testing.assert_array_equal(tensor_load, np.array(tensor))

            state_dict_load = paddle.load(byio, return_numpy=True)
            for k, v in state_dict.items():
                np.testing.assert_array_equal(np.array(v), state_dict_load[k])


if __name__ == '__main__':
    unittest.main()
