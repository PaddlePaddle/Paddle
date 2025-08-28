# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.distributed.passes.pass_utils import split_program
from paddle.vision.models import resnet18 as resnet


class TestSplitProgram(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({'FLAGS_cudnn_deterministic': 1})

    def get_model(self, batch_size):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            image = paddle.static.data(
                shape=[batch_size, 3, 224, 224], dtype='float32', name='image'
            )
            label = paddle.static.data(
                shape=[batch_size, 1], dtype='int64', name='label'
            )

            model = resnet(pretrained=False)
            loss_fn = nn.loss.CrossEntropyLoss()

            pred_out = model(image)
            loss = loss_fn(pred_out, label)

            optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)
        return main, startup, image, label

    def find_startup_vars(self, main_prog, startup_prog):
        self.assertEqual(startup_prog.num_blocks, 1)
        startup_vars = []
        for op in startup_prog.global_block().ops:
            for var_name in op.output_arg_names:
                var = main_prog.global_block().var(var_name)
                if var.persistable:
                    startup_vars.append(var_name)
        return startup_vars

    def test_split_program(self):
        for p in self.get_places():
            vars_expected = self.check_split_program(p, use_split=False)
            vars_actual = self.check_split_program(p, use_split=True)
            self.assertEqual(len(vars_actual), len(vars_expected))
            for actual, expected in zip(vars_actual, vars_expected):
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(
                    actual,
                    expected,
                    err_msg=f'{actual}\n{expected}\n',
                )

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        return places

    def get_var_values(self, scope, var_names):
        values = []
        for var_name in var_names:
            values.append(np.array(scope.find_var(var_name).get_tensor()))
        return values

    def check_split_program(self, place, use_split=True, seed=100, batch_num=5):
        batch_size = 2

        np.random.seed(seed)
        paddle.seed(seed)

        main_prog, startup_prog, image, label = self.get_model(batch_size)
        startup_vars = self.find_startup_vars(main_prog, startup_prog)
        exe = paddle.static.Executor(place)

        image_np = np.random.random(size=image.shape).astype('float32')
        label_np = np.random.randint(
            low=0, high=1000, dtype='int64', size=label.shape
        )

        scope = paddle.static.Scope()
        if not use_split:
            with paddle.static.scope_guard(scope):
                exe.run(startup_prog)
                for _ in range(batch_num):
                    exe.run(
                        main_prog,
                        feed={image.name: image_np, label.name: label_np},
                    )
            return self.get_var_values(scope, startup_vars)

        op_num = len(main_prog.global_block().ops)
        split_op_indices = [int(op_num / 3.0), int(op_num * 3 / 4.0)]
        programs, input_vars, output_vars = split_program(
            main_prog, split_op_indices
        )
        op_nums = [0, *split_op_indices, op_num]
        op_nums = [op_nums[i + 1] - op_nums[i] for i in range(len(op_nums) - 1)]
        num_split = len(split_op_indices) + 1
        self.assertEqual(len(programs), num_split)
        self.assertEqual(len(input_vars), num_split)
        self.assertEqual(len(output_vars), num_split)
        self.assertEqual(len(programs), len(op_nums))
        for p, n in zip(programs, op_nums):
            self.assertEqual(len(p.global_block().ops), n)

        with paddle.static.scope_guard(scope):
            exe.run(startup_prog)
            for _ in range(batch_num):
                tmp_vars = {image.name: image_np, label.name: label_np}
                for i, program in enumerate(programs):
                    feed_dict = {}
                    for in_name in input_vars[i]:
                        if in_name in startup_vars:
                            continue
                        self.assertTrue(in_name in tmp_vars)
                        if tmp_vars[in_name] is not None:
                            feed_dict[in_name] = tmp_vars[in_name]

                    output_var_values = exe.run(
                        program,
                        feed=feed_dict,
                        fetch_list=output_vars[i],
                        return_numpy=False,
                    )
                    for out_name, out_value in zip(
                        output_vars[i], output_var_values
                    ):
                        if not out_value._is_initialized():
                            tmp_vars[out_name] = np.ndarray(
                                out_value._get_dims()
                            ).astype('float32')
                        else:
                            tmp_vars[out_name] = np.array(out_value)

        return self.get_var_values(scope, startup_vars)


if __name__ == "__main__":
    unittest.main()
