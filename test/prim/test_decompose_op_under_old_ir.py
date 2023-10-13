# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


# run this: FLAGS_enable_new_ir_in_executor=True python test_program_prim.py or
#           FLAGS_enable_pir_api=True FLAGS_enable_new_ir_in_executor=True python test_program_prim.py
import unittest

import numpy as np

import paddle
from paddle import pir
from paddle.autograd.ir_backward import grad
from paddle.base import core
from paddle.decomposition import decompose

paddle.enable_static()


def get_gelu_pir_program():
    shape_x = [2, 3, 3]
    main_program = paddle.static.Program()
    start_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, start_program):
        x = paddle.static.data('x', shape_x, dtype='float32')
        x.stop_gradient = False
        res = paddle.nn.functional.gelu(x, approximate=False)

        # print("old ir: ", main_program)
        newir_program = pir.translate_to_new_ir(
            main_program.desc
        )  # set FLAGS_enable_new_ir_in_executor=True
        # print("new ir: ", newir_program)
    return newir_program


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.out_grad = np.random.random(self.shape_x).astype("float32")

    def base_net(self, flag):
        newir_program = get_gelu_pir_program()
        # whole_ops = [op.name() for op in newir_program.global_block().ops]
        # print("ops_1: ", whole_ops)

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            if flag == "forward":
                core._set_prim_backward_enabled(False)
                core._set_prim_forward_enabled(True)

                res_ = newir_program.global_block().ops[1].result(0)
                [res_] = decompose(
                    newir_program,
                    [res_],
                )
                whole_ops = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("ops_2: ", whole_ops)

                core._set_prim_backward_enabled(True)
                input_ = newir_program.global_block().ops[0].result(0)
                gradient = grad(res_, input_)[0]
                whole_ops = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("ops_3: ", whole_ops)

            elif flag == "backward":
                core._set_prim_backward_enabled(True)
                core._set_prim_forward_enabled(False)

                res_ = newir_program.global_block().ops[1].result(0)
                input_ = newir_program.global_block().ops[0].result(0)
                gradient = grad(res_, input_)[0]
                whole_ops = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("ops_2: ", whole_ops)

                core._set_prim_forward_enabled(True)
                [res_] = decompose(
                    newir_program,
                    [res_],
                )
                whole_ops = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("ops_3: ", whole_ops)

            elif flag == "none":
                whole_ops_1 = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("whole ops 1: ", whole_ops_1)

                res_ = newir_program.global_block().ops[1].result(0)
                input_ = newir_program.global_block().ops[0].result(0)
                gradient = grad(res_, input_)[0]

                whole_ops_2 = [
                    op.name() for op in newir_program.global_block().ops
                ]
                print("whole ops 2: ", whole_ops_2)

            exe = paddle.static.Executor()
            outs = exe.run(
                newir_program,
                feed={
                    'x': self.x,
                    'out_grad': self.out_grad,
                },
                fetch_list=[res_, gradient],
            )

        core._set_prim_backward_enabled(False)
        core._set_prim_forward_enabled(False)
        return outs

    def test_prim_custom_vjp(self):
        res1 = self.base_net("forward")
        # res2 = self.base_net("backward")
        # res3 = self.base_net("none")
        # for ref, actual in zip(res1, res2):
        #     np.testing.assert_allclose(ref, actual, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
