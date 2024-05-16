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

import unittest

import paddle
from paddle.base.libpaddle.pir import translate_to_pir

paddle.enable_static()


class TestFusedCReducescatterAddPass(unittest.TestCase):
    def test_fused_matmul_swiglu(self):
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            x = paddle.static.data('X', [2, 2])
            out = paddle.static.data('Out', [2, 2])
            bias = paddle.static.data('Bias', [2, 2])
            main_prog.global_block().append_op(
                type="c_reducescatter",
                inputs={'X': x},
                attrs={'ring_id': 0, 'nranks': 1},
                outputs={'Out': out},
            )
            ans = paddle.add(out, bias)

        pir_program = translate_to_pir(main_prog.desc)

        serialized_pir_program = str(pir_program)

        assert "pd_op.c_reducescatter" in serialized_pir_program
        assert "pd_op.add" in serialized_pir_program

        pm = paddle.pir.PassManager()
        pm.add_pass('fuse_c_reducescatter_add_pass', {})
        pm.run(pir_program)

        serialized_pir_program = str(pir_program)

        assert "pd_op.c_reducescatter\"" not in serialized_pir_program
        assert "pd_op.add" not in serialized_pir_program
        assert "pd_op.c_reducescatter_add" in serialized_pir_program


if __name__ == "__main__":
    unittest.main()
