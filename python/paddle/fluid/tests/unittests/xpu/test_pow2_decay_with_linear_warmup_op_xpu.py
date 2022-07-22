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

import paddle
from paddle.fluid.contrib.layers.nn import pow2_decay_with_linear_warmup
from paddle.optimizer.lr import LinearWarmup
from paddle.optimizer.lr import PolynomialDecay
import unittest
import sys

sys.path.append("..")

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import record_op_test


def gen_pow2_warmup_op_lr(warmup_steps, total_steps, base_lr, end_lr, place):
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        lr = pow2_decay_with_linear_warmup(warmup_steps, total_steps, base_lr,
                                           end_lr)
        exe = paddle.static.Executor(place)
    with paddle.static.scope_guard(paddle.static.Scope()):
        exe.run(startup)
        while True:
            lr_np = exe.run(main, fetch_list=[lr])[0]
            yield lr_np[0]


class Pow2Warmup(LinearWarmup):

    def __init__(self, warmup_steps, total_steps, base_lr, end_lr):
        assert total_steps > warmup_steps
        lr_sch = PolynomialDecay(learning_rate=base_lr,
                                 decay_steps=total_steps - warmup_steps,
                                 end_lr=end_lr,
                                 power=2)

        super(Pow2Warmup, self).__init__(learning_rate=lr_sch,
                                         warmup_steps=warmup_steps,
                                         start_lr=0.0,
                                         end_lr=base_lr)


def gen_pow2_warmup_py_lr(warmup_steps, total_steps, base_lr, end_lr, place):
    lr_sch = Pow2Warmup(warmup_steps, total_steps, base_lr, end_lr)
    lr_sch.step()
    while True:
        yield lr_sch()
        lr_sch.step()


class TestPowWarmup(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        self.op_type = 'pow2_decay_with_linear_warmup'
        self.params = {
            'warmup_steps': 30,
            'total_steps': 100,
            'base_lr': 0.02,
            'end_lr': 0.001,
        }
        self.step_num = 1000

    def check_with_place(self, place):
        kwargs = dict(self.params)
        kwargs['place'] = place
        lr_sch_op = gen_pow2_warmup_op_lr(**kwargs)
        lr_sch_py = gen_pow2_warmup_py_lr(**kwargs)
        for i, (lr_op, lr_py) in enumerate(zip(lr_sch_op, lr_sch_py)):
            self.assertLess(abs(lr_op - lr_py), 1e-6)
            if i > self.step_num:
                break

    def test_main(self):
        self.check_with_place(paddle.XPUPlace(0))


record_op_test("pow2_decay_with_linear_warmup", "float32")

if __name__ == "__main__":
    unittest.main()
