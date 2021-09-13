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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.static as static
from paddle.distributed import fleet
from paddle.optimizer.ema import EMA


def gen_data():
    return {'x': np.random.random(size=(10, 5)).astype('float32')}


class TestParallelExecutorEMA(unittest.TestCase):
    def setUp(self):
        self._places = [paddle.CPUPlace()]
        # if paddle.device.is_compiled_with_cuda():
        #     self._places.append(paddle.CUDAPlace(0))
        self._ema_decay = 0.999
        self._param_name = "fc.weight"
        self._train_program = static.Program()
        self._startup_prog = static.Program()

        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

        with static.program_guard(self._train_program, self._startup_prog):
            with paddle.utils.unique_name.guard():
                data = static.data(name='x', shape=[-1, 5], dtype='float32')
                hidden = static.nn.fc(x=data,
                                      size=10,
                                      weight_attr=self._param_name)
                cost = paddle.mean(hidden)
                self._cost = cost
                optimizer = paddle.optimizer.Adam(learning_rate=0.001)
                optimizer.minimize(cost)

                self._test_program = static.default_main_program().clone(
                    for_test=True)
                self._ema = EMA(self._ema_decay)
                self._ema.update()

    def train(self, place):
        exe = static.Executor(place)
        exe.run(self._startup_prog)

        use_cuda = False if place is paddle.CPUPlace else True
        train_exe = static.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self._train_program,
            loss_name=self._cost.name)
        test_exe = static.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self._test_program,
            share_vars_from=train_exe)

        fetch_list = [self._cost.name]

        params = []
        for pass_id in range(2):
            for batch_id in range(3):
                train_exe.run(feed=gen_data(), fetch_list=fetch_list)
                tmp_param = np.array(static.global_scope().find_var(
                    self._param_name).get_tensor())
                params.append(tmp_param)
        with self._ema.apply(exe, gen_data(), fetch_list):
            final_ema = np.array(static.global_scope().find_var(
                self._param_name).get_tensor())
            test_exe.run(feed=gen_data(), fetch_list=fetch_list)

        return params, final_ema

    def test_check_ema(self):
        for place in self._places:
            params, final_ema = self.train(place)
            manu_ema = np.zeros_like(final_ema)
            if len(params) > 0:
                for param in params:
                    manu_ema = self._ema_decay * manu_ema + (1 - self._ema_decay
                                                             ) * param
                manu_ema = manu_ema / (1.0 - self._ema_decay**len(params))
            # print("\n======== ", place, " =========")
            # print("\nmanu_ema : ", manu_ema)
            # print("\nfinal_ema : ", final_ema)
            self.assertTrue(np.allclose(manu_ema, final_ema))


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
