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

from __future__ import print_function

import numpy as np
import paddle
import unittest

paddle.enable_static()


class TestParallelExecutorRunCinn(unittest.TestCase):
    def test_run_from_cinn(self):
        paddle.set_flags({'FLAGS_use_cinn': True})

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            data = paddle.static.data(
                name='X', shape=[None, 1], dtype='float32')
            prediction = paddle.static.nn.fc(data, 2)
            loss = paddle.mean(prediction)
            adam = paddle.optimizer.Adam()
            adam.minimize(loss)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(startup_program)
        compiled_program = paddle.static.CompiledProgram(
            main_program).with_data_parallel(loss_name=loss.name)

        batch_size = 16
        x = np.random.random(size=(batch_size, 1)).astype('float32')
        fetch = exe.run(compiled_program,
                        feed={'X': x},
                        fetch_list=[prediction.name],
                        return_merged=False)

        paddle.set_flags({'FLAGS_use_cinn': False})


if __name__ == '__main__':
    unittest.main()
