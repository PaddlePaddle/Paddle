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

import logging
import numpy as np
import paddle
import unittest

paddle.enable_static()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def set_cinn_flag(val):
    cinn_compiled = False
    try:
        paddle.set_flags({'FLAGS_use_cinn': val})
        cinn_compiled = True
    except ValueError:
        logger.warning("The used paddle is not compiled with CINN.")
    return cinn_compiled


@unittest.skipIf(not set_cinn_flag(True), "Paddle is not compiled with CINN.")
class TestParallelExecutorRunCinn(unittest.TestCase):
    def test_run_from_cinn(self):
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
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.debug_graphviz_path = "/work/model_struct/dotfiles/viz"
        compiled_program = paddle.static.CompiledProgram(
            main_program,
            build_strategy).with_data_parallel(loss_name=loss.name)

        batch_size = 16
        x = np.random.random(size=(batch_size, 1)).astype('float32')
        fetch = exe.run(compiled_program,
                        feed={'X': x},
                        fetch_list=[prediction.name],
                        return_merged=False)

        set_cinn_flag(False)


if __name__ == '__main__':
    unittest.main()
