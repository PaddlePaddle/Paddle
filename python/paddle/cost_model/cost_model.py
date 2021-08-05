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
import paddle.static as static
import numpy as np


class CostModel():
    def __init__():
        pass

    def build_program():
        paddle.enable_static()

        main_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(
                main_program=main_program, startup_program=startup_program):
            data = paddle.static.data(
                name='X', shape=[None, 1], dtype='float32')
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

        print("main program is: {}".format(main_program))
        print("start up program is: {}".format(startup_program))

        return startup_program, main_program

    def start_cost_model():
        paddle.fluid.profiler.start_profiler("GPU")

    def stop_cost_model(cost_list=["time", "memory"]):

        if not core.is_profiler_enabled():
            return

        cost_list = core.stop_cost_model(cost_list)
        return cost_list

    def get_cost(startup_program, main_program):

        place = paddle.set_device('gpu')
        x = np.random.random(size=(10, 1)).astype('float32')
        exe = paddle.static.Executor(place)

        exe.run(startup_program)
        self.start_cost_model()
        exe.run(main_program, feed={"X": x}, fetch_list=[])

        cost_list = self.stop_cost_model()
        return cost_list

    startup_program, main_program = build_program()

    get_cost(startup_program, main_program)
