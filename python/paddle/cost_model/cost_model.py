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

import json
import os

import numpy as np

import paddle
from paddle import static
from paddle.base import core


class CostModel:
    def __init__(self):
        pass

    def build_program(self):
        paddle.enable_static()

        main_program = static.Program()
        startup_program = static.Program()
        with static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='X', shape=[None, 1], dtype='float32'
            )
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

        print(f"main program is: {main_program}")

        return startup_program, main_program

    def profile_measure(
        self,
        startup_program,
        main_program,
        device='gpu',
        fetch_cost_list=['time'],
    ):
        place = paddle.set_device('gpu')
        x = np.random.random(size=(10, 1)).astype('float32')
        exe = paddle.static.Executor(place)

        exe.run(startup_program)
        p = paddle.profiler.Profiler()
        p.start()
        exe.run(main_program, feed={"X": x}, fetch_list=[])

        cost_model = core.CostModel()
        cost_data = cost_model.ProfileMeasure(device)

    def static_cost_data(self):
        static_cost_data_path = os.path.join(
            os.path.dirname(__file__), "static_op_benchmark.json"
        )
        with open(static_cost_data_path, 'r') as load_f:
            load_dict = json.load(load_f)
        self._static_cost_data = load_dict
        # return all static cost data
        return load_dict

    def get_static_op_time(self, op_name, forward=True, dtype="float32"):
        # if forward is True, return op forward time, otherwise return op backward time.
        if op_name is None:
            raise ValueError(
                'op_name should not be empty when you want to get static op time'
            )

        op_cost = {}
        for op_data in self._static_cost_data:
            if (op_data["op"] == op_name) and (dtype in op_data["config"]):
                if forward:
                    op_cost["op_time"] = op_data["paddle_gpu_time"]
                else:
                    op_cost["op_time"] = op_data["paddle_gpu_time_backward"]
                op_cost["config"] = op_data["config"]

        return op_cost
