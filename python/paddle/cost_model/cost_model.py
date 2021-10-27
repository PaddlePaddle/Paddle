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
import json
from paddle.fluid import core


class CostModel():
    def __init__(self):
        pass

    def build_program(self):
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

        return startup_program, main_program

    def profile_measure(self,
                        startup_program,
                        main_program,
                        device='gpu',
                        fetch_cost_list=['time']):

        place = paddle.set_device('gpu')
        x = np.random.random(size=(10, 1)).astype('float32')
        exe = paddle.static.Executor(place)

        exe.run(startup_program)
        paddle.fluid.profiler.start_profiler("All")
        exe.run(main_program, feed={"X": x}, fetch_list=[])

        cost_model = core.CostModel()
        cost_data = cost_model.ProfileMeasure(device)

    def static_cost_data(self):
        static_cost_data_path = "./static_op_benchmark.json"
        with open(static_cost_data_path, 'r') as load_f:
            load_dict = json.load(load_f)
        self.static_cost_data = load_dict
        # return all static cost data
        return load_dict

    def get_static_op_time(self, op_name, config=None):
        if config == None:
            config = {"op_name": op_name, "forward": True, "dtype": "float32"}
        else:
            config["op_name"] = op_name
        if config["op_name"] == None:
            raise ValueError(
                'op_name should not be empty when you want to get static op time'
            )
        for op_data in self.static_cost_data:
            if (op_data["op"] == config["op_name"]) and (
                    config["dtype"] in op_data["config"]):
                if (config["forward"] == True):
                    return (op_data["paddle_gpu_time"])
                else:
                    return (op_data["paddle_gpu_time_backward"])
        return -1


cost_model = CostModel()
