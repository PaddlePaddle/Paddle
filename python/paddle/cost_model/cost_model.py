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

<<<<<<< HEAD
import json
import os

import numpy as np

import paddle
import paddle.static as static
from paddle.fluid import core


class CostModel:
=======
import paddle
import paddle.static as static
import numpy as np
import json
import os
from paddle.fluid import core


class CostModel():

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        pass

    def build_program(self):
        paddle.enable_static()

        main_program = static.Program()
        startup_program = static.Program()
<<<<<<< HEAD
        with static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='X', shape=[None, 1], dtype='float32'
            )
=======
        with static.program_guard(main_program=main_program,
                                  startup_program=startup_program):
            data = paddle.static.data(name='X',
                                      shape=[None, 1],
                                      dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            hidden = paddle.static.nn.fc(data, 10)
            loss = paddle.mean(hidden)
            paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

        print("main program is: {}".format(main_program))

        return startup_program, main_program

<<<<<<< HEAD
    def profile_measure(
        self,
        startup_program,
        main_program,
        device='gpu',
        fetch_cost_list=['time'],
    ):
=======
    def profile_measure(self,
                        startup_program,
                        main_program,
                        device='gpu',
                        fetch_cost_list=['time']):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        place = paddle.set_device('gpu')
        x = np.random.random(size=(10, 1)).astype('float32')
        exe = paddle.static.Executor(place)

        exe.run(startup_program)
        paddle.fluid.profiler.start_profiler("All")
        exe.run(main_program, feed={"X": x}, fetch_list=[])

        cost_model = core.CostModel()
        cost_data = cost_model.ProfileMeasure(device)

    def static_cost_data(self):
<<<<<<< HEAD
        static_cost_data_path = os.path.join(
            os.path.dirname(__file__), "static_op_benchmark.json"
        )
=======
        static_cost_data_path = os.path.join(os.path.dirname(__file__),
                                             "static_op_benchmark.json")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        with open(static_cost_data_path, 'r') as load_f:
            load_dict = json.load(load_f)
        self._static_cost_data = load_dict
        # return all static cost data
        return load_dict

    def get_static_op_time(self, op_name, forward=True, dtype="float32"):
        # if forward is True, return op forward time, otherwise return op backward time.
<<<<<<< HEAD
        if op_name is None:
=======
        if op_name == None:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            raise ValueError(
                'op_name should not be empty when you want to get static op time'
            )

        op_cost = {}
        for op_data in self._static_cost_data:
            if (op_data["op"] == op_name) and (dtype in op_data["config"]):
<<<<<<< HEAD
                if forward:
=======
                if (forward):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    op_cost["op_time"] = op_data["paddle_gpu_time"]
                else:
                    op_cost["op_time"] = op_data["paddle_gpu_time_backward"]
                op_cost["config"] = op_data["config"]

        return op_cost
