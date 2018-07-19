#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid


class ModelAverage():
    """Model Average.
    Args:
        average_window: The rate of average window.
        min_average_window: The minimum size of average window.
        max_average_window: The maximum size of average window.
    """

    def __init__(self, average_window, min_average_window, max_average_window):
        self.kMaxNumAccumulates = 16384
        self.num_updates = 0
        self.num_accumulates = 0
        self.old_num_accumulates = 0
        self.average_window = average_window
        self.max_average_window = max_average_window
        self.min_average_window = min_average_window
        self.sum_1 = np.array([0])
        self.sum_2 = np.array([0])
        self.sum_3 = np.array([0])

    def accumulate(self, param):
        """Accumulate sum of parameters.
           It should be called in each minibatch training.
        """
        self.num_updates += 1
        self.num_accumulates += 1
        self.sum_1 = self.sum_1 + param
        if self.num_updates % self.kMaxNumAccumulates == 0:
            self.sum_2 = self.sum_2 + self.sum_1
            self.sum_1 = np.array([0])
        if self.num_accumulates >= self.min_average_window and self.num_accumulates >= min(
                self.max_average_window, self.num_updates *
                self.average_window):
            self.sum_3 = self.sum_1 + self.sum_2
            self.sum_1 = np.array([0])
            self.sum_2 = np.array([0])
            self.old_num_accumulates = self.num_accumulates
            self.num_accumulates = 0

    def apply_average(self):
        """Get averaged parameters.
           
        """
        return (self.sum_1 + self.sum_2 + self.sum_3) / (
            self.num_accumulates + self.old_num_accumulates)


def test(use_cuda, windows_rate=0.15, min_window=100, max_window=1000):
    """Test model average.
    Args:
        use_cuda: Whether using cuda.
        windows_rate: The rate of average window.
        min_window: The minimum size of average window.
        max_window: The maximum size of average window.
    """
    prog = fluid.framework.Program()
    with fluid.program_guard(main_program=prog):
        x = fluid.layers.data(name='x', shape=[13], dtype='float32')

        y_predict = fluid.layers.fc(input=x, size=1, act=None)

        y = fluid.layers.data(name='y', shape=[1], dtype='float32')

        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        _, params_grads = sgd_optimizer.minimize(avg_cost)
        model_average = fluid.optimizer.ModelAverage(
            windows_rate,
            min_average_window=min_window,
            max_average_window=max_window)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=1)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    exe.run(fluid.default_startup_program())

    py_average = ModelAverage(windows_rate, min_window, max_window)
    for data in train_reader():
        fc_w = exe.run(prog,
                       feed=feeder.feed(data),
                       fetch_list=[params_grads[0][0]])
        py_average.accumulate(fc_w[0])
    param_0 = fluid.io.get_parameter_value(params_grads[0][0], exe)
    with model_average.apply(exe):
        param_average = fluid.io.get_parameter_value(params_grads[0][0], exe)
    param_1 = fluid.io.get_parameter_value(params_grads[0][0], exe)
    if not np.isclose(a=param_0, b=param_1, rtol=5e-3):
        exit(1)
    if not np.isclose(a=param_average, b=py_average.apply_average(), rtol=5e-3):
        exit(1)


test(False)
test(True)
