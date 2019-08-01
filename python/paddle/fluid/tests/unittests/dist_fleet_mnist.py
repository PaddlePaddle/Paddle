#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import shutil
import tempfile
import time

import paddle.fluid as fluid
import os

from test_dist_fleet_collective_base import runtime_main, FleetDistRunnerBase

DTYPE = "float32"
paddle.dataset.mnist.fetch()

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistCollective2x2(FleetDistRunnerBase):
    def net(self, lr=0.01):
        # Input data
        data = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype=DTYPE)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=data,
            filter_size=5,
            num_filters=20,
            pool_size=2,
            pool_stride=2,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.01)))
        conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=conv_pool_1,
            filter_size=5,
            num_filters=50,
            pool_size=2,
            pool_stride=2,
            act="relu",
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.01)))

        SIZE = 10
        input_shape = conv_pool_2.shape
        param_shape = [reduce(lambda a, b: a * b, input_shape[1:], 1)] + [SIZE]
        scale = (2.0 / (param_shape[0]**2 * SIZE))**0.5

        predict = fluid.layers.fc(
            input=conv_pool_2,
            size=SIZE,
            act="softmax",
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.01)))

        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        self.predict = predict
        self.avg_cost = avg_cost

        # Reader
        train_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)

        return avg_cost

    def do_training(self, fleet):
        device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        place = fluid.CUDAPlace(device_id)

        exe = fluid.Executor(place)
        exe.run(fleet.startup_program)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.values()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        reader_generator = train_reader()

        def get_data():
            origin_batch = next(reader_generator)
            if args.update_method != "local" and args.use_reader_alloc:
                new_batch = []
                for offset, item in enumerate(origin_batch):
                    if offset % 2 == args.trainer_id:
                        new_batch.append(item)
                return new_batch
            else:
                return origin_batch

        for epoch_id in range(2):
            pass_start = time.time()
            exe.run(program=fleet.main_program,
                    fetch_list=[self.avg_cost],
                    feed=feeder.feed(get_data()))
            pass_time = time.time() - pass_start


if __name__ == "__main__":
    runtime_main(TestDistCollective2x2)
