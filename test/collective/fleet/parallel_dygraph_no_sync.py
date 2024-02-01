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

import os
import random

import numpy as np
from legacy_test.test_dist_base import (
    TestParallelDyGraphRunnerBase,
    dump_output,
    print_to_err,
    runtime_main,
)

import paddle
import paddle.distributed as dist
from paddle import base
from paddle.nn import Linear

seed = 90
RUN_STEP = 20
batch_size = 4
batch_num = 1000


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.net_a = Linear(10, 20)
        self.net_b = Linear(20, 5)
        self.net_c = Linear(5, 10)

    def forward(self, x):
        x = self.net_a(x)
        x = self.net_b(x)
        x = self.net_c(x)
        return x


class TestNoSync(TestParallelDyGraphRunnerBase):
    def get_model(self):
        model = SimpleNet()
        train_reader = paddle.batch(
            fake_sample_reader(), batch_size=batch_size, drop_last=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=0.001, parameters=model.parameters()
        )
        return model, train_reader, optimizer

    def run_one_loop(self, model, optimizer, batch):
        x_data = np.array(list(batch))
        x_data = x_data.reshape((-1, 10))
        x = paddle.to_tensor(x_data)
        out = model(x)
        loss = out.sum() / len(batch)
        return loss

    def run_trainer_func(self, args):
        if base.core.is_compiled_with_cuda():
            device_id = int(os.getenv("FLAGS_selected_gpus", "0"))
            place = base.CUDAPlace(device_id)
        else:
            assert "Only support CUDAPlace for now."

        with base.dygraph.guard(place):
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model, train_reader, opt = self.get_model()

            if args.update_method == "nccl2":
                dist.init_parallel_env()
                print_to_err(
                    type(self).__name__,
                    "begin to prepare context in dygraph with nccl2",
                )
                model = paddle.DataParallel(
                    model, find_unused_parameters=args.find_unused_parameters
                )
            print_to_err(type(self).__name__, "model built in dygraph")
            out_losses = self.model_train(args, model, opt, train_reader)
            dump_output(out_losses)
            return out_losses

    def run_trainer_with_spawn_func(self, args):
        # 1. enable dygraph
        paddle.disable_static()

        # 2. init seed
        seed = 90
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # get trainer id
        args.trainer_id = paddle.distributed.get_rank()

        # 3. init parallel env
        if args.update_method in ["nccl2", "gloo"]:
            paddle.distributed.init_parallel_env()

        # 4. train model
        model, train_reader, opt = self.get_model()
        if args.update_method in ["nccl2", "gloo"]:
            model = paddle.DataParallel(
                model, find_unused_parameters=args.find_unused_parameters
            )

        out_losses = self.model_train(args, model, opt, train_reader)
        dump_output(out_losses)
        return out_losses

    def model_train(self, args, model, opt, train_reader):
        out_losses = []
        for step_id, data in enumerate(train_reader()):
            data = self._get_data(data, args)
            if step_id == RUN_STEP:
                break
            if step_id % 3 != 0:
                if args.update_method == "nccl2":
                    with model.no_sync():
                        loss = self.run_one_loop(model, opt, data)
                        loss.backward()
                else:
                    loss = self.run_one_loop(model, opt, data)
                    loss.backward()
            else:
                loss = self.run_one_loop(model, opt, data)
                loss.backward()
                opt.minimize(loss)
                out_losses.append(loss.numpy())
                model.clear_gradients()
        return out_losses


def fake_sample_reader():
    def __reader__():
        for i in range(batch_num):
            x_data = np.random.random_sample((10,)).astype('float32')
            yield x_data

    return __reader__


if __name__ == "__main__":
    runtime_main(TestNoSync)
