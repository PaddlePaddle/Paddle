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
from hybrid_parallel_mp_model import TestDistMPTraning
import paddle.distributed.fleet as fleet
import unittest


class TestMPFP16(TestDistMPTraning):

    def build_optimizer(self, model):
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.001,
                                                         gamma=0.999,
                                                         verbose=True)
        optimizer = paddle.optimizer.SGD(scheduler,
                                         grad_clip=grad_clip,
                                         parameters=model.parameters())

        model, optimizer = paddle.amp.decorate(models=model,
                                               optimizers=optimizer,
                                               level='O2',
                                               save_dtype='float32')

        return optimizer

    def train_batch(self, batch, model, optimizer, is_mp):
        scaler = paddle.amp.GradScaler(init_loss_scaling=5160)
        if is_mp:
            scaler = fleet.distributed_scaler(scaler)
        with paddle.amp.auto_cast(enable=True, level="O2"):
            output = model(batch)
            loss = output.mean()

        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.clear_grad()
        return scaled


if __name__ == "__main__":
    unittest.main()
