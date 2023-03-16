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

import unittest

from hybrid_parallel_mp_model import TestDistMPTraning

import paddle
import paddle.distributed.fleet as fleet
from paddle.framework import core


class TestMPFP16(TestDistMPTraning):
    def build_optimizer(self, model):
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.001, gamma=0.999, verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            scheduler, grad_clip=grad_clip, parameters=model.parameters()
        )

        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O2', save_dtype='float32'
        )

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


class TestMPFP16MainGrad(TestMPFP16):
    def build_optimizer(self, model):
        grad_clip = paddle.nn.ClipGradByGlobalNorm(1.0)
        scheduler = paddle.optimizer.lr.ExponentialDecay(
            learning_rate=0.001, gamma=0.999, verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            scheduler, grad_clip=grad_clip, parameters=model.parameters()
        )

        for param in model.parameters():
            if not param.stop_gradient and not hasattr(param, "main_grad"):
                setattr(param, "main_grad", None)
                param._register_grad_hook(self._update_main_grad_hook(param))

        model, optimizer = paddle.amp.decorate(
            models=model, optimizers=optimizer, level='O2', save_dtype='float32'
        )

        return optimizer

    def _update_main_grad_hook(self, param):
        @paddle.autograd.no_grad()
        def param_hook(tmp_grad):
            assert (
                param.grad is None
            ), "In main_grad node, param.grad should be None, but find param[{}] has grad.".format(
                param.name
            )
            if param.main_grad is None:
                param.main_grad = core.eager.Tensor(
                    value=tmp_grad.detach().value(),
                    place=tmp_grad.place,
                    name="main_grad@" + param.name,
                )
            else:
                param.main_grad.add_(tmp_grad.detach())

            tmp_grad._clear_data()
            return None


if __name__ == "__main__":
    unittest.main()
