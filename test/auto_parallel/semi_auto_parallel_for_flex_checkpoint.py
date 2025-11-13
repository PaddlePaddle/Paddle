# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn

MODEL_STATE_DIC = "model_state"
OPTIMIZER_STATE_DIC = "optimizer_state"
MASTER_WEIGHT_DIC = "master_weight"


class SimpleModel(nn.Layer):
    def __init__(self, hidden_size=3072, layer_num=2):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestCheckpointConsistency:
    def __init__(self):
        self.ckpt_path = tempfile.TemporaryDirectory().name
        self.mesh = dist.ProcessMesh([0, 1], dim_names=['dp'])
        self.hidden_size = 256
        self.batch_size = 2
        paddle.seed(1024)
        random.seed(1024)
        np.random.seed(1024)

    def create_model_and_optimizer(self):
        model = SimpleModel(self.hidden_size)
        opt = paddle.optimizer.AdamW(
            learning_rate=0.001, parameters=model.parameters()
        )
        opt = dist.shard_optimizer(opt, dist.ShardingStage1("dp", self.mesh))
        model, opt = paddle.amp.decorate(
            model, optimizers=opt, level='O2', master_grad=True
        )
        return model, opt

    def run_training_and_save(self):
        model, opt = self.create_model_and_optimizer()

        for step in range(3):
            inputs = paddle.ones(
                [self.batch_size, self.hidden_size], dtype='float16'
            )
            labels = paddle.ones(
                [self.batch_size, self.hidden_size], dtype='float16'
            )
            inputs = dist.shard_tensor(inputs, self.mesh, [dist.Shard(0)])
            logits = model(inputs)
            loss = paddle.nn.functional.mse_loss(logits, labels)
            loss.backward()
            if step == 2:
                loss_md5 = loss._md5sum()
            else:
                opt.step()
            print(f"Train step {step}, loss: {loss.item()}")

        save_md5 = [p._md5sum() for p in model.parameters()]

        # save model and optimizer
        model_state_dict_path = os.path.join(self.ckpt_path, MODEL_STATE_DIC)
        opt_state_dict_path = os.path.join(self.ckpt_path, OPTIMIZER_STATE_DIC)
        master_weights_path = os.path.join(self.ckpt_path, MASTER_WEIGHT_DIC)
        sharded_state_dict = model.sharded_state_dict()
        dist.save_state_dict(sharded_state_dict, model_state_dict_path)
        optimizer_states = {}
        master_weights = {}
        opt_sharded_state_dict = opt.sharded_state_dict(sharded_state_dict)
        for k, v in opt_sharded_state_dict.items():
            if k.endswith(".w_0"):
                master_weights[k] = v
            else:
                optimizer_states[k] = v
        dist.save_state_dict(optimizer_states, opt_state_dict_path)
        dist.save_state_dict(master_weights, master_weights_path)
        return save_md5, loss_md5

    def run_loading_and_validation(self):
        model, opt = self.create_model_and_optimizer()

        # load model and optimizer
        model_state_dict_path = os.path.join(self.ckpt_path, MODEL_STATE_DIC)
        master_weights_path = os.path.join(self.ckpt_path, MASTER_WEIGHT_DIC)
        opt_states_path = os.path.join(self.ckpt_path, OPTIMIZER_STATE_DIC)
        sharded_state_dict = model.sharded_state_dict()
        dist.load_state_dict(sharded_state_dict, model_state_dict_path)
        opt_sharded_state_dict = opt.sharded_state_dict(sharded_state_dict)
        opt_states = {}
        master_weights = {}
        for k, v in opt_sharded_state_dict.items():
            if k.endswith(".w_0"):
                master_weights[k] = v
            else:
                opt_states[k] = v
        dist.load_state_dict(opt_states, opt_states_path)
        dist.load_state_dict(master_weights, master_weights_path)

        load_md5 = [p._md5sum() for p in model.parameters()]

        for step in range(1):
            inputs = paddle.ones(
                [self.batch_size, self.hidden_size], dtype='float16'
            )
            labels = paddle.ones(
                [self.batch_size, self.hidden_size], dtype='float16'
            )
            inputs = dist.shard_tensor(inputs, self.mesh, [dist.Shard(0)])
            logits = model(inputs)
            loss = paddle.nn.functional.mse_loss(logits, labels)
            loss.backward()
            opt.step()
            loss_md5 = loss._md5sum()
            print(f"Train step {step}, loss: {loss.item()}")
        return load_md5, loss_md5

    def run_test(self):
        save_param_md5sum, loss_md5 = self.run_training_and_save()
        load_param_md5sum, loss_md5_reload = self.run_loading_and_validation()
        np.testing.assert_equal(save_param_md5sum, load_param_md5sum)
        np.testing.assert_equal(loss_md5, loss_md5_reload)


if __name__ == '__main__':
    test = TestCheckpointConsistency()
    test.run_test()
