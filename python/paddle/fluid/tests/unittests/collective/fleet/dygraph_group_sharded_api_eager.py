# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import tempfile
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.framework import _test_eager_guard
from paddle.distributed.sharding import (
    group_sharded_parallel,
    save_group_sharded_model,
)

epoch = 10
paddle.seed(2022)
np.random.seed(2022)
base_lr = 0.1
momentum_rate = 0.9
l2_decay = 1e-4
batch_size = 100


class MLP(fluid.Layer):
    def __init__(self, linear_size=1000, param_attr=None, bias_attr=None):
        super().__init__()

        self._linear1 = Linear(linear_size, linear_size)
        self._linear2 = Linear(linear_size, linear_size)
        self._linear3 = Linear(linear_size, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        y = self._linear3(y)
        return y


def reader_decorator(linear_size=1000):
    def __reader__():
        for _ in range(100):
            img = np.random.rand(linear_size).astype('float32')
            label = np.ones(1).astype('int64')
            yield img, label

    return __reader__


def optimizer_setting(model, use_multi_precision, opt_group=False):
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    optimizer = paddle.optimizer.Momentum(
        parameters=[{"params": list(model.parameters())}]
        if opt_group
        else list(model.parameters()),
        learning_rate=0.001,
        weight_decay=0.00001,
        grad_clip=clip,
        multi_precision=use_multi_precision,
    )

    return optimizer


def train_mlp(
    model,
    shard_level,
    use_multi_precision,
    output_dir,
    amp_level='O1',
    sync_buffers=False,
    dp_group=None,
):
    optimizer = optimizer_setting(
        model=model, use_multi_precision=use_multi_precision
    )
    model = paddle.amp.decorate(
        models=model, level=amp_level, save_dtype='float32'
    )
    scaler = paddle.amp.GradScaler(init_loss_scaling=32768)

    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level=shard_level,
        scaler=scaler,
        sync_buffers=sync_buffers,
        dp_group=dp_group,
    )

    train_reader = paddle.batch(
        reader_decorator(), batch_size=batch_size, drop_last=True
    )

    train_loader = paddle.io.DataLoader.from_generator(
        capacity=32,
        use_double_buffer=True,
        iterable=True,
        return_list=True,
        use_multiprocess=True,
    )
    train_loader.set_sample_list_generator(train_reader)

    for eop in range(epoch):
        model.train()
        for batch_id, data in enumerate(train_loader()):
            img, label = data
            label.stop_gradient = True
            img.stop_gradient = True
            with paddle.amp.auto_cast(True, level=amp_level):
                out = model(img)
                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label
                )
            avg_loss = paddle.mean(x=loss.cast(dtype=paddle.float32))

            if not use_multi_precision:
                avg_loss.backward()
                optimizer.step()
            else:
                scaler.scale(avg_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            optimizer.clear_grad()

    save_group_sharded_model(model, output=output_dir, optimizer=optimizer)
    return model.parameters()


def test_sharding_api():
    paddle.distributed.init_parallel_env()
    mlp, mlp1, mlp2 = MLP(), MLP(), MLP()
    state_dict = mlp.state_dict()
    mlp1.set_state_dict(state_dict)
    mlp2.set_state_dict(state_dict)

    output_dir = tempfile.mkdtemp()

    # test sharding + dp, just for test
    dp_group = paddle.distributed.new_group(
        list(range(paddle.distributed.get_world_size()))
    )

    # fp16
    stage2_params = train_mlp(
        mlp1,
        shard_level="os_g",
        use_multi_precision=True,
        output_dir=output_dir,
        amp_level='O2',
    )
    stage3_params = train_mlp(
        mlp2,
        shard_level="p_g_os",
        use_multi_precision=True,
        output_dir=output_dir,
        amp_level='O2',
    )

    for i in range(len(stage3_params)):
        np.testing.assert_allclose(
            stage2_params[i].numpy(),
            stage3_params[i].numpy(),
            rtol=1e-4,
            atol=1e-3,
        )

    # AMP
    mlp3, mlp4 = MLP(), MLP()
    mlp3.set_state_dict(state_dict)
    mlp4.set_state_dict(state_dict)

    stage2_params = train_mlp(
        mlp3,
        shard_level="os_g",
        use_multi_precision=True,
        output_dir=output_dir,
        amp_level='O1',
    )
    stage3_params = train_mlp(
        mlp4,
        shard_level="p_g_os",
        use_multi_precision=True,
        output_dir=output_dir,
        amp_level='O1',
    )


if __name__ == '__main__':
    with _test_eager_guard():
        test_sharding_api()
