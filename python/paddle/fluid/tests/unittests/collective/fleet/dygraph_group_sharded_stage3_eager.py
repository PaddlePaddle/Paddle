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


import numpy as np

import paddle
from paddle import nn
from paddle.distributed.sharding import group_sharded_parallel
from paddle.fluid.framework import _test_eager_guard

paddle.seed(2022)
np.random.seed(2022)


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.first_stage = nn.Linear(4096, 4096, bias_attr=False)
        self.center_stage = nn.Linear(4096, 4096)
        self.center_stage.weight.stop_gradient = True
        self.center_stage.bias.stop_gradient = True
        self.final_stage = nn.Linear(4096, 2, bias_attr=False)

    def forward(self, x):
        x = self.first_stage(x)
        x = self.center_stage(x)
        x = self.final_stage(x)
        return x


def optimizer_setting(model, use_multi_precision):
    optimizer = paddle.optimizer.AdamW(
        learning_rate=0.001,
        parameters=model.parameters(),
        multi_precision=use_multi_precision,
    )
    return optimizer


def train_mlp(
    model,
    shard_level="p_g_os",
    use_multi_precision=False,
    output_dir="",
    amp_level='O1',
    sync_buffers=False,
    use_sharding=True,
    data=None,
):
    optimizer = optimizer_setting(
        model=model, use_multi_precision=use_multi_precision
    )
    if use_multi_precision:
        model = paddle.amp.decorate(models=model, level=amp_level)

    scaler = paddle.amp.GradScaler(init_loss_scaling=32768)

    if use_sharding:
        model, optimizer, scaler = group_sharded_parallel(
            model=model,
            optimizer=optimizer,
            level=shard_level,
            scaler=scaler,
            sync_buffers=sync_buffers,
        )

    res_loss = []
    for i in range(20):
        model.train()
        img = data[i]
        with paddle.amp.auto_cast(use_multi_precision, level=amp_level):
            out = model(img)
            avg_loss = out.mean()

        res_loss.append(avg_loss.item())

        if not use_multi_precision:
            avg_loss.backward()
            optimizer.step()
        else:
            scaler.scale(avg_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        optimizer.clear_grad()

    return res_loss


def test_sharding_api():
    paddle.distributed.init_parallel_env()

    # just test warning
    model = Model()
    model = paddle.amp.decorate(models=model, level="O2")
    optimizer = optimizer_setting(model=model, use_multi_precision=True)
    model, optimizer, scaler = group_sharded_parallel(
        model=model,
        optimizer=optimizer,
        level="p_g_os",
    )

    data = [paddle.randn([8, 4096]) for i in range(20)]

    model = Model()
    sd3_model = Model()
    sd3_model.set_state_dict(model.state_dict())

    # dp fp32
    dp_fp32_loss = train_mlp(
        model, use_multi_precision=False, use_sharding=False, data=data
    )

    # stage3 fp32
    sd3_fp32_loss = train_mlp(
        sd3_model,
        shard_level="p_g_os",
        use_multi_precision=False,
        use_sharding=True,
        data=data,
    )

    print("dp_fp32_loss: ", dp_fp32_loss)
    print("sd3_fp32_loss: ", sd3_fp32_loss)

    for i in range(len(dp_fp32_loss)):
        np.testing.assert_allclose(
            np.array(dp_fp32_loss[i]),
            np.array(sd3_fp32_loss[i]),
            rtol=1e-8,
            atol=1e-8,
        )

    model = Model()
    sd3_model = Model()
    sd3_model.set_state_dict(model.state_dict())

    # dp fp16
    dp_fp16_loss = train_mlp(
        model, use_multi_precision=True, use_sharding=False, data=data
    )

    # stage3 fp16
    sd3_fp16_loss = train_mlp(
        sd3_model,
        shard_level="p_g_os",
        use_multi_precision=True,
        use_sharding=True,
        data=data,
    )

    print("dp_fp316_loss: ", dp_fp32_loss)
    print("sd3_fp32_loss: ", sd3_fp32_loss)

    for i in range(len(dp_fp16_loss)):
        np.testing.assert_allclose(
            np.array(dp_fp16_loss[i]),
            np.array(sd3_fp16_loss[i]),
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == '__main__':
    with _test_eager_guard():
        test_sharding_api()
