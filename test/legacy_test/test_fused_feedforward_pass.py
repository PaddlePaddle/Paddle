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

import unittest

import numpy as np

import paddle
from paddle import nn
from paddle.base import core
from paddle.distributed.passes import PassManager, new_pass

paddle.enable_static()


class FeedForward(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        drop_prob=0.1,
        act_layer=nn.GELU,
        pre_layer_norm=True,
        add_residual=True,
        use_dropout_1=True,
        use_dropout_2=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.in_features = out_features
        self.pre_layer_norm = pre_layer_norm
        self.add_residual = add_residual
        self.use_dropout_1 = use_dropout_1
        self.use_dropout_2 = use_dropout_2

        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.norm = nn.LayerNorm(in_features, epsilon=1e-5)
        self.fc4 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        residual = x
        if self.pre_layer_norm:
            x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        if self.use_dropout_1:
            x = self.drop1(x)
        x = self.fc3(x)
        if self.use_dropout_2:
            x = self.drop2(x)
        if self.add_residual:
            x += residual
        if not self.pre_layer_norm:
            x = self.norm(x)
        x = self.fc4(x)

        return x


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedFeedforwardPass(unittest.TestCase):
    def setUp(self):
        self.pre_layer_norm = True
        self.add_residual = True
        self.use_dropout_1 = True
        self.use_dropout_2 = True

    def get_value(self, use_pass=False):
        batch_size = 2
        in_features = 768
        hidden_features = 3072
        out_features = 768
        act_layer = nn.GELU
        pre_layer_norm = self.pre_layer_norm
        add_residual = self.add_residual
        use_dropout_1 = self.use_dropout_1
        use_dropout_2 = self.use_dropout_2

        np.random.seed(1234)
        x_data = np.random.rand(batch_size, in_features, in_features).astype(
            'float32'
        )

        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        paddle.seed(1234)

        with paddle.static.program_guard(main_prog, startup_prog):
            data = paddle.static.data(
                name="x",
                shape=[2, in_features, in_features],
                dtype='float32',
            )

            feed_forward = FeedForward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                drop_prob=1e-10,
                act_layer=act_layer,
                pre_layer_norm=pre_layer_norm,
                add_residual=add_residual,
                use_dropout_1=use_dropout_1,
                use_dropout_2=use_dropout_2,
            )

            out = feed_forward(data)

            loss = paddle.mean(out)
            sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(loss)

        if use_pass:
            pass_manager = PassManager([new_pass("fused_feedforward")])
            pass_manager.apply([main_prog], [startup_prog])

            ops = main_prog.global_block().ops
            assert 'fused_feedforward' in [op.type for op in ops]
            assert 'fused_feedforward_grad' in [op.type for op in ops]

        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        exe.run(startup_prog)

        for i in range(2):
            ret_loss = exe.run(
                main_prog, feed={"x": x_data}, fetch_list=[loss.name]
            )

        return ret_loss

    def test_pass(self):
        for pre_layer_norm in [True, False]:
            for add_residual in [True, False]:
                for use_dropout_1 in [True, False]:
                    for use_dropout_2 in [True, False]:
                        if not pre_layer_norm and not add_residual:
                            continue
                        if not use_dropout_1 and not use_dropout_2:
                            continue
                        self.pre_layer_norm = pre_layer_norm
                        self.add_residual = add_residual
                        self.use_dropout_1 = use_dropout_1
                        self.use_dropout_2 = use_dropout_2
                        ret_loss = self.get_value()
                        ret_loss_fused = self.get_value(use_pass=True)
                        np.testing.assert_allclose(
                            ret_loss, ret_loss_fused, rtol=1e-5, atol=1e-8
                        )


if __name__ == "__main__":
    unittest.main()
