# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from amp_base_models import AmpTestBase, build_conv_model

import paddle


class TestAMPMPTypeClip(AmpTestBase):
    def get_dypgraph_result(self, dtype, level, use_promote):

        losses = []
        model, optimizer, scaler = build_conv_model(
            use_amp=True,
            amp_dtype=dtype,
            amp_level=level,
            use_promote=use_promote,
        )
        model.train()

        with paddle.amp.auto_cast(
            enable=True, dtype=dtype, level=level, use_promote=use_promote
        ):
            x = paddle.rand(shape=[1, 1, 6, 6], dtype='float32')
            out = model(x)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)
        optimizer.clear_grad()

        return losses

    def get_static_result(self):

        losses = []

        return losses

    def test_compare_o2_dygraph_static(self):
        losses_dygraph = self.get_dypgraph_result()
        losses_static = self.get_static_result()
