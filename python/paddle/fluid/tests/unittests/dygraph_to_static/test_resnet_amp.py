# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import time
import unittest

import numpy as np
from test_resnet import SEED, ResNet, optimizer_setting

import paddle
import paddle.fluid as fluid
from paddle.jit import ProgramTranslator

# NOTE: Reduce batch_size from 8 to 2 to avoid unittest timeout.
batch_size = 2
epoch_num = 1
place = (
    fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
)

program_translator = ProgramTranslator()

if fluid.is_compiled_with_cuda():
    fluid.set_flags({'FLAGS_cudnn_deterministic': True})


def train(to_static, build_strategy=None):
    """
    Tests model decorated by `dygraph_to_static_output` in static mode. For users, the model is defined in dygraph mode and trained in static mode.
    """
    with fluid.dygraph.guard(place):
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        resnet = ResNet()
        if to_static:
            resnet = paddle.jit.to_static(resnet, build_strategy=build_strategy)
        optimizer = optimizer_setting(parameter_list=resnet.parameters())
        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        for epoch in range(epoch_num):
            total_loss = 0.0
            total_acc1 = 0.0
            total_acc5 = 0.0
            total_sample = 0

            for batch_id in range(100):
                start_time = time.time()
                img = paddle.to_tensor(
                    np.random.random([batch_size, 3, 224, 224]).astype(
                        'float32'
                    )
                )
                label = paddle.to_tensor(
                    np.random.randint(0, 100, [batch_size, 1], dtype='int64')
                )
                img.stop_gradient = True
                label.stop_gradient = True

                with paddle.amp.auto_cast():
                    pred = resnet(img)
                    # FIXME(Aurelius84): The followding cross_entropy seems to bring out a
                    # precision problem, need to figure out the underlying reason.
                    # If we remove it, the loss between dygraph and dy2stat is exactly same.
                    loss = paddle.nn.functional.cross_entropy(
                        input=pred,
                        label=label,
                        reduction='none',
                        use_softmax=False,
                    )
                avg_loss = paddle.mean(x=pred)
                acc_top1 = paddle.static.accuracy(input=pred, label=label, k=1)
                acc_top5 = paddle.static.accuracy(input=pred, label=label, k=5)

                scaled = scaler.scale(avg_loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)
                resnet.clear_gradients()

                total_loss += avg_loss
                total_acc1 += acc_top1
                total_acc5 += acc_top5
                total_sample += 1

                end_time = time.time()
                if batch_id % 2 == 0:
                    print(
                        "epoch %d | batch step %d, loss %0.3f, acc1 %0.3f, acc5 %0.3f, time %f"
                        % (
                            epoch,
                            batch_id,
                            total_loss.numpy() / total_sample,
                            total_acc1.numpy() / total_sample,
                            total_acc5.numpy() / total_sample,
                            end_time - start_time,
                        )
                    )
                if batch_id == 10:
                    break

    return total_loss.numpy()


class TestResnet(unittest.TestCase):
    def train(self, to_static):
        program_translator.enable(to_static)
        return train(to_static)

    def test_resnet(self):
        static_loss = self.train(to_static=True)
        dygraph_loss = self.train(to_static=False)
        np.testing.assert_allclose(
            static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg='static_loss: {} \n dygraph_loss: {}'.format(
                static_loss, dygraph_loss
            ),
        )


if __name__ == '__main__':
    unittest.main()
