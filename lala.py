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


import paddle
import paddle.optimizer as optimizer


class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()

        for i in range(500):
            self.add_sublayer(str(i), paddle.nn.Linear(4, 4))

    def forward(self, inputs):
        for layer in self._sub_layers.values():
            inputs = layer(inputs)
        return inputs


def test_step(optclass):
    print(optclass, "step")
    model = Net()
    if optimizer.Lamb == optclass:
        opt = optclass(learning_rate=0.01, parameters=model.parameters())
    else:
        opt = optclass(
            learning_rate=0.01, parameters=model.parameters(), weight_decay=0.01
        )
    inputs = paddle.randn((4, 4), dtype="float32")
    inputs.stop_gradient = False
    out = model(inputs)
    out = paddle.mean(out)
    out.backward()
    print("begin.....")
    opt.step()
    opt.clear_grad()
    print("end.....")


def test_step_amp(optclass):
    print(optclass, "step AMP")
    model = Net()
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
    if optimizer.Lamb == optclass:
        opt = optclass(learning_rate=0.01, parameters=model.parameters())
    else:
        opt = optclass(
            learning_rate=0.01, parameters=model.parameters(), weight_decay=0.01
        )
    inputs = paddle.randn((4, 4), dtype="float32")
    inputs.stop_gradient = False
    with paddle.amp.amp_guard():
        out = model(inputs)
        out = paddle.mean(out)
    scaled = scaler.scale(out)
    scaled.backward()
    print("begin.....")
    scaler.step(opt)
    scaler.update()
    opt.clear_grad()
    print("end.....")


def test_minimize(optclass):
    print(optclass, "minimize")
    model = Net()
    if optimizer.Lamb == optclass:
        opt = optclass(learning_rate=0.01, parameters=model.parameters())
    else:
        opt = optclass(
            learning_rate=0.01, parameters=model.parameters(), weight_decay=0.01
        )
    inputs = paddle.randn((4, 4), dtype="float32")
    inputs.stop_gradient = False
    out = model(inputs)
    out = paddle.mean(out)
    out.backward()
    print("begin.....")
    opt.minimize(out)
    print("end.....")


def test_minimize_amp(optclass):
    print(optclass, "minimize AMP")
    model = Net()
    scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
    if optimizer.Lamb == optclass:
        opt = optclass(learning_rate=0.01, parameters=model.parameters())
    else:
        opt = optclass(
            learning_rate=0.01, parameters=model.parameters(), weight_decay=0.01
        )
    inputs = paddle.randn((4, 4), dtype="float32")
    inputs.stop_gradient = False
    with paddle.amp.amp_guard():
        out = model(inputs)
        out = paddle.mean(out)
        scaled = scaler.scale(out)
        scaled.backward()
        print("begin.....")
        scaler.minimize(opt, scaled)
        print("end.....")


def test_main(optclass):
    test_step(optclass)
    test_minimize(optclass)
    test_step_amp(optclass)
    test_minimize_amp(optclass)


test_main(optimizer.SGD)
test_main(optimizer.Adam)
test_main(optimizer.AdamW)
test_main(optimizer.Adamax)
test_main(optimizer.Adagrad)
test_main(optimizer.Adadelta)
test_main(optimizer.Momentum)
test_main(optimizer.RMSProp)
test_main(optimizer.Lamb)
