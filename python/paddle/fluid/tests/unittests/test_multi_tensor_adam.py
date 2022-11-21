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
import paddle
import paddle.nn as nn
import numpy as np
from paddle.fluid import core
import paddle.fluid as fluid
import os


class MLPLayer(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size, n):
        super(MLPLayer, self).__init__()
        self.linear_first = nn.Linear(input_size, hidden_size)
        self.linear_mid = nn.Sequential(
            ('l0', nn.Linear(hidden_size, hidden_size))
        )
        for i in range(n - 1):
            self.linear_mid.add_sublayer(
                'l' + str(i + 1), nn.Linear(hidden_size, hidden_size)
            )
        self.linear_last = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_first(x)
        x = self.linear_mid(x)
        x = self.linear_last(x)
        return x.mean()


class TestMultiTensorAdam(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.input_size = 8
        self.hidden_size = 5
        self.output_size = 7
        self.n = 10

    def get_adam_or_adamw_out(
        self,
        use_multi_tensor_adam,
        use_adamw,
        test_dict,
        test_fp16,
        test_lrscheduler,
    ):

        paddle.seed(10)
        np.random.seed(10)

        model = MLPLayer(
            self.input_size, self.hidden_size, self.output_size, self.n
        )

        if test_fp16:
            model = paddle.amp.decorate(models=model, level='O2')
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            inp = np.random.random((10, self.input_size)).astype("float16")
            inp = paddle.to_tensor(inp)
        else:
            inp = paddle.uniform([10, self.input_size], dtype="float32")

        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")

        paramters_dict_list = []

        input_paramters = model.parameters()
        learning_rate = 0.01
        if test_lrscheduler:
            learning_rate = paddle.optimizer.lr.LinearWarmup(
                learning_rate=0.5,
                warmup_steps=20,
                start_lr=0,
                end_lr=0.5,
                verbose=True,
            )

        if test_dict:
            i = 0
            for param in model.parameters():
                paramters_dict_list.append(
                    {
                        'params': param,
                        'weight_decay': 0.001 * i,
                        'beta1': 0.80 + 0.001 * i,
                    }
                )
                if not test_lrscheduler:
                    paramters_dict_list[-1]['learning_rate'] = 0.1 * i
                i = i + 1

                input_paramters = paramters_dict_list

        multi_precision = True if test_fp16 else False
        if use_multi_tensor_adam:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=learning_rate,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                    multi_precision=multi_precision,
                    use_multi_tensor=True,
                )
            else:
                opt = paddle.incubate.optimizer.MultiTensorAdamW(
                    learning_rate=learning_rate,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                    multi_precision=multi_precision,
                )
        else:
            if not use_adamw:
                opt = paddle.optimizer.Adam(
                    learning_rate=learning_rate,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                    multi_precision=multi_precision,
                )
            else:
                opt = paddle.optimizer.AdamW(
                    learning_rate=learning_rate,
                    parameters=input_paramters,
                    weight_decay=0.01,
                    beta1=beta1,
                    beta2=beta2,
                    multi_precision=multi_precision,
                )

        num_batch = 2
        if not test_fp16:
            for _ in range(num_batch):
                out = model(inp)
                out.backward()
                opt.step()
                opt.clear_grad()
                if test_lrscheduler:
                    learning_rate.step()
        else:
            for _ in range(num_batch):
                with paddle.amp.auto_cast(level='O2'):
                    out = model(inp)
                scaled = scaler.scale(out)
                scaled.backward()
                scaler.step(opt)
                scaler.update()
                opt.clear_grad()
                if test_lrscheduler:
                    learning_rate.step()

        return model.parameters()

    def run_adam_or_adamw(
        self, use_adamw, test_dict, test_fp16, test_lrscheduler
    ):
        use_multi_tensor_adam = True
        parameters_1 = self.get_adam_or_adamw_out(
            not use_multi_tensor_adam,
            use_adamw,
            test_dict,
            test_fp16,
            test_lrscheduler,
        )
        parameters = self.get_adam_or_adamw_out(
            use_multi_tensor_adam,
            use_adamw,
            test_dict,
            test_fp16,
            test_lrscheduler,
        )
        self.assertEqual(len(parameters), len(parameters_1))
        for i, j in zip(parameters, parameters_1):
            if os.name == 'nt':
                atol = 10
                rtol = 10
            else:
                atol = 1e-3 if test_fp16 else 1e-6
                rtol = 1e-2 if test_fp16 else 1e-6
            np.testing.assert_allclose(
                i.numpy(), j.numpy(), rtol=rtol, atol=atol
            )

    def test_main(self):
        old_device = paddle.get_device()
        for use_gpu in [False, True]:
            if use_gpu and not paddle.is_compiled_with_cuda():
                continue
            if use_gpu:
                paddle.set_device("gpu")
            else:
                paddle.set_device("cpu")
            for use_adamw in [True, False]:
                for test_dict in [True, False]:
                    for test_lrscheduler in [True, False]:
                        test_fp16 = False
                        print(
                            f'use_gpu = {use_gpu} use_adamw = {use_adamw} test_dict = {test_dict} test_fp16 = {test_fp16}'
                        )
                        self.run_adam_or_adamw(
                            use_adamw, test_dict, test_fp16, test_lrscheduler
                        )
                        if use_gpu:
                            test_fp16 = True
                            self.run_adam_or_adamw(
                                use_adamw,
                                test_dict,
                                test_fp16,
                                test_lrscheduler,
                            )
        paddle.set_device(old_device)


class TestStaticMultiTensorAdam(unittest.TestCase):
    def _test(
        self,
        place,
    ):

        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 10
        paddle.seed(SEED)
        np.random.seed(SEED)

        a_np = np.random.random(size=(2, 2)).astype('float32')
        b_np = np.random.random(size=(2, 2)).astype('float32')
        label_np = np.random.randint(2, size=(2, 1)).astype('int64')

        weight_attr1 = paddle.ParamAttr(
            name="weight1",
            initializer=fluid.initializer.Constant(value=1.0),
            trainable=True,
        )
        weight_attr2 = paddle.ParamAttr(
            name="weight2",
            initializer=fluid.initializer.Constant(value=2.0),
            trainable=True,
        )
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)

        with paddle.static.program_guard(main_prog, startup_prog):
            with paddle.utils.unique_name.guard():
                a = paddle.static.data(name="a", shape=[2, 2], dtype='float32')
                b = paddle.static.data(name="b", shape=[2, 2], dtype='float32')
                label = paddle.static.data(
                    name="label", shape=[2, 1], dtype='int64'
                )

                sum = paddle.add(a, b)
                z = paddle.pow(sum, 2.0)

                fc_1 = fluid.layers.fc(input=z, size=2, param_attr=weight_attr1)
                prediction = fluid.layers.fc(
                    input=fc_1, size=2, param_attr=weight_attr2, act='softmax'
                )

                cost = fluid.layers.cross_entropy(input=prediction, label=label)
                loss = fluid.layers.reduce_mean(cost)
                beta1_init = 0.9
                beta2_init = 0.999
                epsilon_init = 1e-8
                beta1 = beta1_init
                beta2 = beta2_init
                epsilon = epsilon_init
                multi_tensor_adam = paddle.optimizer.Adam(
                    learning_rate=0.01,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    grad_clip=clip,
                    use_multi_tensor=True,
                )

                multi_tensor_adam.minimize(loss)

        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            print("Start run on {}".format(place))
            for epoch in range(10):
                pred_res, loss_res = exe.run(
                    main_prog,
                    feed={"a": a_np, "b": b_np, "label": label_np},
                    fetch_list=[prediction, loss],
                )
                print(
                    "Epoch {} | Prediction[0]: {}, Loss: {}".format(
                        epoch, pred_res[0], loss_res
                    )
                )
            paddle.disable_static()
            return pred_res, loss_res

    def _test_with_place(self, place):
        preds = []
        losses = []
        pred, loss = self._test(
            place,
        )
        preds.append(pred)
        losses.append(loss)
        for pred in preds:
            np.testing.assert_allclose(pred, preds[0], rtol=1e-05)
        for loss in losses:
            np.testing.assert_allclose(loss, losses[0], rtol=1e-05)

    def test_adam_api(self):
        # NOTE(zhiqiu): cpu and gpu has different seed, so should compare separatly.
        self._test_with_place(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self._test_with_place(paddle.CUDAPlace(0))


if __name__ == "__main__":
    unittest.main()
