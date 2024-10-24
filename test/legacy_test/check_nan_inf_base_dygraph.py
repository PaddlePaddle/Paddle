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

import argparse

import numpy as np

import paddle
from paddle import nn

# os.environ["GLOG_vmodule"] = "nan_inf_utils_detail=10"


paddle.seed(0)
np.random.seed(0)


class TestLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        w_1_np = np.random.random([32, 400]).astype("float32")
        self.linear1 = nn.Linear(
            in_features=32,
            out_features=400,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(w_1_np)
            ),
        )
        w_2_np = np.random.random([400, 10]).astype("float32")
        self.linear2 = nn.Linear(
            in_features=400,
            out_features=10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(w_2_np)
            ),
        )

    def forward(self, x):
        out = self.linear1(x)
        out = nn.functional.sigmoid(out)
        out = self.linear2(out)
        mask = paddle.randint(low=0, high=2, shape=out.shape).astype("float32")
        out = paddle.divide(out, mask)
        out = nn.functional.softmax(out)
        return out


def check_main(use_cuda, use_amp=False):
    paddle.set_device('gpu' if use_cuda else 'cpu')

    model = TestLayer()
    sgd = paddle.optimizer.SGD(
        learning_rate=0.05, parameters=model.parameters()
    )

    if use_cuda and use_amp:
        scaler = paddle.amp.GradScaler()

    x_np = 10000 * np.random.random([128, 32]).astype("float32")

    x = paddle.to_tensor(x_np)
    if use_cuda and use_amp:
        with paddle.amp.auto_cast(enable=True, dtype="float16", level="O1"):
            out = model(x)
            loss = paddle.mean(out)
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(sgd, scaled)
    else:
        out = model(x)
        loss = paddle.mean(out)
        loss.backward()
    sgd.step()
    sgd.clear_grad()


def run_check(args):
    paddle.set_flags(
        {
            "FLAGS_check_nan_inf": 1,
            "FLAGS_check_nan_inf_level": args.check_nan_inf_level,
        }
    )
    use_cuda = args.use_cuda and paddle.is_compiled_with_cuda()
    if args.check_nan_inf_level == 0:
        if use_cuda:
            try:
                check_main(use_cuda=True, use_amp=args.use_amp)
                raise AssertionError
            except Exception as e:
                print(e)
                print(type(e))
                # Note. Enforce in cuda kernel may not catch in paddle, and
                # Exception type will be RuntimeError
                assert type(e) == OSError or type(e) == RuntimeError
        else:
            try:
                check_main(use_cuda=False, use_amp=False)
                raise AssertionError
            except Exception as e:
                print(e)
                print(type(e))
                assert type(e) == RuntimeError
    else:
        check_main(use_cuda=use_cuda, use_amp=args.use_amp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--check_nan_inf_level', type=int, default=0)
    args = parser.parse_args()
    run_check(args)
