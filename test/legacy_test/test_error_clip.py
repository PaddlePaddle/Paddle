#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import paddle
from paddle import base

BATCH_SIZE = 128
CLIP_MAX = 2e-6
CLIP_MIN = -1e-6

paddle.enable_static()
prog = base.framework.Program()

with base.program_guard(main_program=prog):
    image = paddle.static.data(name='x', shape=[-1, 784], dtype='float32')

    hidden1 = paddle.static.nn.fc(x=image, size=128, activation='relu')
    hidden2 = paddle.static.nn.fc(x=hidden1, size=64, activation='relu')
    predict = paddle.static.nn.fc(x=hidden2, size=10, activation='softmax')

    label = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')

    cost = paddle.nn.functional.cross_entropy(
        input=predict, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(cost)

prog_clip = prog.clone()
prog_clip.block(0).var(hidden1.name)._set_error_clip(
    paddle.nn.clip.ErrorClipByValue(max=CLIP_MAX, min=CLIP_MIN)
)

avg_cost_clip = prog_clip.block(0).var(avg_cost.name)
base.backward.append_backward(loss=avg_cost)
base.backward.append_backward(
    loss=avg_cost_clip, callbacks=[paddle.nn.clip.error_clip_callback]
)

hidden1_grad = prog.block(0).var(hidden1.name + "@GRAD")
hidden1_grad_clip = prog_clip.block(0).var(hidden1.name + "@GRAD")

hidden2_grad = prog.block(0).var(hidden2.name + "@GRAD")
hidden2_grad_clip = prog_clip.block(0).var(hidden2.name + "@GRAD")

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE,
)

place = base.CPUPlace()
exe = base.Executor(place)
feeder = base.DataFeeder(feed_list=[image, label], place=place)
exe.run(base.default_startup_program())

count = 0
for data in train_reader():
    count += 1
    if count > 5:
        break
    out1, out2 = exe.run(
        prog, feed=feeder.feed(data), fetch_list=[hidden1_grad, hidden2_grad]
    )
    out1_clip, out2_clip = exe.run(
        prog_clip,
        feed=feeder.feed(data),
        fetch_list=[hidden1_grad_clip, hidden2_grad_clip],
    )
    if not (
        (out1.clip(min=CLIP_MIN, max=CLIP_MAX) == out1_clip).all()
        and (out2 == out2_clip).all()
    ):
        sys.exit(1)

sys.exit(0)
