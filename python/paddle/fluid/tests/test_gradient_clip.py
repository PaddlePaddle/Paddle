#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 128
CLIP = 1

prog = fluid.framework.Program()
with fluid.program_guard(main_program=prog):
    image = fluid.layers.data(name='x', shape=[784], dtype='float32')

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

    label = fluid.layers.data(name='y', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)

prog_clip = prog.clone()

avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

p_g = fluid.backward.append_backward(loss=avg_cost)
p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

with fluid.program_guard(main_program=prog_clip):
    fluid.clip.set_gradient_clip(
        fluid.clip.GradientClipByGlobalNorm(clip_norm=CLIP))
    p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)

grad_list = [elem[1] for elem in p_g]
grad_clip_list = [elem[1] for elem in p_g_clip]

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
exe.run(fluid.default_startup_program())

count = 0
for data in train_reader():
    count += 1
    if count > 5:
        break
    out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
    out_clip = exe.run(prog_clip,
                       feed=feeder.feed(data),
                       fetch_list=grad_clip_list)
    global_norm = 0
    for v in out[1:]:
        global_norm += np.sum(np.power(v, 2))
    global_norm = np.sqrt(global_norm)

    global_norm_clip = 0
    for v in out_clip[1:]:
        global_norm_clip += np.sum(np.power(v, 2))
    global_norm_clip = np.sqrt(global_norm_clip)

    if not np.isclose(
            a=global_norm_clip, b=np.minimum(global_norm, CLIP), rtol=5e-3):
        exit(1)
exit(0)
