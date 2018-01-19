#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.v2 as paddle
import paddle.v2.fluid as fluid


def _get_global_param_norm_(params_grads):
    res = fluid.layers.fill_constant(shape=[1], dtype="float32", value=0.0)
    for _, grad in params_grads:
        norm_var = fluid.layers.reduce_sum(
            input=fluid.layers.pow(x=grad, factor=2.0))
        fluid.layers.sums(input=[norm_var, res], out=[res])
    fluid.layers.sqrt(x=res, out=res)
    return res


BATCH_SIZE = 128
CLIP = 0.5
prog = fluid.framework.Program()

with fluid.program_guard(main_program=prog):
    image = fluid.layers.data(name='x', shape=[784], dtype='float32')

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

    label = fluid.layers.data(name='y', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

prog_clip = prog.clone()

avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

p_g = fluid.backward.append_backward(loss=avg_cost)
p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

with fluid.program_guard(main_program=prog):
    gloabl_norm = _get_global_param_norm_(p_g)

with fluid.program_guard(main_program=prog_clip):
    fluid.clip.gradient_clip_by_global_norm(clip_norm=CLIP)
    p_g_clip = fluid.clip.append_gradient_clip_ops(p_g_clip)
    gloabl_norm_clip = _get_global_param_norm_(p_g_clip)

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
    out, = exe.run(prog, feed=feeder.feed(data), fetch_list=[gloabl_norm])
    out_clip, = exe.run(prog_clip,
                        feed=feeder.feed(data),
                        fetch_list=[gloabl_norm_clip])

    if not np.allclose(out_clip, np.minimum(out, np.array([CLIP]))):
        exit(1)
exit(0)
