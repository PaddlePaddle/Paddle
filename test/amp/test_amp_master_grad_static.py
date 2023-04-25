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
from paddle.distributed import fleet

paddle.enable_static()


def init_dist_strategy(
    use_distributed, use_amp=False, use_pure_fp16=False, use_bf16=False
):
    if not use_distributed:
        return None

    dist_strategy = fleet.DistributedStrategy()
    dist_strategy.nccl_comm_num = 1
    dist_strategy.gradient_merge = True
    dist_strategy.gradient_merge_configs = {"k_steps": 2, "avg": False}
    dist_strategy.amp = use_amp
    if use_amp:
        dist_strategy.amp_configs = {
            "use_dynamic_loss_scaling": True,
            "use_fp16_guard": False,
            "use_pure_fp16": use_pure_fp16,
            "use_bf16": use_bf16,
        }
    return dist_strategy


def exclude_from_weight_decay(name):
    if not isinstance(name, str):
        name = name.name
    if name.find("layer_norm") > -1:
        return True
    bias_suffix = ["_bias", "_b", ".b_0"]
    for suffix in bias_suffix:
        if name.endswith(suffix):
            return True
    return False


def optimization(use_distributed, use_amp, use_pure_fp16, use_bf16):

    dist_strategy = init_dist_strategy(
        use_distributed, use_amp, use_pure_fp16, use_bf16
    )

    multi_precision = True
    master_grad = True
    clip_norm_thres = 1.0
    clip = paddle.nn.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres)
    scheduled_lr = 0.004
    beta1 = 0.78
    beta2 = 0.836
    epsilon = 1e-4
    weight_decay = 0.01
    optimizer = paddle.optimizer.AdamW(
        learning_rate=scheduled_lr,
        grad_clip=clip,  # fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm_thres),
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        multi_precision=multi_precision,
    )

    if dist_strategy:
        optimizer = fleet.distributed_optimizer(optimizer, dist_strategy)
    elif use_amp:
        optimizer = paddle.static.amp.decorate(
            optimizer,
            init_loss_scaling=1.0,
            use_dynamic_loss_scaling=False,
            use_pure_fp16=use_pure_fp16,
            use_fp16_guard=False,
            use_bf16=use_bf16,
            use_master_grad=master_grad,
        )

    return optimizer


def model(data):
    conv2d = paddle.static.nn.conv2d(input=data, num_filters=6, filter_size=3)
    bn_out = paddle.static.nn.batch_norm(input=conv2d)
    out = paddle.nn.functional.softmax(bn_out)
    return out


def main(
    use_distributed=False, use_amp=False, use_pure_fp16=False, use_bf16=False
):
    if use_distributed:
        fleet.init(is_collective=True)

    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        data = paddle.static.data(
            name='X', shape=[None, 1, 28, 28], dtype='float32'
        )
        # input = data.astype('float32') # ('bfloat16')
        out = model(data)
        loss = paddle.mean(out)
        optimizer = optimization(
            use_distributed, use_amp, use_pure_fp16, use_bf16
        )
        optimizer.minimize(loss)

    print(train_program)

    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    print("===== Run startup_program ======")
    print(startup_program)
    exe.run(startup_program)
    if use_amp and use_pure_fp16:
        print("======= Run amp_init =======")
        optimizer.amp_init(place)

    saved_model_name = "test.fp32"
    if use_amp:
        if use_pure_fp16:
            saved_model_name = "test.fp16_o2"
        else:
            saved_model_name = "test.fp16_o1"
    paddle.static.save(train_program, 'models/' + saved_model_name)

    exe.run(train_program)


use_distributed = False
use_amp = True
use_pure_fp16 = True
use_bf16 = True
main(use_distributed, use_amp, use_pure_fp16, use_bf16)
