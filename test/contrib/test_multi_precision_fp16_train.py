#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.io import Dataset
from paddle.nn import Layer
from paddle.static.amp.fp16_utils import cast_model_to_fp16

paddle.enable_static()


class RandomDataset(Dataset):
    def __init__(self, num_samples, seed=123):
        super().__init__()
        np.random.seed(seed)
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([3, 32, 32]).astype('float32')
        label = np.random.randint(0, 9, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


def reader_decorator(reader):
    def __reader__():
        for i in range(len(reader)):
            yield reader[i]

    return __reader__


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(
        input, ch_out, filter_size, stride, padding, act='relu', bias_attr=False
    ):
        conv = paddle.nn.Conv2D(
            in_channels=input.shape[1],
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr,
        )
        tmp = conv(input)
        bn = paddle.nn.BatchNorm(tmp.shape[1], act=act)
        return bn(tmp)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
        short = shortcut(input, ch_in, ch_out, stride)
        return paddle.nn.functional.relu(paddle.add(x=tmp, y=short))

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1
    )
    if paddle.framework.in_pir_mode():
        with paddle.amp.auto_cast(level='O2'):
            res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
            res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
            res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    else:
        with paddle.static.amp.fp16_guard():
            res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
            res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
            res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = paddle.nn.functional.avg_pool2d(x=res3, kernel_size=8, stride=1)
    return pool


def train(use_pure_fp16=True, use_nesterov=False, optimizer=""):
    classdim = 10
    data_shape = [3, 32, 32]
    PASS_NUM = 1

    train_program = base.Program()
    startup_prog = base.Program()
    paddle.seed(123)
    with base.program_guard(train_program, startup_prog):
        images = paddle.static.data(
            name='pixel', shape=[-1, *data_shape], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        net = resnet_cifar10(images)
        if optimizer == "Adam":
            optimizer = paddle.optimizer.AdamW(
                learning_rate=0.001,
                epsilon=1e-8,
                weight_decay=0.0,
                multi_precision=True,
            )
        elif optimizer == "Lars":
            optimizer = paddle.incubate.optimizer.LarsMomentumOptimizer(
                learning_rate=0.001, momentum=0.9, multi_precision=use_pure_fp16
            )
        else:
            optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                momentum=0.9,
                use_nesterov=use_nesterov,
                weight_decay=paddle.regularizer.L2Decay(1e-4),
                multi_precision=use_pure_fp16,
            )
        if paddle.framework.in_pir_mode() and use_pure_fp16:

            class layer(Layer):
                def __init__(self, classdim, act):
                    super().__init__()
                    self.classdim = classdim
                    self.act = act

                def forward(self, x):
                    logits = paddle.static.nn.fc(
                        x=x, size=self.classdim, activation=self.act
                    )
                    cost = paddle.nn.functional.softmax_with_cross_entropy(
                        logits, label, return_softmax=False
                    )
                    return cost

            model = layer(classdim, "softmax")
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level="O2",
                dtype="float16",
            )
            scaler = paddle.amp.GradScaler(
                init_loss_scaling=128.0, use_dynamic_loss_scaling=True
            )
            with paddle.amp.auto_cast(
                enable=True, level="O2", dtype="float16", use_promote=True
            ):
                cost = model(net)
                sum_cost = paddle.sum(cost)
            value_map = paddle.pir.IrMapping()
            test_program = train_program.clone(value_map)
            fetch_list = [value_map.look_up(sum_cost)]
            scaled = scaler.scale(sum_cost)
            scaler.minimize(optimizer, scaled, startup_program=startup_prog)
        else:
            logits = paddle.static.nn.fc(
                x=net, size=classdim, activation="softmax"
            )
            cost = paddle.nn.functional.softmax_with_cross_entropy(
                logits, label, return_softmax=False
            )
            sum_cost = paddle.sum(cost)

            # Test program
            if paddle.framework.in_pir_mode():
                value_map = paddle.pir.IrMapping()
                test_program = train_program.clone(value_map)
                fetch_list = [value_map.look_up(sum_cost)]
            else:
                test_program = train_program.clone(for_test=True)
                fetch_list = [sum_cost]

            if use_pure_fp16:
                optimizer = paddle.static.amp.decorate(
                    optimizer,
                    init_loss_scaling=128.0,
                    use_dynamic_loss_scaling=True,
                    use_pure_fp16=True,
                )

            optimizer.minimize(sum_cost)

    train_reader = paddle.batch(
        reader_decorator(RandomDataset(16 * 5, seed=123)),
        batch_size=16,
        drop_last=True,
    )

    test_reader = paddle.batch(
        reader_decorator(RandomDataset(4 * 5, seed=456)),
        batch_size=4,
        drop_last=True,
    )

    place = base.CUDAPlace(0)
    exe = base.Executor(place)
    feeder = base.DataFeeder(place=place, feed_list=[images, label])

    def train_loop():
        exe.run(startup_prog)
        if use_pure_fp16 and not paddle.framework.in_pir_mode():
            optimizer.amp_init(
                place, test_program=test_program, use_fp16_test=True
            )

        train_loss_list = []
        test_loss_list = []
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                (loss,) = exe.run(
                    train_program, feed=feeder.feed(data), fetch_list=[sum_cost]
                )
                loss_v = float(loss) if isinstance(loss, np.ndarray) else loss
                print(
                    f'PassID {pass_id:1}, Train Batch ID {batch_id + 1:04}, train loss {float(loss_v):2.4}'
                )
                train_loss_list.append(float(loss_v))

            for tid, test_data in enumerate(test_reader()):
                (loss_t,) = exe.run(
                    program=test_program,
                    feed=feeder.feed(test_data),
                    fetch_list=fetch_list,
                )
                test_loss_list.append(float(loss_t))
                print(
                    f'PassID {pass_id:1}, Test Batch ID {tid + 1:04}, test loss {float(loss_t):2.4}'
                )

        return train_loss_list, test_loss_list

    return train_loop()


class TestImageMultiPrecision(unittest.TestCase):
    def test_resnet_pure_fp16(self):
        if not base.core.is_compiled_with_cuda():
            return

        def do_test(use_nesterov=False, optimizer=""):
            if optimizer == "Adam":
                suffix = "use Adam"
            elif optimizer == "Lars":
                suffix = "use Lars"
            else:
                suffix = "with Nesterov" if use_nesterov else "without Nesterov"
            with self.scope_prog_guard():
                print(f"-----------------FP16 Train {suffix}-----------------")
                train_loss_fp16, test_loss_fp16 = train(
                    use_pure_fp16=True,
                    use_nesterov=use_nesterov,
                    optimizer=optimizer,
                )
            with self.scope_prog_guard():
                print(f"-----------------FP32 Train {suffix}-----------------")
                train_loss_fp32, test_loss_fp32 = train(
                    use_pure_fp16=False,
                    use_nesterov=use_nesterov,
                    optimizer=optimizer,
                )

            np.testing.assert_allclose(
                np.array(train_loss_fp16),
                np.array(train_loss_fp32),
                rtol=0.01,
                atol=1e-05,
                equal_nan=True,
                err_msg='Failed to train in pure FP16.',
            )
            np.testing.assert_allclose(
                np.array(test_loss_fp16),
                np.array(test_loss_fp32),
                rtol=0.01,
                atol=1e-05,
                equal_nan=True,
                err_msg='Failed to test in pure FP16.',
            )

        do_test(use_nesterov=False)
        do_test(use_nesterov=True)
        do_test(optimizer="Adam")
        do_test(optimizer="Lars")

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield


class TestAmpWithNonIterableDataLoader(unittest.TestCase):
    def decorate_with_data_loader(self):
        main_prog = paddle.static.Program()
        start_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            with paddle.base.unique_name.guard():
                image = paddle.static.data(
                    name='image', shape=[-1, 3, 224, 224], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[-1, 1], dtype='int64'
                )
                zero_var = paddle.tensor.fill_constant(
                    shape=[1], dtype='int64', value=0
                )
                one_var = paddle.tensor.fill_constant(
                    shape=[1], dtype='int64', value=1
                )
                label_val = paddle.static.nn.cond(
                    label != zero_var, lambda: zero_var, lambda: one_var
                )
                paddle.assign(label_val, output=label)
                net = resnet_cifar10(image)
                logits = paddle.static.nn.fc(
                    x=net, size=10, activation="softmax"
                )

        block = main_prog.global_block()
        for op in block.ops:
            if op.type == "mul":
                op._set_attr('in_dtype', base.core.VarDesc.VarType.FP32)
                op._set_attr('out_dtype', base.core.VarDesc.VarType.FP32)
                op._set_attr('dtype', base.core.VarDesc.VarType.FP32)

        cast_model_to_fp16(main_prog, use_fp16_guard=False)

    def test_non_iterable_dataloader(self):
        if base.core.is_compiled_with_cuda():
            with paddle.pir_utils.OldIrGuard():
                self.decorate_with_data_loader()


if __name__ == '__main__':
    unittest.main()
