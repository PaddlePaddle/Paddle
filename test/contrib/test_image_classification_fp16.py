#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import math
import os
import sys
import tempfile
import unittest

import numpy

# TODO: remove sys.path.append
sys.path.append("../legacy_test")
import nets

import paddle
from paddle import base
from paddle.framework import in_pir_mode
from paddle.nn import Layer
from paddle.static.amp import decorate

paddle.enable_static()


def img_conv_group_pir(
    input,
    in_channels,
    out_channels,
    conv_num_filter,
    kernel_size,
    pool_size,
    pool_stride=1,
    pool_padding=0,
    pool_type='max',
    global_pooling=False,
    conv_with_batchnorm=False,
    conv_batchnorm_drop_rate=0.0,
    conv_stride=1,
    conv_padding=1,
    conv_filter_size=3,
    conv_dilation=1,
    conv_groups=1,
    param_attr=None,
    bias_attr=None,
    conv_act=None,
    use_cudnn=True,
):
    tmp = input
    assert isinstance(conv_num_filter, (list, tuple))

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            assert len(obj) == len(conv_num_filter)
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    param_attr = __extend_list__(param_attr)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in range(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        conv = paddle.nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=conv_stride,
            padding=conv_padding[i],
            dilation=conv_dilation,
            groups=conv_groups,
            bias_attr=bias_attr,
        )
        conv_out = conv(input)

        if conv_with_batchnorm[i]:
            batch_norm = paddle.nn.BatchNorm(in_channels, act=conv_act)
            tmp = batch_norm(tmp)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = paddle.nn.functional.dropout(x=tmp, p=drop_rate)

    if pool_type == 'max':
        pool_out = paddle.nn.functional.max_pool2d(
            x=tmp,
            kernel_size=pool_size,
            stride=pool_stride,
        )
    else:
        pool_out = paddle.nn.functional.avg_pool2d(
            x=tmp,
            kernel_size=pool_size,
            stride=pool_stride,
        )
    return pool_out


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(
        input,
        ch_out,
        filter_size,
        stride,
        padding,
        act='relu',
        bias_attr=False,
    ):
        if in_pir_mode():
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
        else:
            tmp = paddle.static.nn.conv2d(
                input=input,
                filter_size=filter_size,
                num_filters=ch_out,
                stride=stride,
                padding=padding,
                act=None,
                bias_attr=bias_attr,
            )
            return paddle.static.nn.batch_norm(input=tmp, act=act)

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
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = paddle.nn.functional.avg_pool2d(x=res3, kernel_size=8, stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        if in_pir_mode():
            return img_conv_group_pir(
                input,
                in_channels=3,
                out_channels=num_filter,
                conv_num_filter=[num_filter] * groups,
                kernel_size=3,
                pool_size=2,
                pool_stride=2,
                pool_padding=0,
                pool_type='max',
                conv_act='relu',
                conv_with_batchnorm=True,
            )
        else:
            return nets.img_conv_group(
                input=input,
                pool_size=2,
                pool_stride=2,
                conv_num_filter=[num_filter] * groups,
                conv_filter_size=3,
                conv_act='relu',
                conv_with_batchnorm=True,
                conv_batchnorm_drop_rate=dropouts,
                pool_type='max',
            )

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = paddle.nn.functional.dropout(x=conv5, p=0.5)
    fc1 = paddle.static.nn.fc(x=drop, size=4096, activation=None)
    if in_pir_mode():
        batch_norm = paddle.nn.BatchNorm(4096)
        bn = batch_norm(fc1)
    else:
        bn = paddle.static.nn.batch_norm(input=fc1, act='relu')
    drop2 = paddle.nn.functional.dropout(x=bn, p=0.5)
    fc2 = paddle.static.nn.fc(x=drop2, size=4096, activation=None)
    return fc2


def train(net_type, use_cuda, save_dirname, is_local):
    classdim = 10
    data_shape = [3, 32, 32]

    train_program = paddle.static.Program()
    startup_prog = paddle.static.Program()
    paddle.seed(123)
    with base.program_guard(train_program, startup_prog):
        images = paddle.static.data(
            name='pixel', shape=[-1, *data_shape], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

        if net_type == "vgg":
            print("train vgg net")
            net = vgg16_bn_drop(images)
        elif net_type == "resnet":
            print("train resnet")
            net = resnet_cifar10(images, 32)
        else:
            raise ValueError(f"{net_type} network is not supported")

        optimizer = paddle.optimizer.Lamb(learning_rate=0.001)

        if in_pir_mode():

            class layer(Layer):
                def __init__(self, classdim, act):
                    super().__init__()
                    self.classdim = classdim
                    self.act = act

                def forward(self, x):
                    logits = paddle.static.nn.fc(
                        x=x, size=self.classdim, activation=self.act
                    )
                    (
                        cost,
                        predict,
                    ) = paddle.nn.functional.softmax_with_cross_entropy(
                        logits, label, return_softmax=True
                    )
                    return cost, predict

            model = layer(classdim, "softmax")
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level="O2",
                dtype='float16',
            )
            scaler = paddle.amp.GradScaler(
                init_loss_scaling=8.0, use_dynamic_loss_scaling=True
            )

            with paddle.amp.auto_cast(
                enable=True,
                level='O2',
                dtype='float16',
                custom_black_list={'transpose2', 'concat'},
                use_promote=True,
            ):
                cost, predict = model(net)
                avg_cost = paddle.mean(cost)
                acc = paddle.static.accuracy(input=predict, label=label)
            # Test program
            value_map = paddle.pir.IrMapping()
            test_program = train_program.clone(value_map)
            fetch_list = []
            fetch_list.append(value_map.look_up(avg_cost))
            fetch_list.append(value_map.look_up(acc))

            scaled = scaler.scale(avg_cost)
            scaler.minimize(optimizer, scaled, startup_program=startup_prog)
            loss_scaling = optimizer.get_loss_scaling()
            scaled_loss = optimizer.get_scaled_loss()
        else:
            logits = paddle.static.nn.fc(
                x=net, size=classdim, activation="softmax"
            )
            cost, predict = paddle.nn.functional.softmax_with_cross_entropy(
                logits, label, return_softmax=True
            )
            avg_cost = paddle.mean(cost)
            acc = paddle.static.accuracy(input=predict, label=label)
            # Test program
            test_program = train_program.clone(for_test=True)
            fetch_list = [avg_cost, acc]
            amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
                custom_black_varnames={"loss", "conv2d_0.w_0"}
            )
            mp_optimizer = decorate(
                optimizer=optimizer,
                amp_lists=amp_lists,
                init_loss_scaling=8.0,
                use_dynamic_loss_scaling=True,
            )

            mp_optimizer.minimize(avg_cost)
            loss_scaling = mp_optimizer.get_loss_scaling()
            scaled_loss = mp_optimizer.get_scaled_loss()

    BATCH_SIZE = 128
    PASS_NUM = 1

    # no shuffle for unit test
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=BATCH_SIZE
    )

    test_reader = paddle.batch(
        paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE
    )

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)
    feeder = base.DataFeeder(place=place, feed_list=[images, label])

    def train_loop(main_program):
        exe.run(startup_prog)
        loss = 0.0
        for pass_id in range(PASS_NUM):
            for batch_id, data in enumerate(train_reader()):
                np_scaled_loss, loss = exe.run(
                    main_program,
                    feed=feeder.feed(data),
                    fetch_list=[scaled_loss, avg_cost],
                )
                print(
                    f'PassID {pass_id:1}, BatchID {batch_id + 1:04}, train loss {float(loss):2.4}, scaled train loss {float(np_scaled_loss):2.4}'
                )
                if (batch_id % 10) == 0:
                    acc_list = []
                    avg_loss_list = []
                    for tid, test_data in enumerate(test_reader()):
                        loss_t, acc_t = exe.run(
                            program=test_program,
                            feed=feeder.feed(test_data),
                            fetch_list=fetch_list,
                        )
                        if math.isnan(float(loss_t)):
                            sys.exit("got NaN loss, training failed.")
                        acc_list.append(float(acc_t))
                        avg_loss_list.append(float(loss_t))
                        break  # Use 1 segment for speeding up CI

                    acc_value = numpy.array(acc_list).mean()
                    avg_loss_value = numpy.array(avg_loss_list).mean()

                    print(
                        f'PassID {pass_id:1}, BatchID {batch_id + 1:04}, test loss {float(avg_loss_value):2.2}, acc {float(acc_value):2.2}'
                    )

                    if acc_value > 0.08:  # Low threshold for speeding up CI
                        paddle.static.io.save_inference_model(
                            save_dirname,
                            images,
                            [predict],
                            exe,
                            program=train_program,
                            clip_extra=True,
                        )
                        return

    if is_local:
        train_loop(train_program)
    else:
        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        training_role = os.getenv("PADDLE_TRAINING_ROLE", "TRAINER")
        t = paddle.distributed.transpiler.DistributeTranspiler()
        t.transpile(trainer_id, pservers=pserver_endpoints, trainers=trainers)
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(
                current_endpoint, pserver_prog
            )
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            train_loop(t.get_trainer_program())


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)

    inference_scope = base.core.Scope()
    with base.scope_guard(inference_scope):
        # Use paddle.static.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(save_dirname, exe)

        # The input's dimension of conv should be 4-D or 5-D.
        # Use normalized image pixels as input data, which should be in the range [0, 1.0].
        batch_size = 1
        tensor_img = numpy.random.rand(batch_size, 3, 32, 32).astype("float32")

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )

        print("infer results: ", results[0])

        paddle.static.save_inference_model(
            save_dirname,
            feed_target_names,
            fetch_targets,
            exe,
            program=inference_program,
            clip_extra=True,
        )


class TestImageClassification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def main(self, net_type, use_cuda, is_local=True):
        if use_cuda and not base.core.is_compiled_with_cuda():
            return

        # Directory for saving the trained model
        save_dirname = os.path.join(
            self.temp_dir.name,
            "image_classification_" + net_type + ".inference.model",
        )

        train(net_type, use_cuda, save_dirname, is_local)
        # infer(use_cuda, save_dirname)

    def test_amp_lists(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists()
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_1(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 1. w={'exp}, b=None
        white_list.add('exp')
        black_list.remove('exp')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'exp'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_2(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 2. w={'tanh'}, b=None
        white_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'tanh'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_3(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 3. w={'lstm'}, b=None
        white_list.add('lstm')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists({'lstm'})
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_4(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 4. w=None, b={'conv2d'}
        white_list.remove('conv2d')
        black_list.add('conv2d')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'conv2d'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_5(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 5. w=None, b={'tanh'}
        black_list.add('tanh')
        gray_list.remove('tanh')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'tanh'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_6(self):
        white_list = (
            copy.copy(paddle.static.amp.fp16_lists.white_list)
            | paddle.static.amp.fp16_lists._only_supported_fp16_list
        )
        black_list = copy.copy(
            paddle.static.amp.fp16_lists.black_list
            | paddle.static.amp.fp16_lists._extra_black_list
        )
        gray_list = copy.copy(paddle.static.amp.fp16_lists.gray_list)

        # 6. w=None, b={'lstm'}
        black_list.add('lstm')

        amp_lists = paddle.static.amp.AutoMixedPrecisionLists(
            custom_black_list={'lstm'}
        )
        self.assertEqual(amp_lists.white_list, white_list)
        self.assertEqual(amp_lists.black_list, black_list)
        self.assertEqual(amp_lists.gray_list, gray_list)

    def test_amp_lists_7(self):
        # 7. w={'lstm'} b={'lstm'}
        # raise ValueError
        self.assertRaises(
            ValueError,
            paddle.static.amp.AutoMixedPrecisionLists,
            {'lstm'},
            {'lstm'},
        )

    def test_vgg_cuda(self):
        with self.scope_prog_guard():
            self.main('vgg', use_cuda=True)

    def test_resnet_cuda(self):
        with self.scope_prog_guard():
            self.main('resnet', use_cuda=True)

    @contextlib.contextmanager
    def scope_prog_guard(self):
        prog = base.Program()
        startup_prog = base.Program()
        scope = base.core.Scope()
        with base.scope_guard(scope):
            with base.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
