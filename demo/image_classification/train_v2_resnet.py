# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import paddle.v2 as paddle


def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  active_type=paddle.activation.Relu(),
                  ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)


def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        print("n_in != n_out")
        return conv_bn_layer(ipt, n_out, 1, stride, 0,
                             paddle.activation.Linear())
    else:
        return ipt


def basicblock(ipt, ch_out, stride):
    ch_in = ipt.num_filters
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())


def bottleneck(ipt, ch_out, stride):
    ch_in = ipt.num_filter
    tmp = conv_bn_layer(ipt, ch_out, 1, stride, 0)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1)
    tmp = conv_bn_layer(tmp, ch_out * 4, 1, 1, 0, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out * 4, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())


def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(ipt, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
    return tmp


def resnet_imagenet(ipt, depth=50):
    cfg = {
        18: ([2, 2, 2, 1], basicblock),
        34: ([3, 4, 6, 3], basicblock),
        50: ([3, 4, 6, 3], bottleneck),
        101: ([3, 4, 23, 3], bottleneck),
        152: ([3, 8, 36, 3], bottleneck)
    }
    stages, block_func = cfg[depth]
    tmp = conv_bn_layer(
        ipt, ch_in=3, ch_out=64, filter_size=7, stride=2, padding=3)
    tmp = paddle.layer.img_pool(input=tmp, pool_size=3, stride=2)
    tmp = layer_warp(block_func, tmp, 64, stages[0], 1)
    tmp = layer_warp(block_func, tmp, 128, stages[1], 2)
    tmp = layer_warp(block_func, tmp, 256, stages[2], 2)
    tmp = layer_warp(block_func, tmp, 512, stages[3], 2)
    tmp = paddle.layer.img_pool(
        input=tmp, pool_size=7, stride=1, pool_type=paddle.pooling.Avg())

    tmp = paddle.layer.fc(input=tmp, size=1000, act=paddle.activation.Softmax())
    return tmp


def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(
        ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(
        input=res3, pool_size=8, stride=1, pool_type=paddle.pooling.Avg())
    return pool


def main():
    datadim = 3 * 32 * 32
    classdim = 10

    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))
    net = resnet_cifar10(image, depth=32)
    out = paddle.layer.fc(input=net,
                          size=classdim,
                          act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    parameters = paddle.parameters.create(cost)

    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp',
        batch_size=128)

    trainer = paddle.trainer.SGD(update_equation=momentum_optimizer)
    trainer.train(
        reader=paddle.reader.batched(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size=3072),
            batch_size=128),
        cost=cost,
        num_passes=1,
        parameters=parameters,
        event_handler=event_handler,
        reader_dict={'image': 0,
                     'label': 1}, )


if __name__ == '__main__':
    main()
