#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import gzip

import paddle.v2.dataset.cifar as cifar
import paddle.v2 as paddle
import time
import os

DATA_DIM = 3 * 32 * 32
CLASS_DIM = 10
BATCH_SIZE = os.getenv("BATCH_SIZE")
if BATCH_SIZE:
    BATCH_SIZE = int(BATCH_SIZE)
else:
    BATCH_SIZE = 128
print "batch_size", BATCH_SIZE
NODE_COUNT = int(os.getenv("TRAINERS"))
ts = 0


def vgg(input, nums, class_dim):
    def conv_block(input, num_filter, groups, num_channels=None):
        return paddle.networks.img_conv_group(
            input=input,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            pool_type=paddle.pooling.Max())

    assert len(nums) == 5
    # the channel of input feature is 3
    conv1 = conv_block(input, 64, nums[0], 3)
    conv2 = conv_block(conv1, 128, nums[1])
    conv3 = conv_block(conv2, 256, nums[2])
    conv4 = conv_block(conv3, 512, nums[3])
    conv5 = conv_block(conv4, 512, nums[4])

    fc_dim = 512
    fc1 = paddle.layer.fc(input=conv5,
                          size=fc_dim,
                          act=paddle.activation.Relu(),
                          layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=fc1,
                          size=fc_dim,
                          act=paddle.activation.Relu(),
                          layer_attr=paddle.attr.Extra(drop_rate=0.5))
    out = paddle.layer.fc(input=fc2,
                          size=class_dim,
                          act=paddle.activation.Softmax())
    return out


def vgg13(input, class_dim):
    nums = [2, 2, 2, 2, 2]
    return vgg(input, nums, class_dim)


def vgg16(input, class_dim):
    nums = [2, 2, 3, 3, 3]
    return vgg(input, nums, class_dim)


def vgg19(input, class_dim):
    nums = [2, 2, 4, 4, 4]
    return vgg(input, nums, class_dim)


def main():
    global ts
    paddle.init(use_gpu=False)
    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    extra_layers = None
    # NOTE: for v2 distributed training need averaging updates.
    learning_rate = 1e-3 / NODE_COUNT
    out = vgg16(image, class_dim=CLASS_DIM)
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                         BATCH_SIZE),
        learning_rate=learning_rate / BATCH_SIZE,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            cifar.train10(),
            # To use other data, replace the above line with:
            # reader.train_reader('train.list'),
            buf_size=1000),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        cifar.test10(),
        # To use other data, replace the above line with:
        # reader.test_reader('val.list'),
        batch_size=BATCH_SIZE)

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 extra_layers=extra_layers,
                                 is_local=False)

    # End batch and end pass event handler
    def event_handler(event):
        global ts, ts_pass
        if isinstance(event, paddle.event.BeginPass):
            ts_pass = time.time()
        if isinstance(event, paddle.event.BeginIteration):
            ts = time.time()
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 1 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s, spent: %f" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    time.time() - ts)
        if isinstance(event, paddle.event.EndPass):
            print "Pass %d end, spent: %f" % (event.pass_id,
                                              time.time() - ts_pass)
            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
