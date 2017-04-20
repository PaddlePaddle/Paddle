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
# limitations under the License

import sys

import paddle.v2 as paddle

from api_v2_vgg import vgg_bn_drop


def main():
    datadim = 3 * 32 * 32
    classdim = 10

    # PaddlePaddle init
    paddle.init(use_gpu=False, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(datadim))

    # Add neural network config
    # option 1. resnet
    # net = resnet_cifar10(image, depth=32)
    # option 2. vgg
    net = vgg_bn_drop(image)

    out = paddle.layer.fc(input=net,
                          size=classdim,
                          act=paddle.activation.Softmax())

    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(classdim))
    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)

    # Create optimizer
    momentum_optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
        learning_rate=0.1 / 128.0,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=50000 * 100,
        learning_rate_schedule='discexp',
        batch_size=128)

    # End batch and end pass event handler
    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "\nPass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            result = trainer.test(
                reader=paddle.batch(
                    paddle.dataset.cifar.test10(), batch_size=128),
                feeding={'image': 0,
                         'label': 1})
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=momentum_optimizer)
    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.cifar.train10(), buf_size=50000),
            batch_size=128),
        num_passes=5,
        event_handler=event_handler,
        feeding={'image': 0,
                 'label': 1})


if __name__ == '__main__':
    main()
