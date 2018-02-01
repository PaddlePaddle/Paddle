#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

import sys

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import numpy


def resnet_cifar10(input, depth=32):
    def conv_bn_layer(input, ch_out, filter_size, stride, padding, act='relu'):
        tmp = fluid.layers.conv2d(
            input=input,
            filter_size=filter_size,
            num_filters=ch_out,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=False)
        return fluid.layers.batch_norm(input=tmp, act=act)

    def shortcut(input, ch_in, ch_out, stride):
        if ch_in != ch_out:
            return conv_bn_layer(input, ch_out, 1, stride, 0, None)
        else:
            return input

    def basicblock(input, ch_in, ch_out, stride):
        tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
        tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None)
        short = shortcut(input, ch_in, ch_out, stride)
        return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')

    def layer_warp(block_func, input, ch_in, ch_out, count, stride):
        tmp = block_func(input, ch_in, ch_out, stride)
        for i in range(1, count):
            tmp = block_func(tmp, ch_out, ch_out, 1)
        return tmp

    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(
        input=input, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    return pool


def vgg16_bn_drop(input):
    def conv_block(input, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=input,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    return fc2


classdim = 10
data_shape = [3, 32, 32]

images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

net_type = "vgg"
if len(sys.argv) >= 2:
    net_type = sys.argv[1]

if net_type == "vgg":
    print("train vgg net")
    net = vgg16_bn_drop(images)
elif net_type == "resnet":
    print("train resnet")
    net = resnet_cifar10(images, 32)
else:
    raise ValueError("%s network is not supported" % net_type)

predict = fluid.layers.fc(input=net, size=classdim, act='softmax')
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)
acc = fluid.layers.accuracy(input=predict, label=label)

# Test program 
test_program = fluid.default_main_program().clone()

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

BATCH_SIZE = 128
PASS_NUM = 1

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.cifar.train10(), buf_size=128 * 10),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
exe.run(fluid.default_startup_program())

# directory for saving model
save_dirname = "image_classification_" + net_type + ".inference.model"
tobreak = False

for pass_id in range(PASS_NUM):
    for batch_id, data in enumerate(train_reader()):
        exe.run(feed=feeder.feed(data))
        #loss, acc = exe.run(fluid.default_main_program(),
        #                    feed=feeder.feed(data),
        #                    fetch_list=[avg_cost] + accuracy.metrics)
        #pass_acc = accuracy.eval(exe)
        print(pass_id, batch_id)
        if (batch_id % 10) == 0:
            acc_list = []
            avg_loss_list = []
            for tid, test_data in enumerate(test_reader()):
                loss_t, acc_t = exe.run(program=test_program,
                                        feed=feeder.feed(test_data),
                                        fetch_list=[avg_cost, acc])
                acc_list.append(float(acc_t))
                avg_loss_list.append(float(loss_t))
                break
            acc_value = numpy.array(acc_list).mean()
            avg_loss_value = numpy.array(avg_loss_list).mean()

            print("loss:" + str(avg_loss_value) + " acc:" + str(acc_value))
            # this model is slow, so if we can train two mini batch, we think it works properly.
            if acc_value > 0.01:
                fluid.io.save_inference_model(save_dirname, ["pixel"],
                                              [predict], exe)
                tobreak = True
                break

    if tobreak == True:
        break


def infer(save_dirname=None):
    if save_dirname is None:
        return

    #place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Use fluid.io.load_inference_model to obtain the inference program desc,
    # the feed_target_names (the names of variables that will be feeded 
    # data using feed operators), and the fetch_targets (variables that 
    # we want to obtain data from using fetch operators).
    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)

    #save for checking
    curstr = inference_program.to_string(True)
    f = open("check_pd.txt", 'w')
    f.write(curstr)
    f.close()

    # The input's dimension of conv should be 4-D or 5-D.
    tensor_img = numpy.random.rand(1, 3, 32, 32).astype("float32")

    # Construct feed as a dictionary of {feed_target_name: feed_target_data}
    # and results will contain a list of data corresponding to fetch_targets.
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: tensor_img},
                      fetch_list=fetch_targets)
    print("infer results: ", results[0])


print("Calling infer")
infer(save_dirname)
