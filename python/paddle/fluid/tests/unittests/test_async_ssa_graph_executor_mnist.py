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

from __future__ import print_function

import os
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64
PASS_NUM = 5


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def convolutional_neural_network(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return loss_net(conv_pool_2, label)


def train(use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    prediction, avg_loss, acc = convolutional_neural_network(img, label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(program=train_test_program,
                                          feed=train_test_feed.feed(test_data),
                                          fetch_list=[acc, avg_loss])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # get test acc and loss
        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe.run(fluid.default_startup_program())
    main_program = fluid.default_main_program()

    exec_strategy = fluid.ExecutionStrategy()
    build_strategy = fluid.BuildStrategy()

    cpu_num = int(os.environ.get('CPU_NUM'))
    thread_num = int(os.getenv("NUM_THREADS"))

    print("cpu_num:" + str(cpu_num))
    print("thread_num:" + str(thread_num))

    build_strategy.async_mode = True

    exec_strategy.num_threads = thread_num
    exec_strategy.num_iteration_per_drop_scope = 1
    exec_strategy.num_iteration_per_run = 10

    pe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=avg_loss.name,
        main_program=main_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    lists = []
    step = 0
    for epoch_id in range(PASS_NUM):
        for step_id, data in enumerate(train_reader()):
            loss_val, acc_val = pe.run(feed=feeder.feed(data),
                                       fetch_list=[avg_loss.name, acc.name])
            loss_val = numpy.mean(loss_val)
            acc_val = numpy.mean(acc_val)
            if step % 100 == 0:
                print("Pass %d, Batch %d, Cost %f" % (epoch_id, step, loss_val))
            step += 1
        # test for epoch
        avg_loss_val, acc_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("Test with Epoch %d, avg_cost: %s, acc: %s" %
              (epoch_id, avg_loss_val, acc_val))
        lists.append((epoch_id, avg_loss_val, acc_val))
        if save_dirname is not None:
            fluid.io.save_inference_model(
                save_dirname, ["img"], [prediction],
                exe,
                model_filename=model_filename,
                params_filename=params_filename)

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))


def infer(use_cuda,
          save_dirname=None,
          model_filename=None,
          params_filename=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    def load_image(file):
        im = Image.open(file).convert('L')
        im = im.resize((28, 28), Image.ANTIALIAS)
        im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
        im = im / 255.0 * 2.0 - 1.0
        return im

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    tensor_img = load_image(cur_dir + '/image/infer_3.png')

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be feeded
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             save_dirname, exe, model_filename, params_filename)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: tensor_img},
                          fetch_list=fetch_targets)
        lab = numpy.argsort(results)
        print("Inference result of image/infer_3.png is: %d" % lab[0][0][-1])


def main(use_cuda):
    model_filename = None
    params_filename = None
    save_dirname = "recognize_digits" + ".inference.model"

    # call train() with is_local argument to run distributed train
    train(
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)
    infer(
        use_cuda=use_cuda,
        save_dirname=save_dirname,
        model_filename=model_filename,
        params_filename=params_filename)


if __name__ == '__main__':
    use_cuda = False
    main(use_cuda=use_cuda)
