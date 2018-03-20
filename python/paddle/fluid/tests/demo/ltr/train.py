import numpy as np
import sys
import os
import argparse
import time

import paddle.v2 as paddle
import paddle.fluid as fluid

from config import TrainConfig as conf


def half_ranknet(name_prefix, input_dim):
   """
   """
   data = fluid.layers.data(name=name_prefix + "_data", shape=[input_dim], dtype="float") 	
   hd1 = fluid.layers.fc(input=data, name=name_prefix + "_hidden", size=10)
   output = fluid.layers.fc(input=hd1, name=name_prefix + "_score", size=1)
   return data, output

def ranknet(input_dim):


    label = fluid.layers.data(name="label", shape=[1], dtype="float")

    data_left, output_left = half_ranknet("left", input_dim)
    data_right, output_right = half_ranknet("right", input_dim)
    
    cost = fluid.layers.rank_cost(left=output_left, right=output_right, label=label) 
    avg_cost = fluid.layers.mean(x=cost)

    return label, data_left, data_right, cost, avg_cost


def main():
    label, data1, data2, cost, avg_cost = ranknet(conf.input_dim)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=conf.learning_rate)
    sgd_optimizer.minimize(avg_cost)


    accuracy = fluid.layers.accuracy(input=cost, label=label)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program([avg_cost, accuracy])

    # The training data set.
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mq2007.train, buf_size=512),
        batch_size=conf.batch_size)

    # The testing data set.
    test_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mq2007.test, buf_size=512),
        batch_size=conf.batch_size)

    if conf.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[label, data1, data2], place=place)

    exe.run(fluid.default_startup_program())
    with open("main.proto", "w") as f:
        f.write(str(fluid.default_main_program()))
    with open("startup.proto", "w") as f:
        f.write(str(fluid.default_startup_program()))

    total_time = 0.
    for pass_id in xrange(conf.num_passes):
        start_time = time.time()
        for batch_id, data in enumerate(train_reader()):
            cost_val, acc_val = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_cost])
            pass_acc = 0#accuracy.eval(exe)
            if batch_id and batch_id % conf.log_period == 0:
                print("Pass id: %d, batch id: %d, cost: %f, pass_acc %f" %
                      (pass_id, batch_id, cost_val, pass_acc))
        end_time = time.time()
        total_time += (end_time - start_time)
        pass_test_acc = 0
        print("Pass id: %d, test_acc: %f" % (pass_id, pass_test_acc))
    print("Total train time: %f" % (total_time))


if __name__ == '__main__':
    main()
