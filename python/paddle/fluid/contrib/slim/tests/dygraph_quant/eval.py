# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import reader as reader
import argparse
import functools
from utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('data_dir',         str,  "/dataset/ILSVRC2012/",  "The ImageNet dataset root dir.")
add_arg('inference_model',  str,   "", "The inference model path.")
add_arg('test_samples',     int,  -1,                "Test samples.  if set as -1, eval all test sample")
add_arg('batch_size',       int,  20,                "Batch size.")
# yapf: enable


def eval(args):
    class_dim = args.class_dim
    inference_model = args.inference_model
    data_dir = args.data_dir
    image_shape = [int(m) for m in args.image_shape.split(",")]

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(dirname=inference_model, executor=exe)
    feed_vars = [fluid.framework._get_var(str(var_name), inference_program) \
            for var_name in feed_target_names]
    print('--------------------------input--------------------------')
    for i in feed_vars:
        print(i.name)
    print('--------------------------output--------------------------')
    for o in fetch_targets:
        print(o.name)
    val_reader = paddle.batch(
        reader.val(data_dir=data_dir), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=feed_vars)

    total = 0
    correct = 0
    correct_5 = 0
    for batch_id, data in enumerate(val_reader()):
        labels = []
        in_data = []
        for dd in data:
            labels.append(dd[1])
            in_data.append([np.array(dd[0])])
        t1 = time.time()
        fetch_out = exe.run(inference_program,
                            fetch_list=fetch_targets,
                            feed=feeder.feed(in_data))
        t2 = time.time()
        for i in range(len(labels)):
            label = labels[i]
            result = np.array(fetch_out[0][i])
            index = result.argsort()
            top_1_index = index[-1]
            top_5_index = index[-5:]
            total += 1
            if top_1_index == label:
                correct += 1
            if label in top_5_index:
                correct_5 += 1

        if batch_id % 10 == 0:
            acc1 = float(correct) / float(total)
            acc5 = float(correct_5) / float(total)
            period = t2 - t1
            print("Testbatch {0}, "
                  "acc1 {1}, acc5 {2}, time {3}".format(batch_id, \
                  acc1, acc5, "%2.2f sec" % period))
            sys.stdout.flush()
        if args.test_samples > 0 and \
            batch_id * args.batch_size > args.test_samples:
            break
    total_acc1 = float(correct) / float(total)
    total_acc5 = float(correct_5) / float(total)
    print("End test: test_acc1 {0}, test_acc5 {1}".format(total_acc1,
                                                          total_acc5))
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
