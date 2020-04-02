# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import numpy as np
from scipy.misc import imsave

import paddle.fluid as fluid

from model import Model, Input, set_device
from cyclegan import Generator, GeneratorCombine
import data as data


def main():
    place = set_device(FLAGS.device)
    fluid.enable_dygraph(place) if FLAGS.dynamic else None

    # Generators
    g_AB = Generator()
    g_BA = Generator()
    g = GeneratorCombine(g_AB, g_BA, is_train=False)

    im_shape = [-1, 3, 256, 256]
    input_A = Input(im_shape, 'float32', 'input_A')
    input_B = Input(im_shape, 'float32', 'input_B')
    g.prepare(inputs=[input_A, input_B])
    g.load(FLAGS.init_model, skip_mismatch=True, reset_optimizer=True)

    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    test_data_A = data.TestDataA()
    test_data_B = data.TestDataB()

    for i in range(len(test_data_A)):
        data_A, A_name = test_data_A[i]
        data_B, B_name = test_data_B[i]
        data_A = np.array(data_A).astype("float32")
        data_B = np.array(data_B).astype("float32")

        fake_A, fake_B, cyc_A, cyc_B = g.test([data_A, data_B])

        datas = [fake_A, fake_B, cyc_A, cyc_B, data_A, data_B]
        odatas = []
        for o in datas:
            d = np.squeeze(o[0]).transpose([1, 2, 0])
            im = ((d + 1) * 127.5).astype(np.uint8)
            odatas.append(im)
        imsave(FLAGS.output + "/fakeA_" + B_name, odatas[0])
        imsave(FLAGS.output + "/fakeB_" + A_name, odatas[1])
        imsave(FLAGS.output + "/cycA_" + A_name, odatas[2])
        imsave(FLAGS.output + "/cycB_" + B_name, odatas[3])
        imsave(FLAGS.output + "/inputA_" + A_name, odatas[4])
        imsave(FLAGS.output + "/inputB_" + B_name, odatas[5])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CycleGAN test")
    parser.add_argument(
        "-d", "--dynamic", action='store_false', help="Enable dygraph mode")
    parser.add_argument(
        "-p",
        "--device",
        type=str,
        default='gpu',
        help="device to use, gpu or cpu")
    parser.add_argument(
        "-b", "--batch_size", default=1, type=int, help="batch size")
    parser.add_argument(
        "-o",
        '--output',
        type=str,
        default='output/eval',
        help="The test result to be saved to.")
    parser.add_argument(
        "-m",
        "--init_model",
        type=str,
        default='checkpoint/199',
        help="The init model file of directory.")
    FLAGS = parser.parse_args()
    print(FLAGS)
    main()
