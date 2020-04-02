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
import glob
import numpy as np
import argparse

from PIL import Image
from scipy.misc import imsave

import paddle.fluid as fluid
from check import check_gpu, check_version

from model import Model, Input, set_device
from cyclegan import Generator, GeneratorCombine


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

    out_path = FLAGS.output + "/single"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for f in glob.glob(FLAGS.input):
        image_name = os.path.basename(f)
        image = Image.open(f).convert('RGB')
        image = image.resize((256, 256), Image.BICUBIC)
        image = np.array(image) / 127.5 - 1

        image = image[:, :, 0:3].astype("float32")
        data = image.transpose([2, 0, 1])[np.newaxis, :]

        if FLAGS.input_style == "A":
            _, fake, _, _ = g.test([data, data])

        if FLAGS.input_style == "B":
            fake, _, _, _ = g.test([data, data])

        fake = np.squeeze(fake[0]).transpose([1, 2, 0])

        opath = "{}/fake{}{}".format(out_path, FLAGS.input_style, image_name)
        imsave(opath, ((fake + 1) * 127.5).astype(np.uint8))
        print("transfer {} to {}".format(f, opath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CycleGAN inference")
    parser.add_argument(
        "-d", "--dynamic", action='store_false', help="Enable dygraph mode")
    parser.add_argument(
        "-p",
        "--device",
        type=str,
        default='gpu',
        help="device to use, gpu or cpu")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default='./image/testA/123_A.jpg',
        help="input image")
    parser.add_argument(
        "-o",
        '--output',
        type=str,
        default='output',
        help="The test result to be saved to.")
    parser.add_argument(
        "-m",
        "--init_model",
        type=str,
        default='checkpoint/199',
        help="The init model file of directory.")
    parser.add_argument(
        "-s", "--input_style", type=str, default='A', help="A or B")
    FLAGS = parser.parse_args()
    check_gpu(str.lower(FLAGS.device) == 'gpu')
    check_version()
    main()
