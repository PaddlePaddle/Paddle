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
"""
Convert torch parameter file to paddle model files.

Note: must have torchfile installed in order to use this tool.

Usage: python torch2paddle.py -i torchfile.t7 -l layers.txt -o path/to/paddle_model
"""

import os
import sys
import struct
import numpy as np
import torchfile
import cPickle as pickle
import argparse


# save parameters
def save_layer_parameters(outfile, feats):
    version = 0
    value_size = 4
    ret = ""
    for feat in feats:
        ret += feat.tostring()
    size = len(ret) / 4
    fo = open(outfile, 'wb')
    fo.write(struct.pack('iIQ', version, value_size, size))
    fo.write(ret)
    fo.close()


def save_net_parameters(layers, params, output_path):
    for i in range(len(layers)):
        weight = params[i * 2]
        biases = params[i * 2 + 1]
        weight_file = os.path.join(output_path, '_%s.w0' % layers[i])
        biases_file = os.path.join(output_path, '_%s.wbias' % layers[i])
        print "Saving for layer %s." % layers[i]
        save_layer_parameters(weight_file, [weight])
        save_layer_parameters(biases_file, biases)


def load_layer_parameters(filename):
    fn = open(filename, 'rb')
    version, = struct.unpack('i', fn.read(4))
    value_length, = struct.unpack("I", fn.read(4))
    dtype = 'float32' if value_length == 4 else 'float64'
    param_size, = struct.unpack("L", fn.read(8))
    value = np.fromfile(fn, dtype)
    return value


def main(argv):
    """
    main method of converting torch to paddle files.
    :param argv:
    :return:
    """
    cmdparser = argparse.ArgumentParser(
        "Convert torch parameter file to paddle model files.")
    cmdparser.add_argument(
        '-i', '--input', help='input filename of torch parameters')
    cmdparser.add_argument('-l', '--layers', help='list of layer names')
    cmdparser.add_argument(
        '-o', '--output', help='output file path of paddle model')

    args = cmdparser.parse_args(argv)
    if args.input and args.layers and args.output:
        params = torchfile.load(args.input)
        layers = [line.strip() for line in open(args.layers, 'r')]
        save_net_parameters(layers, params, args.output)
    else:
        print(
            'Usage: python torch2paddle.py -i torchfile.t7 -l layers.txt -o path/to/paddle_model'
        )


if __name__ == "__main__":
    main(sys.argv[1:])
