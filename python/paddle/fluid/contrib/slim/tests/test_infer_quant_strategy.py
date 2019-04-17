# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle
import os
import sys
import argparse
from collections import OrderedDict
from paddle.fluid.contrib.slim.core import Compressor

import reader


def parse_args():
    parser = argparse.ArgumentParser("Inference for stacked LSTMP model.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='The sequence number of a batch data. (default: %(default)d)')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--infer_model_path',
        type=str,
        default='./infer_models/deep_asr.pass_0.infer.model/',
        help='The directory for loading inference model. '
        '(default: %(default)s)')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


class TestInferQuantizeStrategy(object):
    def __init__(self):
        pass

    def convert(self, args):
        if not os.path.exists(args.infer_model_path):
            raise IOError("Invalid inference model path!")

        place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
        exe = fluid.Executor(place)
        [infer_program, feed_dict, fetch_targets
         ] = fluid.io.load_inference_model(args.infer_model_path, exe)
        with fluid.program_guard(infer_program, fluid.Program()):
            test_reader = paddle.batch(
                reader.train(
                    file_list='/aipg/dataset/ILSVRC2012/val_list.txt',
                    data_dir='/aipg/dataset/ILSVRC2012/val/',
                    cycle=True),
                batch_size=args.batch_size)

            image = fluid.layers.data(
                name='data', shape=[3, 224, 224], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

            eval_feed_list = OrderedDict([('x', image.name), ('y', label.name)])
            eval_fetch_list = OrderedDict([('cost', fetch_targets[0])])
            comp = Compressor(
                place,
                fluid.global_scope(),
                fluid.default_main_program(),
                train_reader=None,
                train_feed_list=None,
                train_fetch_list=None,
                eval_program=infer_program,
                eval_reader=test_reader,
                eval_feed_list=eval_feed_list,
                eval_fetch_list=eval_fetch_list,
                teacher_programs=[],
                checkpoint_path='./checkpoints',
                train_optimizer=None,
                load_model_dir=args.infer_model_path,
                distiller_optimizer=None)
            comp.config('./infer_quant_int8/config_int8.yaml')
            comp.run()


if __name__ == "__main__":
    args = parse_args()
    model = TestInferQuantizeStrategy()
    model.convert(args)
