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
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle.fluid.profiler as profiler
import paddle.fluid as fluid

import data_reader

from paddle.fluid.dygraph.base import to_variable
import argparse
import functools
from utility import add_arguments, print_arguments, get_attention_feeder_data
from model import Input, set_device
from nets import OCRAttention, CrossEntropyCriterion
from eval import evaluate

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,         "Minibatch size.")
add_arg('epoch_num',         int,   30,         "Epoch number.")
add_arg('lr',                float, 0.001,         "Learning rate.")
add_arg('lr_decay_strategy', str,   "", "Learning rate decay strategy.")
add_arg('log_period',        int,   200,       "Log period.")
add_arg('save_model_period', int,   2000,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   2000,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./output", "The directory the model to be saved to.")
add_arg('train_images',      str,   None,       "The directory of images to be used for training.")
add_arg('train_list',        str,   None,       "The list file of images to be used for training.")
add_arg('test_images',       str,   None,       "The directory of images to be used for test.")
add_arg('test_list',         str,   None,       "The list file of images to be used for training.")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  True,      "Whether use GPU to train.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('profile',           bool,  False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,  False,      "Whether to skip test phase.")
# model hyper paramters
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('word_vector_dim',   int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,     "Number classes.")
add_arg('gradient_clip',     float, 5.0,     "Gradient clip value.")
add_arg('dynamic',           bool,  False,      "Whether to use dygraph.")


def train(args):
    device = set_device("gpu" if args.use_gpu else "cpu")
    fluid.enable_dygraph(device) if args.dynamic else None

    ocr_attention = OCRAttention(encoder_size=args.encoder_size, decoder_size=args.decoder_size,
                                 num_classes=args.num_classes, word_vector_dim=args.word_vector_dim)
    LR = args.lr
    if args.lr_decay_strategy == "piecewise_decay":
        learning_rate = fluid.layers.piecewise_decay([200000, 250000], [LR, LR * 0.1, LR * 0.01])
    else:
        learning_rate = LR
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate, parameter_list=ocr_attention.parameters())
    # grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(args.gradient_clip)

    inputs = [
        Input([None, 1, 48, 384], "float32", name="pixel"),
        Input([None, None], "int64", name="label_in"),
    ]
    labels = [
        Input([None, None], "int64", name="label_out"),
        Input([None, None], "float32", name="mask")]

    ocr_attention.prepare(optimizer, CrossEntropyCriterion(), inputs=inputs, labels=labels)


    train_reader = data_reader.data_reader(
        args.batch_size,
        shuffle=True,
        images_dir=args.train_images,
        list_file=args.train_list,
        data_type='train')

    # test_reader = data_reader.data_reader(
    #         args.batch_size,
    #         images_dir=args.test_images,
    #         list_file=args.test_list,
    #         data_type="test")

    # if not os.path.exists(args.save_model_dir):
    #     os.makedirs(args.save_model_dir)
    total_step = 0
    epoch_num = args.epoch_num
    for epoch in range(epoch_num):
        batch_id = 0
        total_loss = 0.0

        for data in train_reader():

            total_step += 1
            data_dict = get_attention_feeder_data(data)
            pixel = data_dict["pixel"]
            label_in = data_dict["label_in"].reshape([pixel.shape[0], -1])
            label_out = data_dict["label_out"].reshape([pixel.shape[0], -1])
            mask = data_dict["mask"].reshape(label_out.shape).astype("float32")

            avg_loss = ocr_attention.train(inputs=[pixel, label_in], labels=[label_out, mask])[0]
            total_loss += avg_loss

            if True:#batch_id > 0 and batch_id % args.log_period == 0:
                print("epoch: {}, batch_id: {}, loss {}".format(epoch, batch_id,
                                                                total_loss / args.batch_size / args.log_period))
                total_loss = 0.0

            batch_id += 1


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)