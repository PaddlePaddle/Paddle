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
from __future__ import print_function

import os
import sys
import random
import numpy as np

import argparse
import functools

import paddle.fluid.profiler as profiler
import paddle.fluid as fluid

from hapi.model import Input, set_device
from hapi.vision.transforms import BatchCompose

from utility import add_arguments, print_arguments
from utility import SeqAccuracy, LoggerCallBack
from seq2seq_attn import Seq2SeqAttModel, WeightCrossEntropy
import data

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   32,           "Minibatch size.")
add_arg('epoch',             int,   30,           "Epoch number.")
add_arg('num_workers',       int,   0,            "workers number.")
add_arg('lr',                float, 0.001,        "Learning rate.")
add_arg('lr_decay_strategy', str,   "",           "Learning rate decay strategy.")
add_arg('checkpoint_path',   str,   "checkpoint", "The directory the model to be saved to.")
add_arg('train_images',      str,   None,         "The directory of images to be used for training.")
add_arg('train_list',        str,   None,         "The list file of images to be used for training.")
add_arg('test_images',       str,   None,         "The directory of images to be used for test.")
add_arg('test_list',         str,   None,         "The list file of images to be used for training.")
add_arg('resume_path',       str,   None,         "The init model file of directory.")
add_arg('use_gpu',           bool,  True,         "Whether use GPU to train.")
# model hyper paramters
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('embedding_dim',     int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,     "Number classes.")
add_arg('gradient_clip',     float, 5.0,     "Gradient clip value.")
add_arg('dynamic',           bool,  False,      "Whether to use dygraph.")
# yapf: enable


def main(FLAGS):
    device = set_device("gpu" if FLAGS.use_gpu else "cpu")
    fluid.enable_dygraph(device) if FLAGS.dynamic else None

    model = Seq2SeqAttModel(
        encoder_size=FLAGS.encoder_size,
        decoder_size=FLAGS.decoder_size,
        emb_dim=FLAGS.embedding_dim,
        num_classes=FLAGS.num_classes)

    lr = FLAGS.lr
    if FLAGS.lr_decay_strategy == "piecewise_decay":
        learning_rate = fluid.layers.piecewise_decay(
            [200000, 250000], [lr, lr * 0.1, lr * 0.01])
    else:
        learning_rate = lr
    grad_clip = fluid.clip.GradientClipByGlobalNorm(FLAGS.gradient_clip)
    optimizer = fluid.optimizer.Adam(
        learning_rate=learning_rate,
        parameter_list=model.parameters(),
        grad_clip=grad_clip)

    # yapf: disable
    inputs = [
        Input([None,1,48,384], "float32", name="pixel"),
        Input([None, None], "int64", name="label_in"),
    ]
    labels = [
        Input([None, None], "int64", name="label_out"),
        Input([None, None], "float32", name="mask"),
    ]
    # yapf: enable

    model.prepare(
        optimizer,
        WeightCrossEntropy(),
        SeqAccuracy(),
        inputs=inputs,
        labels=labels)

    train_dataset = data.train()
    train_collate_fn = BatchCompose(
        [data.Resize(), data.Normalize(), data.PadTarget()])
    train_sampler = data.BatchSampler(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    train_loader = fluid.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        places=device,
        num_workers=FLAGS.num_workers,
        return_list=True,
        collate_fn=train_collate_fn)
    test_dataset = data.test()
    test_collate_fn = BatchCompose(
        [data.Resize(), data.Normalize(), data.PadTarget()])
    test_sampler = data.BatchSampler(
        test_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=False,
        shuffle=False)
    test_loader = fluid.io.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        places=device,
        num_workers=0,
        return_list=True,
        collate_fn=test_collate_fn)

    model.fit(train_data=train_loader,
              eval_data=test_loader,
              epochs=FLAGS.epoch,
              save_dir=FLAGS.checkpoint_path,
              callbacks=[LoggerCallBack(10, 2, FLAGS.batch_size)])


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    main(FLAGS)
