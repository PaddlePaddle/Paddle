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
import random
import numpy as np

import argparse
import functools
from PIL import Image

import paddle.fluid.profiler as profiler
import paddle.fluid as fluid

from hapi.model import Input, set_device
from hapi.datasets.folder import ImageFolder
from hapi.vision.transforms import BatchCompose

from utility import add_arguments, print_arguments
from utility import postprocess, index2word
from seq2seq_attn import Seq2SeqAttInferModel, WeightCrossEntropy
import data

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,       "Minibatch size.")
add_arg('image_path',        str,   None,    "The directory of images to be used for test.")
add_arg('init_model',        str,   None,    "The init model file of directory.")
add_arg('use_gpu',           bool,  True,    "Whether use GPU to train.")
# model hyper paramters
add_arg('encoder_size',      int,   200,     "Encoder size.")
add_arg('decoder_size',      int,   128,     "Decoder size.")
add_arg('embedding_dim',     int,   128,     "Word vector dim.")
add_arg('num_classes',       int,   95,      "Number classes.")
add_arg('beam_size',         int,   3,       "Beam size for beam search.")
add_arg('dynamic',           bool,  False,   "Whether to use dygraph.")
# yapf: enable


def main(FLAGS):
    device = set_device("gpu" if FLAGS.use_gpu else "cpu")
    fluid.enable_dygraph(device) if FLAGS.dynamic else None
    model = Seq2SeqAttInferModel(
        encoder_size=FLAGS.encoder_size,
        decoder_size=FLAGS.decoder_size,
        emb_dim=FLAGS.embedding_dim,
        num_classes=FLAGS.num_classes,
        beam_size=FLAGS.beam_size)

    inputs = [Input([None, 1, 48, 384], "float32", name="pixel"), ]

    model.prepare(inputs=inputs, device=device)
    model.load(FLAGS.init_model)

    fn = lambda p: Image.open(p).convert('L')
    test_dataset = ImageFolder(FLAGS.image_path, loader=fn)
    test_collate_fn = BatchCompose([data.Resize(), data.Normalize()])
    test_loader = fluid.io.DataLoader(
        test_dataset,
        places=device,
        num_workers=0,
        return_list=True,
        collate_fn=test_collate_fn)

    samples = test_dataset.samples
    #outputs = model.predict(test_loader)
    ins_id = 0
    for image, in test_loader:
        image = image if FLAGS.dynamic else image[0]
        pred = model.test_batch([image])[0]
        pred = pred[:, :, np.newaxis] if len(pred.shape) == 2 else pred
        pred = np.transpose(pred, [0, 2, 1])
        for ins in pred:
            impath = samples[ins_id]
            ins_id += 1
            print('Image {}: {}'.format(ins_id, impath))
            for beam_idx, beam in enumerate(ins):
                id_list = postprocess(beam)
                word_list = index2word(id_list)
                sequence = "".join(word_list)
                print('{}: {}'.format(beam_idx, sequence))


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    main(FLAGS)
