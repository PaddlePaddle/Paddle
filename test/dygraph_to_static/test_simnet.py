#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import random
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
)
from simnet_dygraph_model import BOW, HingeLoss

import paddle
from paddle.base.framework import unique_name

SEED = 102
random.seed(SEED)


def create_conf_dict():
    conf_dict = {}
    conf_dict["task_mode"] = "pairwise"
    conf_dict["net"] = {"emb_dim": 128, "bow_dim": 128, "hidden_dim": 128}
    conf_dict["loss"] = {"margin": 0.1}
    return conf_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Total examples' number in batch for training.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=32, help="The length of each sentence."
    )
    parser.add_argument(
        "--epoch", type=int, default=1, help="The number of training epoch."
    )
    parser.add_argument(
        "--fake_sample_size",
        type=int,
        default=128,
        help="The number of samples of fake data.",
    )
    args = parser.parse_args([])
    return args


args = parse_args()


def fake_vocabulary():
    vocab = {}
    vocab["<unk>"] = 0
    for i in range(26):
        c = chr(ord('a') + i)
        vocab[c] = i + 1
    return vocab


vocab = fake_vocabulary()


class FakeReaderProcessor(paddle.io.Dataset):
    def __init__(self, args, vocab, length):
        self.vocab = vocab
        self.seq_len = args.seq_len
        self.sample_size = args.fake_sample_size
        self.data_samples = []
        for i in range(self.sample_size):
            query = [random.randint(0, 26) for i in range(self.seq_len)]
            pos_title = query[:]
            neg_title = [26 - q for q in query]
            self.data_samples.append(
                np.array([query, pos_title, neg_title]).astype(np.int64)
            )
        self.query = []
        self.pos_title = []
        self.neg_title = []
        self._init_data(length)

    def get_reader(self, mode, epoch=0):
        def reader_with_pairwise():
            if mode == "train":
                for i in range(self.sample_size):
                    yield self.data_samples[i]

        return reader_with_pairwise

    def _init_data(self, length):
        reader = self.get_reader("train", epoch=args.epoch)()
        for i, yield_data in enumerate(reader):
            if i >= length:
                break
            self.query.append(yield_data[0])
            self.pos_title.append(yield_data[1])
            self.neg_title.append(yield_data[2])

    def __getitem__(self, idx):
        return self.query[idx], self.pos_title[idx], self.neg_title[idx]

    def __len__(self):
        return len(self.query)


simnet_process = FakeReaderProcessor(
    args, vocab, args.batch_size * (args.epoch + 1)
)


def train(conf_dict):
    """
    train process
    """
    with unique_name.guard():
        # Get device
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        else:
            place = paddle.CPUPlace()

        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        conf_dict['dict_size'] = len(vocab)
        conf_dict['seq_len'] = args.seq_len

        net = paddle.jit.to_static(BOW(conf_dict))
        loss = HingeLoss(conf_dict)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            parameters=net.parameters(),
        )

        metric = paddle.metric.Auc(name="auc")

        global_step = 0
        losses = []

        train_loader = paddle.io.DataLoader(
            simnet_process, batch_size=args.batch_size, places=[place]
        )

        for left, pos_right, neg_right in train_loader():
            left = paddle.reshape(left, shape=[-1, 1])
            pos_right = paddle.reshape(pos_right, shape=[-1, 1])
            neg_right = paddle.reshape(neg_right, shape=[-1, 1])
            net.train()
            global_step += 1
            left_feat, pos_score = net(left, pos_right)
            pred = pos_score
            _, neg_score = net(left, neg_right)
            avg_cost = loss.compute(pos_score, neg_score)
            losses.append(np.mean(avg_cost.numpy()))
            avg_cost.backward()
            optimizer.minimize(avg_cost)
            net.clear_gradients()
    return losses


class TestSimnet(Dy2StTestBase):
    def test_dygraph_static_same_loss(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        conf_dict = create_conf_dict()
        with enable_to_static_guard(False):
            dygraph_loss = train(conf_dict)

        static_loss = train(conf_dict)

        self.assertEqual(len(dygraph_loss), len(static_loss))
        for i in range(len(dygraph_loss)):
            self.assertAlmostEqual(dygraph_loss[i], static_loss[i])


if __name__ == '__main__':
    unittest.main()
