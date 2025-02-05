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

import math
import random
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
)

import paddle
from paddle import base
from paddle.nn import Embedding


def fake_text():
    corpus = []
    for i in range(100):
        line = "i love paddlepaddle"
        corpus.append(line)
    return corpus


corpus = fake_text()


def data_preprocess(corpus):
    new_corpus = []
    for line in corpus:
        line = line.strip().lower()
        line = line.split(" ")
        new_corpus.append(line)

    return new_corpus


corpus = data_preprocess(corpus)


def build_dict(corpus, min_freq=3):
    word_freq_dict = {}
    for line in corpus:
        for word in line:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(
        word_freq_dict.items(), key=lambda x: x[1], reverse=True
    )

    word2id_dict = {}
    word2id_freq = {}
    id2word_dict = {}

    word2id_freq[0] = 1.0
    word2id_dict['[oov]'] = 0
    id2word_dict[0] = '[oov]'

    for word, freq in word_freq_dict:
        if freq < min_freq:
            word2id_freq[0] += freq
            continue

        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)
print(f"there are totoally {vocab_size} different words in the corpus")
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print(
        f"word {word}, its id {word_id}, its word freq {word2id_freq[word_id]}"
    )


def convert_corpus_to_id(corpus, word2id_dict):
    new_corpus = []
    for line in corpus:
        new_line = [
            (
                word2id_dict[word]
                if word in word2id_dict
                else word2id_dict['[oov]']
            )
            for word in line
        ]
        new_corpus.append(new_line)
    return new_corpus


corpus = convert_corpus_to_id(corpus, word2id_dict)


def subsampling(corpus, word2id_freq):
    def keep(word_id):
        return random.uniform(0, 1) < math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus)
        )

    new_corpus = []
    for line in corpus:
        new_line = [word for word in line if keep(word)]
        new_corpus.append(line)
    return new_corpus


corpus = subsampling(corpus, word2id_freq)


def build_data(
    corpus,
    word2id_dict,
    word2id_freq,
    max_window_size=3,
    negative_sample_num=10,
):
    dataset = []

    for line in corpus:
        for center_word_idx in range(len(line)):
            window_size = random.randint(1, max_window_size)
            center_word = line[center_word_idx]

            positive_word_range = (
                max(0, center_word_idx - window_size),
                min(len(line) - 1, center_word_idx + window_size),
            )
            positive_word_candidates = [
                line[idx]
                for idx in range(
                    positive_word_range[0], positive_word_range[1] + 1
                )
                if idx != center_word_idx and line[idx] != line[center_word_idx]
            ]

            if not positive_word_candidates:
                continue

            for positive_word in positive_word_candidates:
                dataset.append((center_word, positive_word, 1))

            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1

    return dataset


dataset = build_data(corpus, word2id_dict, word2id_freq)
for _, (center_word, target_word, label) in zip(range(50), dataset):
    print(
        f"center_word {id2word_dict[center_word]}, target {id2word_dict[target_word]}, label {label}"
    )


def build_batch(dataset, batch_size, epoch_num):
    center_word_batch = []
    target_word_batch = []
    label_batch = []
    eval_word_batch = []

    for epoch in range(epoch_num):
        for center_word, target_word, label in dataset:
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append([label])

            if len(eval_word_batch) < 5:
                eval_word_batch.append([random.randint(0, 99)])
            elif len(eval_word_batch) < 10:
                eval_word_batch.append([random.randint(0, vocab_size - 1)])

            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"), np.array(
                    target_word_batch
                ).astype("int64"), np.array(label_batch).astype(
                    "float32"
                ), np.array(
                    eval_word_batch
                ).astype(
                    "int64"
                )
                center_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"), np.array(
            target_word_batch
        ).astype("int64"), np.array(label_batch).astype("float32"), np.array(
            eval_word_batch
        ).astype(
            "int64"
        )


class SkipGram(paddle.nn.Layer):
    def __init__(self, name_scope, vocab_size, embedding_size, init_scale=0.1):
        super().__init__(name_scope)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(
            self.vocab_size,
            self.embedding_size,
            weight_attr=base.ParamAttr(
                name='embedding_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-0.5 / self.embedding_size,
                    high=0.5 / self.embedding_size,
                ),
            ),
        )

        self.embedding_out = Embedding(
            self.vocab_size,
            self.embedding_size,
            weight_attr=base.ParamAttr(
                name='embedding_out_para',
                initializer=paddle.nn.initializer.Uniform(
                    low=-0.5 / self.embedding_size,
                    high=0.5 / self.embedding_size,
                ),
            ),
        )

    def forward(self, center_words, target_words, label):
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # center_words_emb = [batch_size, embedding_size]
        # target_words_emb = [batch_size, embedding_size]
        word_sim = paddle.multiply(center_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)

        pred = paddle.nn.functional.sigmoid(word_sim)

        loss = paddle.nn.functional.binary_cross_entropy_with_logits(
            word_sim, label
        )
        loss = paddle.mean(loss)

        return pred, loss


batch_size = 512
epoch_num = 1
embedding_size = 200
learning_rate = 1e-3
total_steps = len(dataset) * epoch_num // batch_size


def train():
    random.seed(0)
    np.random.seed(0)

    place = (
        base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()
    )
    with base.dygraph.guard(place):
        paddle.seed(1000)

        skip_gram_model = paddle.jit.to_static(
            SkipGram("skip_gram_model", vocab_size, embedding_size)
        )
        adam = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            parameters=skip_gram_model.parameters(),
        )

        step = 0
        ret = []
        for center_words, target_words, label, eval_words in build_batch(
            dataset, batch_size, epoch_num
        ):
            center_words_var = paddle.to_tensor(center_words)
            target_words_var = paddle.to_tensor(target_words)
            label_var = paddle.to_tensor(label)
            pred, loss = skip_gram_model(
                center_words_var, target_words_var, label_var
            )

            loss.backward()
            adam.minimize(loss)
            skip_gram_model.clear_gradients()

            step += 1
            mean_loss = np.mean(loss.numpy())
            print(f"step {step} / {total_steps}, loss {mean_loss:f}")
            ret.append(mean_loss)
        return np.array(ret)


class TestWord2Vec(Dy2StTestBase):
    def test_dygraph_static_same_loss(self):
        with enable_to_static_guard(False):
            dygraph_loss = train()

        static_loss = train()
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
