#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import tarfile
import os
import shutil

proto_str = ('name: "MultiSlotDataFeed"\n'
             'batch_size: 2\n'
             'multi_slot_desc {\n'
             '   slots {\n'
             '       name: "words"\n'
             '       type: "uint64"\n'
             '       is_dense: false\n'
             '       is_used: true\n'
             '   }\n'
             '   slots {\n'
             '       name: "label"\n'
             '       type: "uint64"\n'
             '       is_dense: false\n'
             '       is_used: true\n'
             '   }\n'
             '}')

URL = 'http://paddle-unittest-data.gz.bcebos.com/python_paddle_fluid_tests_demo_async-executor/train_data.tar.gz'
MD5 = '2a405a31508969b3ab823f42c0f522ca'


def bow_net(data,
            label,
            dict_dim=89528,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    models/fluid/PaddleNLP/text_classification/nets.py
    """
    # embedding
    emb = fluid.layers.embedding(
        input=data, size=[dict_dim, emb_dim], is_sparse=True)
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bowh = fluid.layers.tanh(bow)
    # fc layer after conv
    fc_1 = fluid.layers.fc(input=bowh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    # probability of each class
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    # cross entropy loss
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    # mean loss
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction


class TestAsyncExecutor(unittest.TestCase):
    def setUp(self):
        with open('./data.prototxt', 'w+') as f:
            f.write(proto_str)
            f.close()

        with tarfile.open(paddle.dataset.common.download(URL, "imdb",
                                                         MD5)) as tarf:
            tarf.extractall(path='./')
            tarf.close()


if __name__ == '__main__':
    unittest.main()
