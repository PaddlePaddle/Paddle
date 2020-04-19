#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import paddle
import logging
import paddle.fluid as fluid

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


if __name__ == "__main__":
    vocab = load_vocab('imdb.vocab')
    dict_dim = len(vocab)
    model_name = sys.argv[1]
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_batch_size(128)
    dataset.set_pipe_command("python imdb_reader.py")

    dataset.set_use_var([data, label])
    desc = dataset.proto_desc

    with open("data.proto", "w") as f:
        f.write(dataset.desc())

    from nets import *
    if model_name == 'cnn':
        logger.info("Generate program description of CNN net")
        avg_cost, acc, prediction = cnn_net(data, label, dict_dim)
    elif model_name == 'bow':
        logger.info("Generate program description of BOW net")
        avg_cost, acc, prediction = bow_net(data, label, dict_dim)
    else:
        logger.error("no such model: " + model_name)
        exit(0)
    # optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimizer = fluid.optimizer.Adagrad(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    with open(model_name + "_main_program", "wb") as f:
        f.write(fluid.default_main_program().desc.serialize_to_string())

    with open(model_name + "_startup_program", "wb") as f:
        f.write(fluid.default_startup_program().desc.serialize_to_string())
