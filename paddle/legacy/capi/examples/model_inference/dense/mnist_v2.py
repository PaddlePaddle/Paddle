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

import os
import sys
import gzip
import logging
import argparse
from PIL import Image
import numpy as np

import paddle.v2 as paddle
from paddle.utils.dump_v2_config import dump_v2_config

logger = logging.getLogger("paddle")
logger.setLevel(logging.INFO)


def multilayer_perceptron(img, layer_size, lbl_dim):
    for idx, size in enumerate(layer_size):
        hidden = paddle.layer.fc(input=(img if not idx else hidden),
                                 size=size,
                                 act=paddle.activation.Relu())
    return paddle.layer.fc(input=hidden,
                           size=lbl_dim,
                           act=paddle.activation.Softmax())


def network(input_dim=784, lbl_dim=10, is_infer=False):
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(input_dim))

    predict = multilayer_perceptron(
        images, layer_size=[128, 64], lbl_dim=lbl_dim)

    if is_infer:
        return predict
    else:
        label = paddle.layer.data(
            name='label', type=paddle.data_type.integer_value(lbl_dim))
        return paddle.layer.classification_cost(input=predict, label=label)


def main(task="train", use_gpu=False, trainer_count=1, save_dir="models"):
    if task == "train":
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        paddle.init(use_gpu=use_gpu, trainer_count=trainer_count)
        cost = network()
        parameters = paddle.parameters.create(cost)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.1 / 128.0,
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=optimizer)

        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    logger.info("Pass %d, Batch %d, Cost %f, %s" %
                                (event.pass_id, event.batch_id, event.cost,
                                 event.metrics))
            if isinstance(event, paddle.event.EndPass):
                with gzip.open(
                        os.path.join(save_dir, "params_pass_%d.tar" %
                                     event.pass_id), "w") as f:
                    trainer.save_parameter_to_tar(f)

        trainer.train(
            reader=paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.mnist.train(), buf_size=8192),
                batch_size=128),
            event_handler=event_handler,
            num_passes=5)
    elif task == "dump_config":
        predict = network(is_infer=True)
        dump_v2_config(predict, "trainer_config.bin", True)
    else:
        raise RuntimeError(("Error value for parameter task. "
                            "Available options are: train and dump_config."))


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="PaddlePaddle MNIST demo for CAPI.")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help=("A string indicating the taks type. "
              "Available options are: \"train\", \"dump_config\"."),
        default="train")
    parser.add_argument(
        "--use_gpu",
        type=bool,
        help=("A bool flag indicating whether to use GPU device or not."),
        default=False)
    parser.add_argument(
        "--trainer_count",
        type=int,
        help=("This parameter is only used in training task. It indicates "
              "how many computing threads are created in training."),
        default=1)
    parser.add_argument(
        "--save_dir",
        type=str,
        help=("This parameter is only used in training task. It indicates "
              "path of the directory to save the trained models."),
        default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmd()
    main(args.task, args.use_gpu, args.trainer_count, args.save_dir)
