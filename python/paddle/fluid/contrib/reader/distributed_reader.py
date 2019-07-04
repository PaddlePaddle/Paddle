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

from __future__ import print_function
import os

__all__ = ["distributed_batch_reader"]


def distributed_batch_reader(batch_reader):
    """
    Create a reader for multi-process training. The input must be a batch reader.

    Args:
        batch_reader (callable): The input reader should be a batch reader.

    Examples:

    .. code-block:: python
           import paddle
           import paddle.fluid as fluid

           train_reader = paddle.batch(paddle.dataset.mnist.train(),
                    batch_size=32,drop_last=True)
           train_reader = fluid.contrib.reader.distributed_batch_reader(
                    train_reader)

    """
    trainers_num = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID", 0))
    assert trainer_id < trainers_num

    def decorate_for_multi_process():
        if trainers_num > 1:
            print("start data reader (trainers_num: {}, trainer_id: {})".format(
                trainers_num, trainer_id))

        train_data, idx = None, 1
        for batch_id, data in enumerate(batch_reader()):
            if trainers_num > 1:
                if idx < trainers_num:
                    if idx == trainer_id + 1:
                        train_data = data
                    idx += 1
                else:
                    if idx == trainer_id + 1:
                        train_data = data
                    assert train_data is not None, "train data should not be None."
                    yield train_data
                    train_data, idx = None, 1
            else:
                yield data

    return decorate_for_multi_process
