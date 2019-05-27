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

__all__ = ["distributed_sampler"]


def distributed_sampler(reader, batch_size):
    """
    Create a distributed reader.

    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: the size of the batch
    :type batch_size: int
    """

    def batch_reader():
        if not os.getenv('PADDLE_TRAINER_ID'):
            raise RuntimeError(
                "The current program is not in distributed mode.")

        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))

        def _slice_data(size):
            per_node_lines = size // trainer_count
            return [
                trainer_id * per_node_lines, (trainer_id + 1) * per_node_lines
            ]

        r = reader()
        b = []

        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                if len(b) >= trainer_count:
                    begin, end = _slice_data(len(b))
                    yield b[begin:end]
                b = []

        if len(b) >= trainer_count:
            begin, end = _slice_data(len(b))
            yield b[begin:end]

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))

    return batch_reader
