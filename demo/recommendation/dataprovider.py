# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer.PyDataProvider2 import *
import common_utils  # parse


def hook(settings, meta, **kwargs):
    """
    Init hook is invoked before process data. It will set obj.slots and store
    data meta.

    :param obj: global object. It will passed to process routine.
    :type obj: object
    :param meta: the meta file object, which passed from trainer_config. Meta
                 file record movie/user features.
    :param kwargs: unused other arguments.
    """
    del kwargs  # unused kwargs

    # Header define slots that used for paddle.
    #    first part is movie features.
    #    second part is user features.
    #    final part is rating score.
    # header is a list of [USE_SEQ_OR_NOT?, SlotType]
    headers = list(common_utils.meta_to_header(meta, 'movie'))
    headers.extend(list(common_utils.meta_to_header(meta, 'user')))
    headers.append(dense_vector(1))  # Score

    # slot types.
    settings.input_types = headers
    settings.meta = meta


@provider(init_hook=hook, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, filename):
    with open(filename, 'r') as f:
        for line in f:
            # Get a rating from file.
            user_id, movie_id, score = map(int, line.split('::')[:-1])

            # Scale score to [-5, +5]
            score = float(score) * 2 - 5.0

            # Get movie/user features by movie_id, user_id
            movie_meta = settings.meta['movie'][movie_id]
            user_meta = settings.meta['user'][user_id]

            outputs = [movie_id - 1]

            # Then add movie features
            for each_meta in movie_meta:
                outputs.append(each_meta)

            # Then add user id.
            outputs.append(user_id - 1)

            # Then add user features.
            for each_meta in user_meta:
                outputs.append(each_meta)

            # Finally, add score
            outputs.append([score])
            # Return data to paddle
            yield outputs
