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

import core
import numpy

__all__ = ['create_lod_tensor']


def _validate_lod(lod, tensor_height):
    assert isinstance(lod, list), "lod should be a list"
    # Empty lod is fine
    if len(lod) == 0:
        return True

    lod_sum = []
    for level in lod:
        assert isinstance(level, list), "each item in lod should be a list"
        # Each level of lod should have at least one length info
        if len(level) < 1:
            return False
        level_sum = 0
        for lod_len in level:
            # Each length in a level should be > 0
            if lod_len <= 0:
                return False
            level_sum += lod_len
        lod_sum.append(level_sum)

    for idx, val in enumerate(lod_sum[:-1]):
        # Each level's sum should be equal to 
        # the number of items in the next level
        if val != len(lod[idx + 1]):
            return False

    # Last level's sum should be equal to the tensor height
    if lod_sum[-1] != tensor_height:
        return False
    else:
        return True


def _convert_lod(lod):
    new_lod = []
    for level in lod:
        cur_len = 0
        new_level = [cur_len]
        for lod_len in level:
            cur_len += lod_len
            new_level.append(cur_len)
        new_lod.append(new_level)
    return new_lod


def create_lod_tensor(data, lod, place):
    if isinstance(data, core.LoDTensor):
        return create_lod_tensor(numpy.array(data), lod, place)
    elif isinstance(data, numpy.ndarray):
        assert _validate_lod(lod,
                             data.shape[0]), "the provided lod info is invalid"
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_lod(_convert_lod(lod))
        return tensor
    else:
        raise Exception("data should be either a LoDTensor or a Numpy array, \
        	             but you pass type %s instead" % (type(data)))
