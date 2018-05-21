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
import numpy as np

__all__ = ['create_lod_tensor', 'create_random_int_lodtensor']


def _validate_lod(lod, tensor_height=-1):
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
    if tensor_height == -1:
        return True
    else:
        return lod_sum[-1] == tensor_height


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
        return create_lod_tensor(np.array(data), lod, place)
    elif isinstance(data, np.ndarray):
        assert _validate_lod(lod,
                             data.shape[0]), "the provided lod info is invalid"
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_lod(_convert_lod(lod))
        return tensor
    else:
        raise Exception("data should be either a LoDTensor or a Numpy array, \
        	             but you pass type %s instead" % (type(data)))


def create_random_int_lodtensor(lod, shape, low, high, place):
    assert _validate_lod(lod), "the provided lod info is invalid"
    assert isinstance(shape, list), "shape should be a list"
    converted_lod = _convert_lod(lod)
    shape.insert(0, converted_lod[-1][-1])
    # The range of integer data elements is [low, high]    
    data = np.random.random_integers(low, high, shape).astype("int64")
    return create_lod_tensor(data, lod, place)
