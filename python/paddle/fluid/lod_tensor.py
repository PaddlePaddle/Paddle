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
    """Check whether the input length-based lod info is valid.

    There are several things to check:
    1. lod should be a list of lists. Empty list is fine.
    2. The length of each sublist (a lod level) should be at least one.
    3. Each element in each lod level should be an integer greater than 0.
    4. The sum of one lod level should be equal to the length of the next lod level.
    5. The sum of the last lod level should be equal to the tensor height. 
       Bypass this check if user does not provide tensor_height as input.

    Args:
        lod: the length-based lod info, e.g., [[2, 3], [2, 1, 2, 3, 4]].
        tensor_height: the outermost dimension of the tensor with which the input 
            lod is associated with. 

    Returns:
        A boolean indicating whether the input lod is valid or not.
    """
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

    if tensor_height == -1:
        return True
    else:
        # Last level's sum should be equal to the tensor height
        return lod_sum[-1] == tensor_height


def _convert_lod(lod):
    """Convert a length-based lod to a offset-based lod.

    If the length-based lod is [[2, 3], [2, 1, 2, 3, 4]],
    then the offset-based lod is [[0, 2, 5], [0, 2, 3, 5, 8, 12]].

    Args:
        lod: a length-based lod info. 

    Returns:
        A list of lists as the offset-based lod converted to from the input lod.
    """
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
    """Create a lod tensor from a numpy array, a list, or an existing lod tensor.

    Create a lod tensor by doing the following:
    1. Check that the length-based input lod is valid.
    2. Convert the length-based lod to a offset-based LoD.
    3. Copy the data from a numpy array, a list or a existing lod tensor to 
       CPU or GPU device (based on input place).
    4. Set the level of detail (LoD) using the offset-based LoD.
    
    Use example:
    Suppose we want LoDTensor to hold data for sequences of word, where each word is
    represented by an integer. If we want to create a LoDTensor to represent two 
    sentences, one of 2 words, and one of 3 words. 

    Then 'data' can be a numpy array of integers with shape (5, 1).
    'lod' will be [[2, 3]], indicating the length(# of words) in each sentence.
    This length-based input lod [[2, 3]] will be converted to offset-based lod [[0, 2, 5]]
    inside the function call.

    Please refer to 
    github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/lod_tensor.md
    for more details regarding LoD.

    Args:
        data: a numpy array or a LoDTensor or a list holding the data to be copied.
        lod: a list of lists indicating the length-based LoD info specified by the user. 
        place: CPU or GPU place indicating where the data in the new LoDTensor will be stored.

    Returns:
        A fluid LoDTensor object with tensor data and lod info.
    """
    if isinstance(data, core.LoDTensor):
        return create_lod_tensor(np.array(data), lod, place)
    elif isinstance(data, list):
        # When input data is a list, it only deal with the case where the base element 
        # is an index of shape [1] and dtype int64 (e.g., word id). Hence, the generated 
        # LoDTensor will be of shape [n, 1] and dtype int64, where `n` is the total number 
        # of words or other indexes in the sequence. 
        new_lod = []
        for seq in data:
            new_lod.append(len(seq))
        assert [new_lod] == lod, "data and lod do not match"
        flattened_data = np.concatenate(data, axis=0).astype("int64")
        flattened_data = flattened_data.reshape([len(flattened_data), 1])
        return create_lod_tensor(flattened_data, lod, place)
    elif isinstance(data, np.ndarray):
        assert _validate_lod(lod,
                             data.shape[0]), "the provided lod info is invalid"
        tensor = core.LoDTensor()
        tensor.set(data, place)
        tensor.set_lod(_convert_lod(lod))
        return tensor
    else:
        raise TypeError(
            "data should be either a LoDTensor, a Numpy array or a list")


def create_random_int_lodtensor(lod, base_shape, place, low, high):
    """Create a LoDTensor containing random integers.

    This function is frequently used in the book examples. So we revised it based on 
    the new create_lod_tensor API and put it here in the lod_tensor module to simplify 
    the code. 

    The function does the following:
    1. Calculate the overall shape of the LoDTensor based on the length-based 'lod' input 
    and the shape of the basic element in 'base_shape'.
    2. Create a numpy array of this shape.
    3. Create the LoDTensor using create_lod_tensor API.

    Suppose we want LoDTensor to hold data for sequences of word, where each word is
    represented by an integer. If we want to create a LoDTensor to represent two 
    sentences, one of 2 words, and one of 3 words. Then 'base_shape' is [1], input 
    length-based 'lod' is [[2, 3]]. Then the overall shape of the LoDTensor would be 
    [5, 1], holding 5 words for two sentences. 

    Args:
        data: a numpy array or a LoDTensor holding the data to be copied.
        lod: a list of lists indicating the length-based LoD info specified by the user.
        base_shape: the shape of the basic element to be held by the LoDTensor. 
        place: CPU or GPU place indicating where the data in the new LoDTensor will be stored.
        low: the lower bound of the random integers.
        high: the upper bound of the random integers.

    Returns:
        A fluid LoDTensor object with tensor data and lod info. 
    """
    assert isinstance(base_shape, list), "base_shape should be a list"
    converted_lod = _convert_lod(lod)
    # append the total number of basic elements to the front of its shape
    overall_shape = [converted_lod[-1][-1]] + base_shape
    # the range of integer data elements is [low, high]    
    data = np.random.random_integers(low, high, overall_shape).astype("int64")
    return create_lod_tensor(data, lod, place)
