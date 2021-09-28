# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def _check_tensors(in_out_list, name):
    assert in_out_list is not None, "{} should not be None".format(name)

    if isinstance(in_out_list, (list, tuple)):
        assert len(in_out_list) > 0, "{} connot be empyt".format(name)
        for each_var in in_out_list:
            assert isinstance(
                each_var,
                paddle.Tensor), "Elements of {} must be paddle.Tensor".format(
                    name)
        return list(in_out_list)
    else:
        assert isinstance(
            in_out_list,
            paddle.Tensor), "{} must be Tensor or list of Tensor".format(name)
        return [in_out_list]


def _stack_tensor_or_return_none(origin_list):
    assert len(origin_list) > 0, "Can't not stack an empty list"
    return paddle.stack(
        origin_list, axis=0) if isinstance(origin_list[0],
                                           paddle.Tensor) else None


def _replace_none_with_zero_tensor(t, spec_t):
    if t is None:
        zero_t = paddle.zeros(shape=spec_t.shape, dtype=spec_t.dtype)
        zero_t.stop_gradient = spec_t.stop_gradient
        return zero_t
    else:
        return t
