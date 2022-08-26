# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy
import paddle.tensor.math as math
import numpy as np
import paddle


def frexp(x, dtype=paddle.float32):
    x = paddle.to_tensor(x, dtype=dtype)
    input_x = paddle.abs(x)
    exponent = paddle.floor(paddle.log2(input_x))
    exponent = paddle.where(paddle.isinf(exponent),
                            paddle.full_like(exponent, 0), exponent)

    # 0填充
    mantissa = paddle.divide(input_x, 2**exponent)
    # 计算exponent
    exponent = paddle.where((mantissa <= -1),
                            paddle.add(exponent, paddle.ones_like(exponent)),
                            exponent)
    exponent = paddle.where((mantissa >= 1),
                            paddle.add(exponent, paddle.ones_like(exponent)),
                            exponent)
    mantissa = paddle.where((mantissa <= -1),
                            paddle.divide(mantissa,
                                          2**paddle.ones_like(exponent)),
                            mantissa)
    mantissa = paddle.where((mantissa >= -1),
                            paddle.divide(mantissa,
                                          2**paddle.ones_like(exponent)),
                            mantissa)
    mantissa = paddle.where((x < 0), mantissa * -1, mantissa)
    return mantissa, exponent


def compare(np_num, pd_num):
    for i in range(2):
        if (pd_num[i].numpy() == np_num[i]).all():
            pass
        else:
            return False
    return True


def test(num):
    assert compare(np.frexp(num), frexp(num, dtype=paddle.float64)), "error"


if __name__ == "__main__":
    # 测试整数
    # 测试数据为负正整数
    test([-1, 0, 1, 3.14])
    test([[-1, 0, 1, 3.14], [-1, 0, 1, 3.14]])
    test([[1.1111111, 1.1111111]])
    test([[1.1111111, 1.1111111], [1.1111111, 1.1111111]])
    test([-1111111111111111, 0, 111111111111111111])
    test([[1.1111111111111111, 122222222222.1111111]])
