# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

u1 = paddle.uint8
i1 = paddle.int8
i2 = paddle.int16
i4 = paddle.int32
i8 = paddle.int64
f2 = paddle.float16
f4 = paddle.float32
f8 = paddle.float64
c4 = paddle.complex64
c8 = paddle.complex128
b1 = paddle.bool
bf = paddle.bfloat16


Number = {
    paddle.uint8: 0,
    paddle.int8: 1,
    paddle.int16: 2,
    paddle.int32: 3,
    paddle.int64: 4,
    paddle.float16: 5,
    paddle.float32: 6,
    paddle.float64: 7,
    paddle.complex64: 8,
    paddle.complex128: 9,
    paddle.bool: 10,
    paddle.bfloat16: 11,
}

promoteTypesLookup = [
    [u1, i2, i2, i4, i8, f2, f4, f8, c4, c8, u1, bf],
    [i2, i1, i2, i4, i8, f2, f4, f8, c4, c8, i1, bf],
    [i2, i2, i2, i4, i8, f2, f4, f8, c4, c8, i2, bf],
    [i4, i4, i4, i4, i8, f2, f4, f8, c4, c8, i4, bf],
    [i8, i8, i8, i8, i8, f2, f4, f8, c4, c8, i8, bf],
    [f2, f2, f2, f2, f2, f2, f4, f8, c4, c8, f2, f4],
    [f4, f4, f4, f4, f4, f4, f4, f8, c4, c8, f4, f4],
    [f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, f8, f8],
    [c4, c4, c4, c4, c4, c4, c4, c8, c4, c8, c4, c4],
    [c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8],
    [u1, i1, i2, i4, i8, f2, f4, f8, c4, c8, b1, bf],
    [bf, bf, bf, bf, bf, f4, f4, f8, c4, c8, bf, bf],
]


def get_result_dtype(x_dtype, y_dtype):
    if x_dtype == y_dtype:
        return x_dtype
    else:
        try:
            return promoteTypesLookup[Number[x_dtype]][Number[y_dtype]]
        except:
            print(
                "got unsupport dtype for type promotion: {} and {}.".format(
                    x_dtype, y_dtype
                )
            )
