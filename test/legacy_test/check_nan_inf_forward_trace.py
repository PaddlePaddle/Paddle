# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


def _stale_frame_marker():
    return paddle.rand([2])


def main():
    paddle.set_flags(
        {
            "FLAGS_check_nan_inf": 1,
            "FLAGS_check_nan_inf_level": 0,
            "FLAGS_call_stack_level": 1,
        }
    )
    cpu_place = paddle.CPUPlace()
    _stale_frame_marker()
    x = paddle.to_tensor([0.0, 1.0], stop_gradient=False, place=cpu_place)
    z = paddle.sqrt(x)
    paddle.autograd.backward([z])


if __name__ == "__main__":
    main()
