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


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor
    You can also see at https://github.com/keras-team/keras/blob/8ecef127f70db723c158dbe9ed3268b3d610ab55/keras/applications/mobilenet_v2.py#L505

    Args:
        divisor (int): The divisor for number of channels. Default: 8.
        min_value (int, optional): The minimum value of number of channels, if it is None,
                the default is divisor. Default: None.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
