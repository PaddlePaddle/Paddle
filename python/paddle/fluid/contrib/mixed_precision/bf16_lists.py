#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .fp16_lists import AutoMixedPrecisionLists, \
    white_list as white_list_fp16, black_list as black_list_fp16, \
    gray_list as gray_list_fp16, unsupported_fp16_list

__all__ = ["AutoMixedPrecisionListsBF16"]


class AutoMixedPrecisionListsBF16(AutoMixedPrecisionLists):
    def __init__(self,
                 custom_white_list=None,
                 custom_black_list=None,
                 custom_black_varnames=None):
        super(AutoMixedPrecisionListsBF16, self).__init__(
            white_list,
            black_list,
            gray_list,
            unsupported_list,
            custom_white_list=custom_white_list,
            custom_black_list=custom_black_list,
            custom_black_varnames=custom_black_varnames)


white_list = {'elementwise_add'}
black_list = black_list_fp16.copy().copy()
black_list.update(white_list_fp16)
black_list.update(gray_list_fp16)
gray_list = set()
unsupported_list = unsupported_fp16_list

CustomOpListsBF16 = AutoMixedPrecisionListsBF16
