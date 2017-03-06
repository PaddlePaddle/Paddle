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

import paddle.trainer_config_helpers.poolings
import copy

__all__ = []
suffix = 'Pooling'

for name in paddle.trainer_config_helpers.poolings.__all__:
    new_name = name[:-len(suffix)]
    globals()[new_name] = copy.copy(
        getattr(paddle.trainer_config_helpers.poolings, name))
    globals()[new_name].__name__ = new_name
    __all__.append(new_name)
