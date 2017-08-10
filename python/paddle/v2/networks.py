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

import paddle.trainer_config_helpers.networks as conf_nw
import inspect
from config_base import __convert_to_v2__

__all__ = []


def __initialize__():
    for each_subnetwork in conf_nw.__all__:
        if each_subnetwork in ['inputs', 'outputs']:
            continue
        func = getattr(conf_nw, each_subnetwork)
        globals()[each_subnetwork] = func
        globals()[each_subnetwork].__name__ = each_subnetwork
        global __all__
        __all__.append(each_subnetwork)


__initialize__()
