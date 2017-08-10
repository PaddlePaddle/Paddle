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

import paddle.trainer_config_helpers.evaluators as evs
from config_base import __convert_to_v2__
import inspect

__all__ = []


def initialize():
    def convert_to_new_name(nm):
        return nm[:-len("_evaluator")]

    for __ev_name__ in filter(lambda x: x.endswith('_evaluator'), evs.__all__):
        __ev__ = getattr(evs, __ev_name__)
        __new_name__ = convert_to_new_name(__ev_name__)

        globals()[__new_name__] = __convert_to_v2__(__ev__, __new_name__,
                                                    __name__)
        globals()[__new_name__].__name__ = __new_name__
        __all__.append(__new_name__)


initialize()
