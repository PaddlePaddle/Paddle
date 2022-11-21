# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

from .api import save  # noqa: F401
from .api import load  # noqa: F401
from .api import TracedLayer  # noqa: F401
from .api import set_code_level  # noqa: F401
from .api import set_verbosity  # noqa: F401
from .api import declarative as to_static  # noqa: F401
from .api import not_to_static  # noqa: F401
from .dy2static import ProgramTranslator  # noqa: F401
from ..fluid.dygraph.io import TranslatedLayer  # noqa: F401

from . import dy2static  # noqa: F401

__all__ = [  # noqa
    'save',
    'load',
    'TracedLayer',
    'to_static',
    'ProgramTranslator',
    'TranslatedLayer',
    'set_code_level',
    'set_verbosity',
    'not_to_static',
]
