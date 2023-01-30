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

<<<<<<< HEAD
from .api import save
from .api import load
from .api import to_static
from .api import not_to_static
from .api import ignore_module
from .dy2static.program_translator import enable_to_static

from .dy2static.logging_utils import set_code_level, set_verbosity
from .translated_layer import TranslatedLayer

__all__ = [  # noqa
    'save',
    'load',
    'to_static',
    'ignore_module',
    'TranslatedLayer',
    'set_code_level',
    'set_verbosity',
    'not_to_static',
    'enable_to_static',
=======
from __future__ import print_function

from ..fluid.dygraph.jit import save  # noqa: F401
from ..fluid.dygraph.jit import load  # noqa: F401
from ..fluid.dygraph.jit import TracedLayer  # noqa: F401
from ..fluid.dygraph.jit import set_code_level  # noqa: F401
from ..fluid.dygraph.jit import set_verbosity  # noqa: F401
from ..fluid.dygraph.jit import declarative as to_static  # noqa: F401
from ..fluid.dygraph.jit import not_to_static  # noqa: F401
from ..fluid.dygraph import ProgramTranslator  # noqa: F401
from ..fluid.dygraph.io import TranslatedLayer  # noqa: F401

from . import dy2static  # noqa: F401

__all__ = [  # noqa
    'save', 'load', 'TracedLayer', 'to_static', 'ProgramTranslator',
    'TranslatedLayer', 'set_code_level', 'set_verbosity', 'not_to_static'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
]
