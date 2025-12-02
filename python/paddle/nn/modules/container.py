# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import re
import types

from paddle.nn import LayerDict, LayerList

ModuleList = LayerList
ModuleDict = LayerDict


def replace_layer(text: str) -> str:
    text = re.sub(r'\blayer\b', 'module', text)
    text = re.sub(r'\bLayer\b', 'Module', text)
    return text


def patch_doc(cls, new_name: str):
    cls.__name__ = cls.__qualname__ = new_name

    if cls.__doc__:
        cls.__doc__ = replace_layer(cls.__doc__)

    for _, attr in cls.__dict__.items():
        if isinstance(attr, (types.FunctionType, classmethod, staticmethod)):
            func = (
                attr.__func__
                if isinstance(attr, (classmethod, staticmethod))
                else attr
            )
            if func.__doc__:
                func.__doc__ = replace_layer(func.__doc__)

    return cls


ModuleList = patch_doc(LayerList, "ModuleList")
ModuleDict = patch_doc(LayerDict, "ModuleDict")
