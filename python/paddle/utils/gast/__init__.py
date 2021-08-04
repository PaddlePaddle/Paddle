# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# NOTE(paddle-dev): We introduce third-party library Gast as unified AST
# representation. See https://github.com/serge-sans-paille/gast for details.

# Copyright (c) 2016, Serge Guelton
# All rights reserved.

from .gast import *
from ast import NodeVisitor, NodeTransformer, iter_fields, dump
