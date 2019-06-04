#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import sys

from paddle.fluid.incubate.atarashi.collection import add_to, get_from

KEY_SUMMARY_SCALAR = 1
KEY_SUMMARY_HISTOGRAM = 2


def scalar(name, tensor):
    tensor.persistable = True
    add_to(KEY_SUMMARY_SCALAR, (name, tensor.name))


def histogram(name, tensor):
    tensor.persistable = True
    add_to(KEY_SUMMARY_HISTOGRAM, (name, tensor.name))
