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

_global_collection = None


class Collections(object):
    """global collections to record everything"""

    def __init__(self):
        self.col = {}

    def __enter__(self):
        global _global_collection
        _global_collection = self
        return self

    def __exit__(self, err_type, err_value, trace):
        global _global_collection
        _global_collection = None

    def add_to(self, key, val):
        self.col.setdefault(key, []).append(val)

    def get_from(self, key):
        return self.col.get(key, None)


def add_to(key, val):
    if _global_collection is not None:
        _global_collection.add_to(key, val)


def get_from(key):
    if _global_collection is not None:
        return _global_collection.get_collection(key)
    return None
