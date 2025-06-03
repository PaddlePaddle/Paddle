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

import __builtin__

DataType = __builtin__.DataType
DataValue = __builtin__.DataValue
PointerType = __builtin__.PointerType
PointerValue = __builtin__.PointerValue
MutableList = __builtin__.MutableList
OrderedDict = __builtin__.OrderedDict
MutableOrderedDict = __builtin__.MutableOrderedDict
AttrMap = __builtin__.AttrMap
SerializableAttrMap = __builtin__.SerializableAttrMap

_raise = __builtin__._raise

range = __builtin__.range
map = __builtin__.map
reduce = __builtin__.reduce
filter = __builtin__.filter
zip = __builtin__.zip
flat_map = __builtin__.flat_map
apply = __builtin__.apply
to_pure_function = __builtin__.to_pure_function
replace_or_trim_left_comma = __builtin__.replace_or_trim_left_comma
quoted = __builtin__.quoted

registry = __builtin__registry  # noqa: F821

sorted = __builtin__.sorted

dirname = __builtin__.dirname
basename = __builtin__.basename

auto_immutable_value_registry_key = (
    __builtin__.auto_immutable_value_registry_key
)
is_immutable_value_registered = __builtin__.is_immutable_value_registered
get_registered_immutable_value = __builtin__.get_registered_immutable_value
register_immutable_value = __builtin__.register_immutable_value


def do_nothing():
    pass


def import_by_file_path(file_path):
    return __builtin__import(None, file_path)  # noqa: F821


def foreach(lst):
    return lambda f: __builtin__.foreach(f, lst)
