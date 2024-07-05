# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ..pir import Value

_already_patch_value_in_dist = False


def monkey_patch_value_in_dist():
    def dist_attr(self):
        dist_type = self.type().as_dist_type()
        if dist_type is not None:
            return dist_type.dist_attr()
        return None

    @property
    def placements(self):
        dist_type = self.type().as_dist_type()
        if dist_type is not None:
            return dist_type.dist_attr().placements
        return None

    value_methods = [
        ('dist_attr', dist_attr),
        ('placements', placements),
    ]

    global _already_patch_value_in_dist
    if not _already_patch_value_in_dist:
        for method in value_methods:
            method_name = method[0]
            method_impl = method[1]
            setattr(Value, method_name, method_impl)

        _already_patch_value_in_dist = True
