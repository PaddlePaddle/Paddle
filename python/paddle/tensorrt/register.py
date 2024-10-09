# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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


class ConverterOpRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, op_name, trt_version=None):
        def decorator(func):
            if op_name not in self._registry:
                self._registry[op_name] = []
            self._registry[op_name].append((trt_version, func))
            return func

        return decorator

    def get(self, op_name, trt_version=None):
        if op_name not in self._registry:
            return None
        for version_range, func in self._registry[op_name]:
            if self._version_match(trt_version, version_range):
                return func
        return self._registry.get(op_name)

    def _version_match(self, trt_version, version_range):
        if version_range is None:
            return True

        trt_major, trt_minor = map(int, trt_version.split('.')[:2])
        if version_range.startswith('trt_version_ge='):
            min_version = float(version_range.split('=')[1])
            return float(trt_major) + trt_minor / 10 >= min_version
        elif version_range.startswith('trt_version_le='):
            max_version = float(version_range.split('=')[1])
            return float(trt_major) + trt_minor / 10 <= max_version
        elif 'x' in version_range:
            major_version = int(version_range.split('.')[0])
            return trt_major == major_version
        else:
            return False


converter_registry = ConverterOpRegistry()
