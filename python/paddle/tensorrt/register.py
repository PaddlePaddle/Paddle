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

        # Extract major, minor, and patch version numbers
        trt_major, trt_minor, trt_patch = map(int, trt_version.split('.'))

        if version_range.startswith('trt_version_ge='):
            min_version = version_range.split('=')[1]
            min_major, min_minor, min_patch = map(int, min_version.split('.'))
            # Compare major, minor, and patch versions
            if trt_major > min_major:
                return True
            elif trt_major == min_major and trt_minor > min_minor:
                return True
            elif (
                trt_major == min_major
                and trt_minor == min_minor
                and trt_patch >= min_patch
            ):
                return True
            else:
                return False

        elif version_range.startswith('trt_version_le='):
            max_version = version_range.split('=')[1]
            max_major, max_minor, max_patch = map(int, max_version.split('.'))
            # Compare major, minor, and patch versions
            if trt_major < max_major:
                return True
            elif trt_major == max_major and trt_minor < max_minor:
                return True
            elif (
                trt_major == max_major
                and trt_minor == max_minor
                and trt_patch <= max_patch
            ):
                return True
            else:
                return False

        elif 'x' in version_range:
            major_version = int(version_range.split('.')[0])
            return trt_major == major_version

        return False


converter_registry = ConverterOpRegistry()
