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
            else:
                raise ValueError(
                    f"Requested TensorRT version : {trt_version} does not match the range of pip installed tensorrt versions : {version_range}"
                )
        return self._registry.get(op_name)

    def _version_match(self, trt_version, version_range):
        """
        Check if a given TensorRT version matches the specified version range.

        Args:
            trt_version (str): The TensorRT version, e.g., "8.4.1".
            version_range (str): The version range to check against, e.g.,
                                "trt_version_ge=8.2", "trt_version_le=7.1", or "8.x".

        Returns:
            bool: True if the version matches the range, False otherwise.
        """

        def _normalize_version(version):
            """
            Normalize the version string into a 3-tuple for easy comparison.
            If the version has fewer than 3 parts, it pads with zeros.

            Args:
                version (str): The version string, e.g., "8.4.1", "8.2", or "9".

            Returns:
                tuple: A tuple representing the version, e.g., (8, 4, 1).
            """
            return tuple(map(int, [*version.split('.'), '0', '0'][:3]))

        # Convert the given TensorRT version to a normalized tuple
        trt_version_tuple = _normalize_version(trt_version)
        # Split the version range into comparator and reference version
        if '=' in version_range:
            comparator, ref_version = version_range.split('=')
            # Normalize the reference version into a tuple
            ref_version_tuple = _normalize_version(ref_version)
            # Check the comparator and compare the versions
            return (
                comparator == 'trt_version_ge'
                and trt_version_tuple >= ref_version_tuple
            ) or (
                comparator == 'trt_version_le'
                and trt_version_tuple <= ref_version_tuple
            )
        # Check if the version range includes 'x' (e.g., "8.x")
        if 'x' in version_range:
            # Match only the major version (first part)
            return trt_version_tuple[0] == int(version_range.split('.')[0])

        return False


converter_registry = ConverterOpRegistry()
