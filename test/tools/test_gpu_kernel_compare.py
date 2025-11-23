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

import unittest

from gpu_kernel_compare import KernelManifestComparer


class TestKernelManifestComparer(unittest.TestCase):
    def setUp(self):
        # Baseline manifest simulating existing kernels
        self.baseline_manifest = {
            "kernel_a": [
                "{data_type[float]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
                "{data_type[double]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
            ],
            "kernel_b": [
                "{data_type[double]; data_layout[Undefined(AnyLayout)]; place[Place(cpu)]; library_type[PLAIN]}"
            ],
        }

        self.comparator = KernelManifestComparer(self.baseline_manifest)

    def test_all_gpu_kernel_detection(self):
        target_manifest = {
            "kernel_a": [  # Existing kernel with additional data type
                "{data_type[float]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
                "{data_type[double]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
                "{data_type[::phi::dtype::float16]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
            ],
            "kernel_b": [  # Existing kernel now has GPU support
                "{data_type[double]; data_layout[Undefined(AnyLayout)]; place[Place(cpu)]; library_type[PLAIN]}",
                "{data_type[double]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
            ],
            "kernel_c": [  # New kernel with GPU support
                "{data_type[float]; data_layout[Undefined(AnyLayout)]; place[Place(gpu:0)]; library_type[PLAIN]}",
            ],
        }

        summary = self.comparator.compare(target_manifest)
        self.assertIn("kernel_c", summary["new_kernels_with_gpu"])
        self.assertEqual(len(summary["new_kernels_with_gpu"]), 1)
        self.assertIn("kernel_a", summary["kernels_with_new_gpu_support"])
        self.assertIn("kernel_b", summary["kernels_with_new_gpu_support"])
        self.assertEqual(len(summary["kernels_with_new_gpu_support"]), 2)
        self.assertIn("kernel_a", summary["gpu_kernels_with_new_data_types"])
        self.assertIn(
            "::phi::dtype::float16",
            summary["gpu_kernels_with_new_data_types"]["kernel_a"],
        )
        self.assertEqual(
            len(summary["gpu_kernels_with_new_data_types"]["kernel_a"]), 1
        )
