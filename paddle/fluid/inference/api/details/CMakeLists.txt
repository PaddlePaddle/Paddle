# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cc_library(
  reset_tensor_array
  SRCS reset_tensor_array.cc
  DEPS lod_tensor scope)
if(WITH_ONNXRUNTIME)
  cc_library(
    zero_copy_tensor
    SRCS zero_copy_tensor.cc
    DEPS scope lod_tensor phi onnxruntime common)
  cc_library(
    zero_copy_tensor_dummy
    SRCS zero_copy_tensor_dummy.cc
    DEPS onnxruntime phi common)
else()
  cc_library(
    zero_copy_tensor
    SRCS zero_copy_tensor.cc
    DEPS scope lod_tensor phi common)
  cc_library(
    zero_copy_tensor_dummy
    SRCS zero_copy_tensor_dummy.cc
    DEPS phi common)
endif()

cc_test(
  zero_copy_tensor_test
  SRCS zero_copy_tensor_test.cc
  DEPS paddle_inference_api)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(zero_copy_tensor_test)
endif()
