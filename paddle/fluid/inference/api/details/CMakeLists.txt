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

cc_library(reset_tensor_array SRCS reset_tensor_array.cc DEPS lod_tensor scope)
cc_library(tensor_handle SRCS tensor_handle.cc DEPS scope lod_tensor enforce)
cc_library(tensor_handle_dummy SRCS tensor_handle_dummy.cc)

cc_test(tensor_handle_test SRCS tensor_handle_test.cc DEPS paddle_inference_api)
