// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #The file has been adapted from pytorch project
// #Licensed under   BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#include <ATen/AccumulateType.h>

namespace at {

c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                    \
  case ScalarType::TypeNum:                                               \
    switch (device) {                                                     \
      case DeviceType::CUDA:                                              \
        return CppTypeToScalarType<                                       \
            at::acc_type_device<scalar_t, c10::DeviceType::CUDA>>::value; \
      default:                                                            \
        return CppTypeToScalarType<                                       \
            at::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;  \
    }

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(DEFINE_CASE)
#undef DEFINE_CASE

    default:
      TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda) {
  return is_cuda ? toAccumulateType(type, c10::DeviceType::CUDA)
                 : toAccumulateType(type, c10::DeviceType::CPU);
}

}  // namespace at
