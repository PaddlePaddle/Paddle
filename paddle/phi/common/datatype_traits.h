/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// #include "paddle/fluid/platform/float16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"

#pragma once

namespace phi {

// namespace plat = paddle::platform;
// using float16 = plat::float16;

// template <typename D>
// class PDDataTypeTraits;

// template <>
// class PDDataTypeTraits<float> {
//  public:
//   typedef float DataType;
// };

// template <>
// class PDDataTypeTraits<float16> {
//  public:
//   typedef half DataType;
// };

template <typename T>
struct PDDataTypeTraits {
  using DataType = T;
};

template <>
struct PDDataTypeTraits<phi::dtype::float16> {
  // Since LayerNormDirectCUDAFunctor register half type, we need to convert
  // phi::float16 to half.
  using DataType = half;
};

template <>
class PDDataTypeTraits<phi::dtype::bfloat16> {
 public:
  using DataType = __nv_bfloat16;
};

}  // namespace phi
