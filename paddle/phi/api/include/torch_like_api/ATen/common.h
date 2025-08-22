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

#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/OptionalArrayRef.h>

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/PhiloxCudaState.h>
#endif

namespace at {

#define REDEFINE_CONSTANT_IN_AT(_1, _2, name) \
  constexpr ScalarType k##name = c10::k##name;
FOREACH_PADDLE_AND_TORCH_DTYPES(REDEFINE_CONSTANT_IN_AT)
#undef REDEFINE_CONSTANT_IN_AT

}  // namespace at
