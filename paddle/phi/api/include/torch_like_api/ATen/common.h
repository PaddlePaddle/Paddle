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
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/optional_array_ref.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include "paddle/phi/common/scalar.h"
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/PhiloxCudaState.h>
#endif

namespace at {

// TensorOptions
using c10::TensorOptions;

// DataType
using c10::BFloat16;
using c10::Half;

// ScalarType
using c10::ScalarType;

#define REDEFINE_CONSTANT_IN_AT(_1, _2, name) \
  constexpr ScalarType k##name = c10::k##name;
FOREACH_PADDLE_AND_TORCH_DTYPES(REDEFINE_CONSTANT_IN_AT)
#undef REDEFINE_CONSTANT_IN_AT

// IntArrayRef
using c10::IntArrayRef;

// OptionalIntArrayRef
using c10::OptionalIntArrayRef;

// MemoryFormat
using c10::MemoryFormat;

// Scalar
using Scalar = paddle::experimental::Scalar;

// Device
using c10::Device;
using c10::DeviceType;
constexpr c10::DeviceType kCPU = c10::kCPU;
constexpr c10::DeviceType kCUDA = c10::kCUDA;
constexpr c10::DeviceType kCUSTOM = c10::kCUSTOM;  // Paddle only

// CUDA namespace
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
namespace cuda {
using c10::cuda::CUDAGuard;
}
#endif
}  // namespace at
