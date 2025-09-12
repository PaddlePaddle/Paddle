// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <dlpack/dlpack.h>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {

/*
dlpack related code ref:
https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/DLConvertor.cpp
and paddle/phi/api/lib/tensor_utils.cc
*/
using Deleter = std::function<void(void*)>;

phi::Place DLDeviceToPlace(const DLDevice& device);

TEST_API DLManagedTensor* toDLPack(const phi::DenseTensor& src);
DLManagedTensorVersioned* toDLPackVersioned(const phi::DenseTensor& src);
phi::DenseTensor fromDLPack(DLManagedTensor* src, Deleter deleter);
phi::DenseTensor fromDLPackVersioned(DLManagedTensorVersioned* src,
                                     Deleter deleter);

// A traits to support both DLManagedTensor and DLManagedTensorVersioned
template <typename T>
struct DLPackTraits {};

template <>
struct DLPackTraits<DLManagedTensor> {
  inline static const char* capsule = "dltensor";
  inline static const char* used = "used_dltensor";
  inline static auto toDLPack = framework::toDLPack;
};

template <>
struct DLPackTraits<DLManagedTensorVersioned> {
  inline static const char* capsule = "dltensor_versioned";
  inline static const char* used = "used_dltensor_versioned";
  inline static auto toDLPack = framework::toDLPackVersioned;
};

}  // namespace framework
}  // namespace paddle
