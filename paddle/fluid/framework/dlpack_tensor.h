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

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

class Tensor;

class DLPackTensor {
 public:
  using LaneType = decltype(::DLTensor::dtype.lanes);  // uint16_t
  using ShapeType =
      std::remove_reference<decltype(::DLTensor::shape[0])>::type;  // int64_t

  // lanes is only used in CPU to enable vectorization
  explicit DLPackTensor(const Tensor& tensor, LaneType lanes = 1);

  inline operator const ::DLTensor&() const { return t_; }

  inline operator ::DLTensor&() { return t_; }

  ::DLManagedTensor* ToCudfCompatibleDLManagedTensor();

 private:
  ::DLTensor t_;

  // The shape in DLTensor is defined as int64_t*
  // Add this member to make TVMTensor init without heap allocation
  ShapeType shape_[DDim::kMaxRank];
};

}  // namespace framework
}  // namespace paddle
