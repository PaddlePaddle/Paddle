// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// A naive implementation of MetaTensor
#pragma once
#include "paddle/infrt/common/common.h"

namespace infrt {
namespace tensor {
struct DenseHostTensor;
struct TensorShape;
}  // namespace tensor

namespace naive {

class MetaTensor {
 public:
  MetaTensor() = default;
  explicit MetaTensor(tensor::DenseHostTensor* tensor)
      : mutable_tensor_(tensor) {}
  explicit MetaTensor(const tensor::DenseHostTensor* tensor)
      : mutable_tensor_(&Reference(tensor)) {}
  explicit MetaTensor(MetaTensor&& other)
      : mutable_tensor_(other.mutable_tensor_) {}
  explicit MetaTensor(const MetaTensor& other)
      : mutable_tensor_(other.mutable_tensor_) {}

  const tensor::TensorShape& shape() const;
  tensor::TensorShape* mutable_shape();

 private:
  tensor::DenseHostTensor* mutable_tensor_{};
};

}  // namespace naive
}  // namespace infrt
