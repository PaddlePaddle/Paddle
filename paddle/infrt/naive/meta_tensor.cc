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

#include "paddle/infrt/naive/meta_tensor.h"

#include "paddle/infrt/tensor/dense_host_tensor.h"
#include "paddle/infrt/tensor/tensor_shape.h"

namespace infrt {
namespace naive {

const tensor::TensorShape& MetaTensor::shape() const {
  return mutable_tensor_->shape();
}
tensor::TensorShape* MetaTensor::mutable_shape() {
  return mutable_tensor_->mutable_shape();
}

}  // namespace naive
}  // namespace infrt
