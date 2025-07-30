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

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"

namespace phi::distributed {

phi::DDim DistMetaTensor::dims() const {
  // member values in tensor_ have higher priority than those in DistMetaTensor
  if (tensor_ != nullptr) {
    PADDLE_ENFORCE_EQ(this->is_dist(),
                      true,
                      common::errors::InvalidArgument(
                          "The current MetaTensor doesn't contains "
                          "DistTensor when call `dims` method."));
    return MetaTensor::dims();
  } else {
    return dims_;
  }
}

const distributed::TensorDistAttr& DistMetaTensor::dist_attr() const {
  // member values in tensor_ have higher priority than those in DistMetaTensor
  if (tensor_ != nullptr) {
    PADDLE_ENFORCE_EQ(this->is_dist(),
                      true,
                      common::errors::InvalidArgument(
                          "The current MetaTensor doesn't contains "
                          "DistTensor when call `dist_attr` method."));
    return static_cast<phi::distributed::DistTensor*>(tensor_)->dist_attr();
  } else {
    return dist_attr_;
  }
}

bool DistMetaTensor::initialized() const {
  return tensor_ != nullptr || dist_attr_ != TensorDistAttr();
}

}  // namespace phi::distributed
