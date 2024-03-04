// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "paddle/common/exception.h"
#include "paddle/phi/common/data_type.h"

namespace phi {

class DenseTensor;

// In static model pre analysis, we can't get the data from tensor
class TensorRef {
 public:
  // Constructor support implicit
  TensorRef() = default;
  explicit TensorRef(const DenseTensor* base) : tensor_base_(base) {}

  const DenseTensor* Get() const {
    assert(tensor_base_ != nullptr);
    return tensor_base_;
  }

 private:
  const DenseTensor* tensor_base_{nullptr};
};

}  // namespace phi
