// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/pass/fusion_helper_base.h"
#include "paddle/cinn/utils/small_vector.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace api {

class Shape final {
 public:
  explicit Shape(const utils::ShapeType& shape)
      : shape_(shape.begin(), shape.end()) {}

  Shape(const Shape& other) = delete;
  Shape(Shape&& other) = delete;

  Shape& operator=(const Shape& other) = delete;

  bool operator==(const Shape& other) const { return shape_ == other.shape_; }

  size_t operator[](size_t index) const { return shape_[index]; }

  size_t at(size_t index) const { return shape_[index]; }

  size_t size() const { return shape_.size(); }

  // Returns the total number of elements in the shape.
  size_t numel() const {
    return std::accumulate(
        shape_.begin(), shape_.end(), 1, std::multiplies<int>());
  }

 private:
  cinn::utils::SmallVector<int64_t, 12> shape_;
};

}  // namespace api
}  // namespace cinn
