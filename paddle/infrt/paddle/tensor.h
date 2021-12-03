// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "paddle/infrt/common/buffer.h"
#include "paddle/infrt/common/common.h"
#include "paddle/infrt/common/object.h"

namespace infrt {
namespace paddle {
using common::Target;

struct Shape {
  using dim_t = int;

  Shape() = default;
  explicit Shape(const std::vector<dim_t>& data) : data_(data) {}

  void SetData(const std::vector<dim_t>& data) { data_ = data; }

  const std::vector<dim_t>& data() const INFRT_RESULT_SHOULD_USE {
    return data_;
  }
  std::vector<dim_t>& data() INFRT_RESULT_SHOULD_USE { return data_; }
  size_t size() const INFRT_RESULT_SHOULD_USE { return data_.size(); }
  uint32_t numel() const INFRT_RESULT_SHOULD_USE {
    return std::accumulate(
        data_.begin(), data_.end(), 1, [](dim_t a, dim_t b) { return a * b; });
  }

 private:
  std::vector<dim_t> data_;
};

class _Tensor_ : public common::Object {
 public:
  _Tensor_() : buffer_(std::make_shared<Buffer>()) {}

  Shape& shape() { return shape_; }

  void Resize(const Shape& shape) {
    shape_ = shape;
    buffer_->data()->resize(
        reinterpret_cast<const infrt_dimension_t*>(shape.data().data()),
        shape.size());
  }

  template <typename T>
  inline T* mutable_data(const Target& target) {
    set_type(type_of<T>());
    if (target == common::DefaultHostTarget()) {
      int alignment = type_of<T>().ElementOf().bits();
      buffer_->ResizeLazy(alignment, shape_.numel() * sizeof(T), target);
    } else {
      buffer_->ResizeLazy(shape_.numel() * sizeof(T), target);
    }
    return reinterpret_cast<T*>(buffer_->data()->memory);
  }

  template <typename T>
  const T* data() const {
    return reinterpret_cast<T*>(buffer_->data()->memory);
  }

  const Type& type() { return type_; }

  void set_type(Type type) { type_ = type; }
  const Type& type() const { return type_; }

  infrt_buffer_t* buffer() { return buffer_->data(); }

  const char* type_info() const override { return __type_info__; }

 private:
  common::Type type_;
  // A shared ptr to make it easier to share buffer between tensors.
  std::shared_ptr<Buffer> buffer_;
  Shape shape_;

  static constexpr const char* __type_info__ = "_frontend_tensor_";
};

class Tensor : public Shared<_Tensor_> {
 public:
  Tensor() : Shared(new _Tensor_) {}
  explicit Tensor(_Tensor_* x) : Shared(x) {}
};

}  // namespace paddle
}  // namespace infrt
