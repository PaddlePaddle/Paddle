// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/strings/string_view.h>

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/buffer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace framework {
using common::Target;

struct Shape {
  using dim_t = int;

  Shape() = default;
  explicit Shape(const std::vector<dim_t>& data) : data_(data) {}

  void SetData(const std::vector<dim_t>& data) { data_ = data; }

  const std::vector<dim_t>& data() const CINN_RESULT_SHOULD_USE {
    return data_;
  }
  std::vector<dim_t>& data() CINN_RESULT_SHOULD_USE { return data_; }
  size_t size() const CINN_RESULT_SHOULD_USE { return data_.size(); }
  uint32_t numel() const CINN_RESULT_SHOULD_USE {
    return std::accumulate(
        data_.begin(), data_.end(), 1, [](dim_t a, dim_t b) { return a * b; });
  }

 private:
  std::vector<dim_t> data_;
};

class _Tensor_ : public Object {
 public:
  _Tensor_() : buffer_(std::make_shared<Buffer>()) {}

  Shape& shape() { return shape_; }

  void Resize(const Shape& shape) {
    shape_ = shape;
    buffer_->data()->resize(
        reinterpret_cast<const cinn_dimension_t*>(shape.data().data()),
        shape.size());
  }

  inline void* mutable_data(const Target& target, const Type& type) {
    set_type(type);
    if (target == common::DefaultHostTarget()) {
      buffer_->ResizeLazy(1024, shape_.numel() * type.bytes(), target);
    } else {
      buffer_->ResizeLazy(shape_.numel() * type.bytes(), target);
    }
    return reinterpret_cast<void*>(buffer_->data()->memory);
  }

  template <typename T>
  inline T* mutable_data(const Target& target) {
    set_type(type_of<T>());
    if (target == common::DefaultHostTarget()) {
      buffer_->ResizeLazy(1024, shape_.numel() * sizeof(T), target);
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

  void set_type(Type type);
  const Type& type() const { return type_; }

  cinn_buffer_t* buffer() { return buffer_->data(); }
  std::shared_ptr<Buffer> get_buffer() { return buffer_; }
  void set_buffer(std::shared_ptr<Buffer> buffer) { buffer_ = buffer; }

  const char* type_info() const override { return __type_info__; }

 private:
  common::Type type_;
  // A shared ptr to make it easier to share buffer between tensors.
  std::shared_ptr<Buffer> buffer_;
  Shape shape_;

  static constexpr char* __type_info__ = "_frontend_tensor_";
};

class Tensor : public Shared<_Tensor_> {
 public:
  Tensor() : Shared(new _Tensor_) {}
  explicit Tensor(_Tensor_* x) : Shared(x) {}
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
