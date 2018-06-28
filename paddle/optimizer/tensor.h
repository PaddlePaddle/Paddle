//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
/**
 * @brief tensor used by optimizer
 */

#include <string.h>
#include <memory>
#include "paddle/utils/Common.h"
#include "paddle/utils/Logging.h"

namespace paddle {
namespace optimizer {

template <class T>
class TensorT {
 public:
  TensorT(size_t size) : height_(1), width_(size) {
    // new T[size]() initializes all element to zero value.
    data_ptr_ = std::shared_ptr<T>(new T[size](), std::default_delete<T[]>());
    data_ = data_ptr_.get();
  }

  TensorT(T* data, size_t size)
      : height_(1), width_(size), data_ptr_(nullptr), data_(data) {}

  TensorT(T* data, size_t h, size_t w)
      : height_(h), width_(w), data_ptr_(nullptr), data_(data) {}

  virtual ~TensorT() {}

  T* get_buffer() { return this->data_; }

  T& operator[](const size_t idx) {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  T& operator[](const size_t idx) const {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  // TODO: replace with tensorshape
  size_t size() const { return this->width_ * this->height_; }

 protected:
  size_t height_;
  size_t width_;
  std::shared_ptr<T> data_ptr_;
  T* data_;
};

// TODO(zhihong): design problem of dynamic datatype, need to fix it
typedef TensorT<float> Tensor;

}  // namespace optimizer
}  // namespace paddle
