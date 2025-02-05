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
#include <glog/logging.h>

#include <string>
#include "paddle/common/enforce.h"
/**
 * runtime::Buffer is an encapsulation of memory operations.
 */
namespace cinn {
namespace runtime {

/**
 * Shape of the buffers.
 */
struct Shape {
  using value_type = int32_t;

  Shape() = default;

  Shape(const Shape& other);

  bool defined() const { return data_; }

  //! Get the number of dimensions.
  uint32_t ndims() const { return ndims_; }

  //! Get the mutable data.
  value_type* data() { return data_; }
  //! Get the immutable data.
  const value_type* data() const { return data_; }

  //! Resize the number of dimensions.
  void Resize(int ndim);

  //! Get the number of elements the shape defines.
  uint32_t num_elements() const;

  //! Get i-th element.
  value_type& operator[](int i);
  //! Get i-th element.
  value_type operator[](int i) const;

 private:
  uint32_t ndims_{0};
  int32_t* data_{};
};

/**
 * A C++ wrapper for buffer.
 */
template <typename T>
class Buffer {
 public:
  explicit Buffer(const Shape& shape) : shape_(shape) {}

  //! Allocate the memory in host device.
  void AllocHost() {
    PADDLE_ENFORCE_EQ(
        shape_.defined(),
        true,
        ::common::errors::InvalidArgument("shape haven't been defined."));
    data_ = new T[shape_.num_elements()];
    PADDLE_ENFORCE_NOT_NULL(data_,
                            ::common::errors::NotFound("alloc buffer failed."));
  }
  //! Deallocate the memory in host device.
  void DeallocHost() {
    if (data_) delete data_;
    data_ = nullptr;
  }

  T& operator()(int i0) {
    PADDLE_ENFORCE_EQ(shape_.ndims(),
                      1,
                      ::common::errors::InvalidArgument(
                          "Expected shape has 1 dimension, but received %d.",
                          shape_.ndims()));
    return static_cast<T*>(data_)[i0];
  }
  T& operator()(int i0, int i1) {
    PADDLE_ENFORCE_EQ(shape_.ndims(),
                      2,
                      ::common::errors::InvalidArgument(
                          "Expected shape has 2 dimensions, but received %d.",
                          shape_.ndims()));
    return static_cast<T*>(data_)[i0 * shape_[0] + i1];
  }
  T& operator()(int i0, int i1, int i2) {
    PADDLE_ENFORCE_EQ(shape_.ndims(),
                      3,
                      ::common::errors::InvalidArgument(
                          "Expected shape has 3 dimensions, but received %d.",
                          shape_.ndims()));
    return static_cast<T*>(
        data_)[i0 * shape_[1] * shape_[2] + i1 * shape_[2] + i2];
  }

 private:
  Shape shape_;
  void* data_{};
};

}  // namespace runtime
}  // namespace cinn
