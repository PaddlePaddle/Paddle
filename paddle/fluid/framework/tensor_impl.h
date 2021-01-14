/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace framework {
template <typename T>
inline const T* Tensor::data() const {
  check_memory_size();
  bool valid =
      std::is_same<T, void>::value || type_ == DataTypeTrait<T>::DataType();
  PADDLE_ENFORCE_EQ(
      valid, true,
      platform::errors::InvalidArgument(
          "Tensor holds the wrong type, it holds %s, but desires to be %s.",
          DataTypeToString(type_),
          DataTypeToString(DataTypeTrait<T>::DataType())));

  return reinterpret_cast<const T*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
}

inline bool Tensor::IsInitialized() const { return holder_ != nullptr; }

template <typename T>
inline T* Tensor::data() {
  check_memory_size();
  bool valid =
      std::is_same<T, void>::value || type_ == DataTypeTrait<T>::DataType();
  PADDLE_ENFORCE_EQ(
      valid, true,
      platform::errors::InvalidArgument(
          "Tensor holds the wrong type, it holds %s, but desires to be %s",
          DataTypeToString(type_),
          DataTypeToString(DataTypeTrait<T>::DataType())));

  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                              offset_);
}

template <typename T>
inline T* Tensor::mutable_data(const DDim& dims, const platform::Place& place,
                               size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  Resize(dims);
  return mutable_data<T>(place, requested_size);
}

template <typename T>
inline T* Tensor::mutable_data(const platform::Place& place,
                               size_t requested_size) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  return reinterpret_cast<T*>(
      mutable_data(place, DataTypeTrait<T>::DataType(), requested_size));
}

inline Tensor ReshapeToMatrix(const Tensor& src, int num_col_dims) {
  int rank = src.dims().size();
  PADDLE_ENFORCE_GE(
      rank, 2, platform::errors::InvalidArgument(
                   "'ReshapeToMatrix()' is only used for flatten high rank "
                   "tensors to matrixs. The dimensions of Tensor must be "
                   "greater or equal than 2. "
                   "But received dimensions of Tensor is %d",
                   rank));
  if (rank == 2) {
    return src;
  }
  Tensor res;
  res.ShareDataWith(src);
  res.Resize(flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

}  // namespace framework
}  // namespace paddle
