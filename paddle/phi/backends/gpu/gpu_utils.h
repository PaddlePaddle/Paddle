// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#define EIGEN_USE_GPU

#include <array>

#include "paddle/phi/core/enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace phi {
namespace funcs {

template <typename T, int Size, T DefaultValue>
struct DeviceArray {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T& operator[](int index) const {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T& operator[](int index) {
    return data[index];
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceArray() {
    for (int i = 0; i < Size; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceArray(T a0) {
    data[0] = a0;
    for (int i = 1; i < Size; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceArray(T a0, T a1) {
    data[0] = a0;
    data[1] = a1;
    for (int i = 2; i < Size; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceArray(T a0, T a1, T a2) {
    data[0] = a0;
    data[1] = a1;
    data[2] = a2;
    for (int i = 3; i < Size; i++) {
      data[i] = DefaultValue;
    }
  }
  EIGEN_STRONG_INLINE DeviceArray(const std::array<T, Size>& sa) {
    for (int i = 0; i < Size; i++) {
      data[i] = sa[i];
    }
  }
  T data[Size];
};

template <typename T>
struct Dim3 : DeviceArray<T, 3, 1> {
  typedef DeviceArray<T, 3, 1> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3(T a0, T a1, T a2)
      : Base(a0, a1, a2) {}
};

// Flat index with real dimension
template <typename IndexType = int>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE IndexType
FlatTensorIndex(const Dim3<IndexType>& index, const Dim3<IndexType>& dims) {
  IndexType flat_index = index[0];
#pragma unroll
  for (int i = 1; i < 3; ++i) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// Convert index to tensor index with dimension.
template <typename IndexType = int>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3<IndexType> ConvertTensorIndex(
    IndexType index, const Dim3<IndexType>& dims) {
  Dim3<IndexType> tensor_index;
#pragma unroll
  for (int i = 2; i >= 0; --i) {
    IndexType new_index = index / dims[i];
    tensor_index[i] = static_cast<int>(index - dims[i] * new_index);
    index = new_index;
  }
  return tensor_index;
}

}  // namespace funcs
}  // namespace phi
