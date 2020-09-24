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
#include "paddle/fluid/platform/enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

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

struct Dim3 : DeviceArray<int, 3, 1> {
  typedef DeviceArray<int, 3, 1> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Dim3(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
  EIGEN_STRONG_INLINE Dim3(const std::array<int, 3>& array) : Base(array) {}
};

struct Index3 : DeviceArray<int, 3, 0> {
  typedef DeviceArray<int, 3, 0> Base;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3() : Base() {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3(int a0, int a1, int a2)
      : Base(a0, a1, a2) {}
};

// Flat index with real dimension
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int FlatTensorIndex(const Index3& index,
                                                          const Dim3& dims) {
  int flat_index = index[0];
  for (int i = 1; i < 3; i++) {
    flat_index = flat_index * dims[i] + index[i];
  }
  return flat_index;
}

// Convert index to tensor index with dimension.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index3
ConvertTensorIndex(int index, const Dim3& dims) {
  Index3 tensor_index;
  for (int i = 2; i >= 0; i--) {
    int new_index = index / dims[i];
    tensor_index[i] = index - dims[i] * new_index;
    index = new_index;
  }
  return tensor_index;
}

template <typename IntType, bool ceil>
IntType CeilOrFloor(IntType x, IntType deviser) {
  PADDLE_ENFORCE_GT(deviser, 0, platform::errors::InvalidArgument(
                                    "deviser should be greater than 0, "
                                    "but received is:%d",
                                    deviser));

  PADDLE_ENFORCE_GT(
      x, 0, platform::errors::InvalidArgument("input should be greater than 0, "
                                              "but received is:%d",
                                              x));

  const IntType round_to_zero = x / deviser;
  const IntType inte_result = round_to_zero * deviser;

  if (ceil) {
    const bool do_adjustment =
        (round_to_zero >= 0) && (deviser > 0 && x > inte_result);
    const IntType adjustment = static_cast<IntType>(do_adjustment);
    const IntType ceil_val = round_to_zero + adjustment;
    return ceil_val;
  } else {
    const bool do_adjustment =
        (round_to_zero <= 0) && (deviser > 0 && x < inte_result);

    const IntType adjustment = static_cast<IntType>(do_adjustment);
    const IntType floor_val = round_to_zero - adjustment;
    return floor_val;
  }
}

}  // namespace framework
}  // namespace paddle
