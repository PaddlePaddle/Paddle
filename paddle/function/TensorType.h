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

#include "paddle/math/Matrix.h"

namespace paddle {

enum ValueType {
  VALUE_TYPE_INT32 = 0,
  VALUE_TYPE_FLOAT = 1,
  VALUE_TYPE_DOUBLE = 2,
  VALUE_TYPE_BYTE = 3
};

enum DeviceType {
  DEVICE_TYPE_UNSPECIFIED = 0,
  DEVICE_TYPE_CPU = 1,
  DEVICE_TYPE_GPU = 2
};

enum SparseDataType { T_NO_VALUE = 0, T_FLOAT_VALUE = 1 };

enum SparseDataFormat { T_SPARSE_CSR = 0, T_SPARSE_CSC = 1 };

inline int sizeOfValuType(ValueType valueType) {
  if (valueType == VALUE_TYPE_INT32) {
    return 4;
  } else if (valueType == VALUE_TYPE_FLOAT) {
    return 4;
  } else if (valueType == VALUE_TYPE_DOUBLE) {
    return 8;
  } else {
    LOG(FATAL) << "Unknown type: " << valueType;
    return 0;
  }
}

template <typename T>
struct DataType;

template <>
struct DataType<float> {
  static const ValueType value = VALUE_TYPE_FLOAT;
};

template <>
struct DataType<double> {
  static const ValueType value = VALUE_TYPE_DOUBLE;
};

template <>
struct DataType<int> {
  static const ValueType value = VALUE_TYPE_INT32;
};

namespace detail {

template <typename VType, DeviceType Device>
struct MatrixT;

template <>
struct MatrixT<real, DEVICE_TYPE_CPU> {
  using type = CpuMatrix;
};

template <>
struct MatrixT<real, DEVICE_TYPE_GPU> {
  using type = GpuMatrix;
};

template <>
struct MatrixT<int, DEVICE_TYPE_CPU> {
  using type = void;  // Not implemented
};

template <>
struct MatrixT<int, DEVICE_TYPE_GPU> {
  using type = void;  // Not implemented
};

template <typename VType, DeviceType Device>
struct SparseMatrixT;

template <>
struct SparseMatrixT<real, DEVICE_TYPE_CPU> {
  using type = CpuSparseMatrix;
};

template <>
struct SparseMatrixT<real, DEVICE_TYPE_GPU> {
  using type = GpuSparseMatrix;
};

template <>
struct SparseMatrixT<int, DEVICE_TYPE_CPU> {
  using type = void;  // Not implemented
};

template <>
struct SparseMatrixT<int, DEVICE_TYPE_GPU> {
  using type = void;  // Not implemented
};

template <typename VType, DeviceType Device>
struct VectorT;

template <>
struct VectorT<real, DEVICE_TYPE_CPU> {
  using type = CpuVector;
};

template <>
struct VectorT<real, DEVICE_TYPE_GPU> {
  using type = GpuVector;
};

template <>
struct VectorT<int, DEVICE_TYPE_CPU> {
  using type = CpuIVector;
};

template <>
struct VectorT<int, DEVICE_TYPE_GPU> {
  using type = GpuIVector;
};

}  // namespace detail

template <typename VType, DeviceType DType>
struct Tensor {
  typedef typename detail::VectorT<VType, DType>::type Vector;
  typedef typename detail::MatrixT<VType, DType>::type Matrix;
  typedef typename detail::SparseMatrixT<VType, DType>::type SparseMatrix;
};

}  // namespace paddle
