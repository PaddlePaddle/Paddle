/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

/**
 * This file provides a TensorCheck template function, which can be used to
 * compare CpuMatrix and GpuMatrix, CpuVector and GpuVector, and so on.
 */

#include <cmath>
#include "paddle/math/Matrix.h"

namespace autotest {

using paddle::Matrix;
using paddle::CpuMatrix;
using paddle::GpuMatrix;
using paddle::VectorT;
using paddle::CpuVectorT;
using paddle::GpuVectorT;

class AssertEqual {
public:
  AssertEqual(real err = 0) : err_(err) {}

  inline bool operator()(real a, real b) {
    if (err_ == 0) {
      if (a != b) {
        return false;
      }
    } else {
      if (std::fabs(a - b) > err_) {
        if ((std::fabs(a - b) / std::fabs(a)) > (err_ / 10.0f)) {
          return false;
        }
      }
    }

    return true;
  }

private:
  real err_;
};

template <typename Tensor>
class CopyToCpu;

template <>
class CopyToCpu<CpuMatrix> {
public:
  explicit CopyToCpu(const CpuMatrix& arg) : arg_(arg) {}
  const CpuMatrix& copiedArg() const { return arg_; }

private:
  const CpuMatrix& arg_;
};

template <>
class CopyToCpu<GpuMatrix> {
public:
  explicit CopyToCpu(const GpuMatrix& arg)
      : arg_(arg.getHeight(), arg.getWidth()) {
    arg_.copyFrom(arg);
  }
  CpuMatrix& copiedArg() { return arg_; }

private:
  CpuMatrix arg_;
};

template <>
class CopyToCpu<Matrix> {
public:
  explicit CopyToCpu(const Matrix& arg)
      : arg_(arg.getHeight(), arg.getWidth()) {
    arg_.copyFrom(arg);
  }
  CpuMatrix& copiedArg() { return arg_; }

private:
  CpuMatrix arg_;
};

template <typename T>
class CopyToCpu<CpuVectorT<T>> {
public:
  explicit CopyToCpu(const CpuVectorT<T>& arg) : arg_(arg) {}
  const CpuVectorT<T>& copiedArg() const { return arg_; }

private:
  const CpuVectorT<T>& arg_;
};

template <typename T>
class CopyToCpu<GpuVectorT<T>> {
public:
  explicit CopyToCpu(const GpuVectorT<T>& arg) : arg_(arg.getSize()) {
    arg_.copyFrom(arg);
  }
  CpuVectorT<T>& copiedArg() { return arg_; }

private:
  CpuVectorT<T> arg_;
};

template <typename T>
class CopyToCpu<VectorT<T>> {
public:
  explicit CopyToCpu(const VectorT<T>& arg) : arg_(arg.getSize()) {
    arg_.copyFrom(arg);
  }
  CpuVectorT<T>& copiedArg() { return arg_; }

private:
  CpuVectorT<T> arg_;
};

template <typename AssertEq>
void TensorCheck(AssertEq compare,
                 const CpuMatrix& matrix1,
                 const CpuMatrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();
  const real* data1 = matrix1.getData();
  const real* data2 = matrix2.getData();
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      real a = data1[i * width + j];
      real b = data2[i * width + j];
      if (!compare(a, b)) {
        count++;
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

template <typename AssertEq, class T>
void TensorCheck(AssertEq compare,
                 const CpuVectorT<T>& vector1,
                 const CpuVectorT<T>& vector2) {
  CHECK(vector1.getSize() == vector2.getSize());

  const T* data1 = vector1.getData();
  const T* data2 = vector2.getData();
  size_t size = vector1.getSize();
  int count = 0;
  for (size_t i = 0; i < size; i++) {
    real a = data1[i];
    real b = data2[i];
    if (!compare(a, b)) {
      count++;
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

template <typename AssertEq, typename Tensor1, typename Tensor2>
void TensorCheck(AssertEq compare,
                 const Tensor1& tensor1,
                 const Tensor2& tensor2) {
  TensorCheck(compare,
              CopyToCpu<Tensor1>(tensor1).copiedArg(),
              CopyToCpu<Tensor2>(tensor2).copiedArg());
}

template <typename AssertEq>
void TensorCheck(AssertEq compare, real args1, real args2) {
  EXPECT_EQ(compare(args1, args2), true) << "[Test error] args1 = " << args1
                                         << ", args2 = " << args2;
}

template <typename AssertEq>
void TensorCheck(AssertEq compare, size_t args1, size_t args2) {
  EXPECT_EQ(args1, args2) << "[Test error] args1 = " << args1
                          << ", args2 = " << args2;
}

template <typename Tensor1, typename Tensor2>
void TensorCheckEqual(const Tensor1& tensor1, const Tensor2& tensor2) {
  AssertEqual compare(0);
  TensorCheck(compare,
              CopyToCpu<Tensor1>(tensor1).copiedArg(),
              CopyToCpu<Tensor2>(tensor2).copiedArg());
}

template <typename Tensor1, typename Tensor2>
void TensorCheckErr(const Tensor1& tensor1, const Tensor2& tensor2) {
#ifndef PADDLE_TYPE_DOUBLE
  AssertEqual compare(1e-3);
#else
  AssertEqual compare(1e-10);
#endif
  TensorCheck(compare,
              CopyToCpu<Tensor1>(tensor1).copiedArg(),
              CopyToCpu<Tensor2>(tensor2).copiedArg());
}

}  // namespace autotest
