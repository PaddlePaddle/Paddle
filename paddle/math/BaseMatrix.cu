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

#include <cmath>
#include <string.h>
#include <paddle/utils/Logging.h>
#include "BaseMatrix.h"
#include "hl_matrix_ops.cuh"
#include "hl_matrix_base.cuh"
#include "hl_matrix_apply.cuh"
#include "SIMDFunctions.h"
#include "MathFunctions.h"

namespace paddle {

const char* SPARSE_SUPPORT_ERROR = "Sparse Matrix/Vector is not supported.";

template<class T>
template <class Op>
int BaseMatrixT<T>::applyUnary(Op op) {
  MatrixOffset offset(0, 0);
  applyUnary(op, height_, width_, offset);
  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyUnary(Op op, int numRows, int numCols,
                               MatrixOffset& offset) {
  CHECK(!this->isSparse()) << SPARSE_SUPPORT_ERROR;
  int dimM = numRows;
  int dimN = numCols;
  int lda = stride_;

  T* A = data_;
  CAL_MATRIX_START_ADDRESS(A, height_, width_, lda, offset.aCol_, offset.aRow_);

  CHECK_LE(dimM + offset.aRow_, this->height_);
  CHECK_LE(dimN + offset.aCol_, this->width_);
  if (true == useGpu_) {
    hl_gpu_apply_unary_op(op, A, dimM, dimN, lda);
  } else {
    hl_cpu_apply_unary_op(op, A, dimM, dimN, lda);
  }
  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyBinary(Op op, BaseMatrixT& b) {
  CHECK(height_ == b.height_ && width_ == b.width_)
      << "Matrix dimensions are not equal";

  MatrixOffset offset(0, 0, 0, 0);
  applyBinary(op, b, height_, width_, offset);
  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyBinary(Op op, BaseMatrixT& b, int numRows, int numCols,
                                MatrixOffset& offset) {
  applyBinary(op, b, numRows, numCols, offset, false_type(), false_type());
  return 0;
}

template<class T>
template <class Op, class bAsRowVector, class bAsColVector>
int BaseMatrixT<T>::applyBinary(Op op, BaseMatrixT& b, int numRows, int numCols,
                            MatrixOffset& offset, bAsRowVector, bAsColVector) {
  CHECK(!this->isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!b.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(useGpu_ == b.useGpu_) << "Matrix type mismatch";

  int dimM = numRows;
  int dimN = numCols;
  int lda = stride_;
  int ldb = b.stride_;

  T* A = data_;
  T* B = b.data_;
  CAL_MATRIX_START_ADDRESS(A, height_, width_, lda, offset.aCol_, offset.aRow_);
  CAL_MATRIX_START_ADDRESS(B, b.height_, b.width_, ldb, offset.bCol_,
                           offset.bRow_);
  CHECK_LE(dimM + offset.aRow_, this->height_);
  CHECK_LE(dimN + offset.aCol_, this->width_);
  if (!bAsRowVector::value && !bAsColVector::value) {
    CHECK_LE(dimM + offset.bRow_, b.height_);
    CHECK_LE(dimN + offset.bCol_, b.width_);
  } else if (bAsRowVector::value && !bAsColVector::value) {
    CHECK_LE(dimN + offset.bCol_, b.width_);
  } else if (!bAsRowVector::value && bAsColVector::value) {
    CHECK_LE(dimM + offset.bRow_, b.height_);
  } else {
  }
  if (true == useGpu_) {
    hl_gpu_apply_binary_op<T, Op, bAsRowVector::value, bAsColVector::value>(
        op, A, B, dimM, dimN, lda, ldb);
  } else {
    hl_cpu_apply_binary_op<T, Op, bAsRowVector::value, bAsColVector::value>(
        op, A, B, dimM, dimN, lda, ldb);
  }

  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyTernary(Op op, BaseMatrixT& b, BaseMatrixT& c) {
  CHECK_EQ(height_, b.height_);
  CHECK_EQ(width_, b.width_);
  CHECK_EQ(height_, c.height_);
  CHECK_EQ(width_, c.width_);

  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  applyTernary(op, b, c, height_, width_, offset);

  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyTernary(Op op, BaseMatrixT& b, BaseMatrixT& c,
                                 int numRows, int numCols,
                                 MatrixOffset& offset) {
  applyTernary(op, b, c, numRows, numCols, offset, false_type(), false_type());

  return 0;
}

template<class T>
template <class Op, class cAsRowVector, class cAsColVector>
int BaseMatrixT<T>::applyTernary(Op op, BaseMatrixT& b, BaseMatrixT& c,
                                 int numRows, int numCols, MatrixOffset& offset,
                                 cAsRowVector, cAsColVector) {
  CHECK(!this->isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!b.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!c.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK_EQ(useGpu_, b.useGpu_);
  CHECK_EQ(useGpu_, c.useGpu_);

  int dimM = numRows;
  int dimN = numCols;
  int lda = stride_;
  int ldb = b.stride_;
  int ldc = c.stride_;

  T* A = data_;
  T* B = b.data_;
  T* C = c.data_;
  CAL_MATRIX_START_ADDRESS(A, height_, width_, lda, offset.aCol_, offset.aRow_);
  CAL_MATRIX_START_ADDRESS(B, b.height_, b.width_, ldb, offset.bCol_,
                           offset.bRow_);
  CAL_MATRIX_START_ADDRESS(C, c.height_, c.width_, ldc, offset.cCol_,
                           offset.cRow_);

  CHECK_LE(dimM + offset.aRow_, this->height_);
  CHECK_LE(dimN + offset.aCol_, this->width_);
  CHECK_LE(dimM + offset.bRow_, b.height_);
  CHECK_LE(dimN + offset.bCol_, b.width_);
  if (!cAsRowVector::value && !cAsColVector::value) {
    CHECK_LE(dimM + offset.cRow_, c.height_);
    CHECK_LE(dimN + offset.cCol_, c.width_);
  } else if (cAsRowVector::value && !cAsColVector::value) {
    CHECK_LE(dimN + offset.cCol_, c.width_);
  } else if (!cAsRowVector::value && cAsColVector::value) {
    CHECK_LE(dimM + offset.cRow_, c.height_);
  } else {
  }

  if (true == useGpu_) {
    hl_gpu_apply_ternary_op
      <T, Op, cAsRowVector::value, cAsColVector::value>(
        op, A, B, C, dimM, dimN, lda, ldb, ldc);
  } else {
    hl_cpu_apply_ternary_op
      <T, Op, cAsRowVector::value, cAsColVector::value>(
        op, A, B, C, dimM, dimN, lda, ldb, ldc);
  }

  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyQuaternary(Op op, BaseMatrixT& b, BaseMatrixT& c,
                                    BaseMatrixT& d) {
  CHECK_EQ(height_, b.height_);
  CHECK_EQ(width_, b.width_);
  CHECK_EQ(height_, c.height_);
  CHECK_EQ(width_, c.width_);
  CHECK_EQ(height_, d.height_);
  CHECK_EQ(width_, d.width_);

  MatrixOffset offset(0, 0, 0, 0, 0, 0, 0, 0);
  applyQuaternary(op, b, c, d, height_, width_, offset);

  return 0;
}

template<class T>
template <class Op>
int BaseMatrixT<T>::applyQuaternary(Op op, BaseMatrixT& b, BaseMatrixT& c,
                                    BaseMatrixT& d, int numRows, int numCols,
                                    MatrixOffset& offset) {
  CHECK(!this->isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!b.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!c.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK(!d.isSparse()) << SPARSE_SUPPORT_ERROR;
  CHECK_EQ(useGpu_, b.useGpu_);
  CHECK_EQ(useGpu_, c.useGpu_);
  CHECK_EQ(useGpu_, d.useGpu_);

  int dimM = numRows;
  int dimN = numCols;
  int lda = stride_;
  int ldb = b.stride_;
  int ldc = c.stride_;
  int ldd = d.stride_;

  T* A = data_;
  T* B = b.data_;
  T* C = c.data_;
  T* D = d.data_;
  CAL_MATRIX_START_ADDRESS(A, height_, width_, lda, offset.aCol_, offset.aRow_);
  CAL_MATRIX_START_ADDRESS(B, b.height_, b.width_, ldb, offset.bCol_,
                           offset.bRow_);
  CAL_MATRIX_START_ADDRESS(C, c.height_, c.width_, ldc, offset.cCol_,
                           offset.cRow_);
  CAL_MATRIX_START_ADDRESS(D, d.height_, d.width_, ldd, offset.dCol_,
                           offset.dRow_);

  CHECK_LE(dimM + offset.aRow_, this->height_);
  CHECK_LE(dimN + offset.aCol_, this->width_);
  CHECK_LE(dimM + offset.bRow_, b.height_);
  CHECK_LE(dimN + offset.bCol_, b.width_);
  CHECK_LE(dimM + offset.cRow_, c.height_);
  CHECK_LE(dimN + offset.cCol_, c.width_);
  CHECK_LE(dimM + offset.dRow_, d.height_);
  CHECK_LE(dimN + offset.dCol_, d.width_);
  if (true == useGpu_) {
    hl_gpu_apply_quaternary_op(op, A, B, C, D, dimM, dimN, lda, ldb,
                               ldc, ldd);
  } else {
    hl_cpu_apply_quaternary_op(op, A, B, C, D, dimM, dimN, lda, ldb,
                               ldc, ldd);
  }

  return 0;
}

template<class T>
template <class Agg, class Op, class Saver, class aAsRowVector,
          class aAsColVector>
int BaseMatrixT<T>::aggregate(Agg agg, Op op, Saver sv, BaseMatrixT& b,
                              int numRows, int numCols, MatrixOffset& offset,
                              aAsRowVector, aAsColVector) {
  CHECK_EQ(useGpu_, b.useGpu_);

  int ld = stride_;
  int ldb = b.stride_;

  T* dst = data_;
  T* B = b.data_;
  CAL_MATRIX_START_ADDRESS(dst, height_, width_, ld, offset.aCol_,
                           offset.aRow_);
  CAL_MATRIX_START_ADDRESS(B, b.height_, b.width_, ldb, offset.bCol_,
                           offset.bRow_);

  if (aAsRowVector::value && !aAsColVector::value) {
    if (useGpu_) {
      hl_gpu_matrix_column_op(agg, op, sv, numRows, numCols, dst, B, ldb);
    } else {
      hl_cpu_matrix_column_op(agg, op, sv, numRows, numCols, dst, B, ldb);
    }
  } else if (!aAsRowVector::value && aAsColVector::value) {
    if (useGpu_) {
      hl_gpu_matrix_row_op(agg, op, sv, numRows, numCols, dst, ld, B, ldb);
    } else {
      hl_cpu_matrix_row_op(agg, op, sv, numRows, numCols, dst, ld, B, ldb);
    }
  } else {
    LOG(FATAL) << "not supported";
  }

  return 0;
}

template<class T>
template <class Agg, class Op, class Saver, class aAsRowVector,
          class aAsColVector>
int BaseMatrixT<T>::aggregate(Agg agg, Op op, Saver sv, BaseMatrixT& b,
                              BaseMatrixT& c, int numRows, int numCols,
                              MatrixOffset& offset, aAsRowVector,
                              aAsColVector) {
  CHECK_EQ(useGpu_, b.useGpu_);
  CHECK_EQ(useGpu_, c.useGpu_);

  int ld = stride_;
  int ldb = b.stride_;
  int ldc = c.stride_;

  T* dst = data_;
  T* B = b.data_;
  T* C = c.data_;
  CAL_MATRIX_START_ADDRESS(dst, height_, width_, ld, offset.aCol_,
                           offset.aRow_);
  CAL_MATRIX_START_ADDRESS(B, b.height_, b.width_, ldb, offset.bCol_,
                           offset.bRow_);
  CAL_MATRIX_START_ADDRESS(C, c.height_, c.width_, ldc, offset.cCol_,
                           offset.cRow_);

  if (aAsRowVector::value && !aAsColVector::value) {
    if (useGpu_) {
      hl_gpu_matrix_column_op(agg, op, sv, numRows, numCols, dst, B,
                              ldb, C, ldc);
    } else {
      hl_cpu_matrix_column_op(agg, op, sv, numRows, numCols, dst, B,
                              ldb, C, ldc);
    }
  } else if (!aAsRowVector::value && aAsColVector::value) {
    if (useGpu_) {
      hl_gpu_matrix_row_op(agg, op, sv, numRows, numCols, dst, ld, B,
                           ldb, C, ldc);
    } else {
      hl_cpu_matrix_row_op(agg, op, sv, numRows, numCols, dst, ld, B,
                           ldb, C, ldc);
    }
  } else {
    LOG(FATAL) << "not supported";
  }

  return 0;
}

/**
 * @brief   unary operator.
 *
 */

DEFINE_MATRIX_UNARY_OP(Neg, a = -a);
template<class T>
void BaseMatrixT<T>::neg() { applyUnary(unary::Neg<T>()); }

DEFINE_MATRIX_UNARY_OP(Exp, a = exp(a));
template<>
void BaseMatrixT<real>::exp2() { applyUnary(unary::Exp<real>()); }

DEFINE_MATRIX_UNARY_OP(Log, a = log(a));
template<>
void BaseMatrixT<real>::log2() {
  if (useGpu_) {
    applyUnary(unary::Log<real>());
  } else {
    vLog(height_ * width_, data_, data_);
  }
}

DEFINE_MATRIX_UNARY_OP(Sqrt, a = sqrt(a));
template<>
void BaseMatrixT<real>::sqrt2() { applyUnary(unary::Sqrt<real>()); }

DEFINE_MATRIX_UNARY_OP(Square, a = a * a);
template<class T>
void BaseMatrixT<T>::square2() { applyUnary(unary::Square<T>()); }

DEFINE_MATRIX_UNARY_OP(Reciprocal, a = 1.0f / a);
template<class T>
void BaseMatrixT<T>::reciprocal2() { applyUnary(unary::Reciprocal<T>()); }

DEFINE_MATRIX_UNARY_OP(Abs, a = a > 0 ? a : -a);
template<class T>
void BaseMatrixT<T>::abs2() { applyUnary(unary::Abs<T>()); }

DEFINE_MATRIX_UNARY_OP(Sign, a = (a > 0) - (a < 0));
template<class T>
void BaseMatrixT<T>::sign2() { applyUnary(unary::Sign<T>()); }

DEFINE_MATRIX_UNARY_OP(Zero, a = 0);
template<class T>
void BaseMatrixT<T>::zero() { applyUnary(unary::Zero<T>()); }

template<class T>
void BaseMatrixT<T>::zeroAtOffset(int64_t columnOffset, int64_t numColumns) {
  int numRows = height_;
  int numCols = numColumns;
  MatrixOffset offset(columnOffset, 0);
  applyUnary(unary::Zero<T>(), numRows, numCols, offset);
}

DEFINE_MATRIX_UNARY_OP(One, a = 1);
template<class T>
void BaseMatrixT<T>::one() { applyUnary(unary::One<T>()); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(Pow, ONE_PARAMETER, a = pow(a, p));
template<>
void BaseMatrixT<real>::pow2(real p) {
  if (useGpu_) {
    applyUnary(unary::Pow<real>(p));
  } else {
    vPow(height_ * width_, data_, p, data_);
  }
}

DEFINE_MATRIX_UNARY_PARAMETER_OP(SubScalar, ONE_PARAMETER, a -= p);
template<class T>
void BaseMatrixT<T>::subScalar(T p) { applyUnary(unary::SubScalar<T>(p)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(MulScalar, ONE_PARAMETER, a *= p);
template<class T>
void BaseMatrixT<T>::mulScalar(T p) { applyUnary(unary::MulScalar<T>(p)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(DivScalar, ONE_PARAMETER, a /= p);
template<class T>
void BaseMatrixT<T>::divScalar(T p) { applyUnary(unary::DivScalar<T>(p)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(Assign, ONE_PARAMETER, a = p);
template<class T>
void BaseMatrixT<T>::assign(T p) { applyUnary(unary::Assign<T>(p)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(Add, ONE_PARAMETER, a += p);
template<class T>
void BaseMatrixT<T>::add(T p) { applyUnary(unary::Add<T>(p)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(Add2, TWO_PARAMETER, a = a * p1 + p2);
template<class T>
void BaseMatrixT<T>::add(T p1, T p2) { applyUnary(unary::Add2<T>(p1, p2)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(Clip, TWO_PARAMETER,
                                 a = a < p1 ? p1 : (a > p2 ? p2 : a));
template<class T>
void BaseMatrixT<T>::clip(T p1, T p2) { applyUnary(unary::Clip<T>(p1, p2)); }

DEFINE_MATRIX_UNARY_PARAMETER_OP(BiggerThanScalar, ONE_PARAMETER,
                                 a = a > p ? 1.0f : 0.0f);
template<class T>
void BaseMatrixT<T>::biggerThanScalar(T p) {
  applyUnary(unary::BiggerThanScalar<T>(p));
}

DEFINE_MATRIX_UNARY_PARAMETER_OP(DownClip, ONE_PARAMETER,
                                 a = a > p ? a : p);
template<class T>
void BaseMatrixT<T>::downClip(T p) {
  applyUnary(unary::DownClip<T>(p));
}

/**
 * @brief   binary operator.
 *
 */

DEFINE_MATRIX_BINARY_OP(Add, a += b);
template<class T>
void BaseMatrixT<T>::add(BaseMatrixT& b) {
  applyBinary(binary::Add<T>(), b);
}

template<>
void BaseMatrixT<real>::add(BaseMatrixT& b) {
  if (useGpu_) {
    applyBinary(binary::Add<real>(), b);
  } else {  // cpu branch
    CHECK_EQ(height_, b.height_);
    CHECK_EQ(width_, b.width_);
    vAdd(height_ * width_, data_, b.data_, data_);
  }
}

template<class T>
void BaseMatrixT<T>::addAtOffset(BaseMatrixT& b, int64_t columnOffset) {
  if (columnOffset + b.width_ <= width_) {
    int numRows = height_;
    int numCols = b.width_;
    MatrixOffset offset(columnOffset, 0, 0, 0);
    applyBinary(binary::Add<T>(), b, numRows, numCols, offset);
  } else if (columnOffset + width_ <= b.width_) {
    int numRows = height_;
    int numCols = width_;
    MatrixOffset offset(0, 0, columnOffset, 0);
    applyBinary(binary::Add<T>(), b, numRows, numCols, offset);
  } else {
    LOG(FATAL) << "Wrong argument "
               << " a.width=" << width_ << " b.width=" << b.width_
               << " columnOffset=" << columnOffset;
  }
}

template<class T>
void BaseMatrixT<T>::addP2P(BaseMatrixT& b) {
  T* A = data_;
  T* B = b.data_;
  int dimM = height_;
  int dimN = width_;

  hl_gpu_apply_binary_op<T, binary::Add<T>, 0, 0>
    (binary::Add<T>(), A, B, dimM, dimN, dimN, dimN);
}

template<class T>
void BaseMatrixT<T>::addColVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::Add<T>(), b, numRows, numCols, offset, false_type(),
              true_type() /* bAsColVector */);
}

template<class T>
void BaseMatrixT<T>::addRowVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::Add<T>(), b, numRows, numCols, offset,
              true_type() /* bAsRowVector */, false_type());
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(Add1, ONE_PARAMETER, a += b * p);
template<class T>
void BaseMatrixT<T>::add(BaseMatrixT& b, T p) {
  applyBinary(binary::Add1<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(Pow, ONE_PARAMETER, a = pow(b, p));
template<>
void BaseMatrixT<real>::pow2(BaseMatrixT& b, real p) {
  if (useGpu_) {
    applyBinary(binary::Pow<real>(p), b);
  } else {
    vPow(height_ * width_, b.data_, p, data_);
  }
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(Add2, TWO_PARAMETER, a = p1 * a + p2 * b);
template<class T>
void BaseMatrixT<T>::add(BaseMatrixT& b, T p1, T p2) {
  applyBinary(binary::Add2<T>(p1, p2), b);
}

template<class T>
void BaseMatrixT<T>::addBias(BaseMatrixT& b, T scale) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::Add1<T>(scale), b, numRows, numCols, offset,
              true_type() /* bAsRowVector */, false_type());
}

DEFINE_MATRIX_BINARY_OP(Sub, a -= b);
template<class T>
void BaseMatrixT<T>::sub(BaseMatrixT& b) { applyBinary(binary::Sub<T>(), b); }

DEFINE_MATRIX_BINARY_PARAMETER_OP(Sub1, ONE_PARAMETER, a -= b * p);
template<class T>
void BaseMatrixT<T>::sub(BaseMatrixT& b, T p) {
  applyBinary(binary::Sub1<T>(p), b);
}

DEFINE_MATRIX_BINARY_OP(Relu, b = a > 0.0f ? a : 0.0f);
template<class T>
void BaseMatrixT<T>::relu(BaseMatrixT& b) { applyBinary(binary::Relu<T>(), b); }

DEFINE_MATRIX_BINARY_OP(ReluDerivative, a *= (b > 0.0f ? 1.0f : 0.0f));
template<class T>
void BaseMatrixT<T>::reluDerivative(BaseMatrixT& b) {
  applyBinary(binary::ReluDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(Softrelu, const T THRESHOLD = 40.0;
                        b = log(1.0 + exp((a > THRESHOLD)
                                              ? THRESHOLD
                                              : ((a < -THRESHOLD) ? (-THRESHOLD)
                                                                  : a))));
template<>
void BaseMatrixT<real>::softrelu(BaseMatrixT& b) {
  applyBinary(binary::Softrelu<real>(), b);
}

DEFINE_MATRIX_BINARY_OP(
    SoftreluDerivative, const T THRESHOLD = 40.0;
    a *= (1.0 - exp(-1.0 * ((b > THRESHOLD)
                                ? THRESHOLD
                                : ((b < -THRESHOLD) ? (-THRESHOLD) : b)))));
template<>
void BaseMatrixT<real>::softreluDerivative(BaseMatrixT& b) {
  applyBinary(binary::SoftreluDerivative<real>(), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(Brelu, TWO_PARAMETER, b = a > p1 ? a : p1;
                                  b = b < p2 ? b : p2);
template<class T>
void BaseMatrixT<T>::brelu(BaseMatrixT& b) {
  int p1 = 0, p2 = 24;    //! TODO(yuyang18): Make p1,p2 configuable.
  applyBinary(binary::Brelu<T>(p1, p2), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(BreluDerivative, TWO_PARAMETER,
                                  a *= (b > p1 && b < p2) ? 1.0 : 0.0);
template<class T>
void BaseMatrixT<T>::breluDerivative(BaseMatrixT& b) {
  int p1 = 0, p2 = 24;
  applyBinary(binary::BreluDerivative<T>(p1, p2), b);
}

DEFINE_MATRIX_BINARY_OP(Square, b = a * a);
template<class T>
void BaseMatrixT<T>::square2(BaseMatrixT& b) {
  applyBinary(binary::Square<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(SquareDerivative, a *= 2.0 * b);
template<class T>
void BaseMatrixT<T>::squareDerivative(BaseMatrixT& b) {
  applyBinary(binary::SquareDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(Tanh,
    T tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    b = 2.0 / (1.0 + std::exp(tmp)) - 1.0);
template<>
void BaseMatrixT<real>::tanh(BaseMatrixT& b) {
  applyBinary(binary::Tanh<real>(), b);
}

DEFINE_MATRIX_BINARY_OP(TanhDerivative, a *= 1 - b * b);
template<class T>
void BaseMatrixT<T>::tanhDerivative(BaseMatrixT& b) {
  applyBinary(binary::TanhDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(ScaledTanh, TWO_PARAMETER,
                                  b = p1 *
                                      (2.0 / (1.0 + exp(-2 * p2 * a)) - 1.0));
template<>
void BaseMatrixT<real>::scaledTanh(BaseMatrixT& b, real p1, real p2) {
  applyBinary(binary::ScaledTanh<real>(p1, p2), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(ScaledTanhDerivative, TWO_PARAMETER,
                                  a *= p2 * (p1 - b * b));
template<class T>
void BaseMatrixT<T>::scaledTanhDerivative(BaseMatrixT& b, T p1, T p2) {
  applyBinary(binary::ScaledTanhDerivative<T>(p1 * p1, p2 / p1), b);
}

DEFINE_MATRIX_BINARY_OP(Reciprocal, b = 1.0f / a);
template<class T>
void BaseMatrixT<T>::reciprocal2(BaseMatrixT& b) {
  applyBinary(binary::Reciprocal<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(ReciprocalDerivative, a *= -b * b);
template<class T>
void BaseMatrixT<T>::reciprocalDerivative(BaseMatrixT& b) {
  applyBinary(binary::ReciprocalDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(Abs, b = a > 0.0f ? a : -a);
template<class T>
void BaseMatrixT<T>::abs2(BaseMatrixT& b) { applyBinary(binary::Abs<T>(), b); }

DEFINE_MATRIX_BINARY_OP(AbsDerivative, a = (b > 0) ? a : (b < 0) ? -a : 0);
template<class T>
void BaseMatrixT<T>::absDerivative(BaseMatrixT& b) {
  applyBinary(binary::AbsDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(
    Sigmoid, const T THRESHOLD_MIN = -40.0; const T THRESHOLD_MAX = 13.0;
    T tmp = (a < THRESHOLD_MIN) ? THRESHOLD_MIN
                                   : ((a > THRESHOLD_MAX) ? THRESHOLD_MAX : a);
    b = 1.0f / (1.0f + exp(-tmp)));
template<>
void BaseMatrixT<real>::sigmoid(BaseMatrixT& b) {
  if (useGpu_) {
    applyBinary(binary::Sigmoid<real>(), b);
  } else {  // cpu versioni
    size_t numSamples = this->height_;
    size_t dim = this->width_;
    CHECK_EQ(b.height_, numSamples);
    CHECK_EQ(b.width_, dim);
    const real* in = this->data_;
    real* out = b.data_;

    // out = - in
    const float THRESHOLD_MIN = -40.0;  // make sure sigmoid(x) > 0
    const float THRESHOLD_MAX = 13.0;   // make sure sigmoid(x) < 1
    for (size_t i = 0; i < numSamples * dim; ++i) {
      real tmp = in[i];
      tmp = (tmp < THRESHOLD_MIN)
                ? THRESHOLD_MIN
                : ((tmp > THRESHOLD_MAX) ? THRESHOLD_MAX : tmp);
      out[i] = -tmp;
    }

    // out = exp(out)
    vExp(numSamples * dim, out, out);

    // out = 1 / (1 + out)
    for (size_t i = 0; i < numSamples * dim; ++i) {
      out[i] = 1 / (1 + out[i]);
    }
  }
}

DEFINE_MATRIX_BINARY_OP(SigmoidDerivative, a *= b * (1 - b));
template<class T>
void BaseMatrixT<T>::sigmoidDerivative(BaseMatrixT& b) {
  applyBinary(binary::SigmoidDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(ExpDerivative, a *= b);
template<class T>
void BaseMatrixT<T>::expDerivative(BaseMatrixT& b) {
  applyBinary(binary::ExpDerivative<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(Sign, b = a > 0.0f ? 1.0f : -1.0f);
template<class T>
void BaseMatrixT<T>::sign2(BaseMatrixT& b) {
  applyBinary(binary::Sign<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(Exp, a = exp(b));
template<>
void BaseMatrixT<real>::exp2(BaseMatrixT& b) {
  applyBinary(binary::Exp<real>(), b);
}

DEFINE_MATRIX_BINARY_OP(Log, a = log(b));
template<>
void BaseMatrixT<real>::log2(BaseMatrixT& b) {
  if (useGpu_) {
    applyBinary(binary::Log<real>(), b);
  } else {
    vLog(height_ * width_, b.data_, data_);
  }
}

DEFINE_MATRIX_BINARY_OP(Sqrt, a = sqrt(b));
template<>
void BaseMatrixT<real>::sqrt2(BaseMatrixT& b) {
  applyBinary(binary::Sqrt<real>(), b);
}

DEFINE_MATRIX_BINARY_OP(InvSqrt, a = 1.0f / sqrt(b));
template<>
void BaseMatrixT<real>::invSqrt(BaseMatrixT& b) {
  if (useGpu_) {
    applyBinary(binary::InvSqrt<real>(), b);
  } else {  // cpu branch
    CHECK_EQ(height_, b.height_);
    CHECK_EQ(width_, b.width_);
    vInvSqrt(height_ * width_, b.data_, data_);
  }
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(IsEqual, ONE_PARAMETER, a = (b == p));
template<class T>
void BaseMatrixT<T>::isEqualTo(BaseMatrixT& b, T value) {
  applyBinary(binary::IsEqual<T>(value), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(AddScalar, ONE_PARAMETER, a = b + p);
template<class T>
void BaseMatrixT<T>::addScalar(BaseMatrixT& b, T p) {
  applyBinary(binary::AddScalar<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(SubScalar, ONE_PARAMETER, a = b - p);
template<class T>
void BaseMatrixT<T>::subScalar(BaseMatrixT& b, T p) {
  applyBinary(binary::SubScalar<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(MulScalar, ONE_PARAMETER, a = b * p);
template<class T>
void BaseMatrixT<T>::mulScalar(BaseMatrixT& b, T p) {
  applyBinary(binary::MulScalar<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(DivScalar, ONE_PARAMETER, a = b / p);
template<class T>
void BaseMatrixT<T>::divScalar(BaseMatrixT& b, T p) {
  applyBinary(binary::DivScalar<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(ScalarDiv, ONE_PARAMETER, a = p / b);
template<class T>
void BaseMatrixT<T>::scalarDiv(BaseMatrixT& b, T p) {
  applyBinary(binary::ScalarDiv<T>(p), b);
}

/**
 * @brief   ternary operator.
 *
 */

DEFINE_MATRIX_TERNARY_OP(SoftCrossEntropy,
                         a = -c * log(b) - (1 - c) * log(1 - b));
template<>
void BaseMatrixT<real>::softCrossEntropy(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::SoftCrossEntropy<real>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(SoftCrossEntropyBp, a += (b - c) / (b * (1 - b)));
template<class T>
void BaseMatrixT<T>::softCrossEntropyBp(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::SoftCrossEntropyBp<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(BinaryCrossEntropy,
                         a = c > 0.5 ? -log(b) : -log(1.0 - b));
template<>
void BaseMatrixT<real>::binaryLabelCrossEntropy(BaseMatrixT& b,
                                                BaseMatrixT& c) {
  if (useGpu_) {
    applyTernary(ternary::BinaryCrossEntropy<real>(), b, c);
  } else {
    CHECK_EQ(height_, b.height_);
    CHECK_EQ(height_, c.height_);
    CHECK_EQ(width_, b.width_);
    CHECK_EQ(width_, c.width_);

    size_t size = height_ * width_;
    real* out = b.data_;
    real* label = c.data_;
    real* cost = data_;

    for (size_t i = 0; i < size; ++i) {
      cost[i] = label[i] > 0.5 ? out[i] : 1.0 - out[i];
    }
    vLog(size, cost, cost);
    for (size_t i = 0; i < size; ++i) {
      cost[i] *= -1.0;
    }
  }
}

DEFINE_MATRIX_TERNARY_OP(BinaryCrossEntropyBp,
                         a += c > 0.5 ? -1.0 / b : 1.0 / (1.0 - b));
template<class T>
void BaseMatrixT<T>::binaryLabelCrossEntropyBp(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::BinaryCrossEntropyBp<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(Add, a = b + c);
template<class T>
void BaseMatrixT<T>::add(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::Add<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(Add1, TWO_PARAMETER, a = p1 * b + p2 * c);
template<class T>
void BaseMatrixT<T>::add(BaseMatrixT& b, T p1, BaseMatrixT& c, T p2) {
  applyTernary(ternary::Add1<T>(p1, p2), b, c);
}

DEFINE_MATRIX_TERNARY_OP(Sub, a = b - c);
template<class T>
void BaseMatrixT<T>::sub(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::Sub<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(Sub1, TWO_PARAMETER, a = p1 * b - p2 * c);
template<class T>
void BaseMatrixT<T>::sub(BaseMatrixT& b, T p1, BaseMatrixT& c, T p2) {
  applyTernary(ternary::Sub1<T>(p1, p2), b, c);
}

DEFINE_MATRIX_TERNARY_OP(Add2, a = a + b + c);
template<class T>
void BaseMatrixT<T>::add2(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::Add2<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(Add3, THREE_PARAMETER,
                                   a = p1 * a + p2 * b + p3 * c);
template<class T>
void BaseMatrixT<T>::add2(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2, T p3) {
  applyTernary(ternary::Add3<T>(p1, p2, p3), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(SgdUpdate, THREE_PARAMETER,
                                   c = p2 * c - p1 * (b + p3 * a);
                                   a = a + c);
template<class T>
void BaseMatrixT<T>::sgdUpdate(BaseMatrixT& b,  // grad
                               BaseMatrixT& c,  // mom
                               T p1,        // learningRate,
                               T p2,        // momentum,
                               T p3) {      // decayRate
  applyTernary(ternary::SgdUpdate<T>(p1, p2, p3), b, c);
}

DEFINE_MATRIX_QUATERNARY_PARAMETER_OP(SgdUpdate, THREE_PARAMETER,
                                      c = p2 * c - p1 * d * (b + p3 * a);
                                      a += c);
template<class T>
void BaseMatrixT<T>::sgdUpdate(BaseMatrixT& b,  // grad,
                               BaseMatrixT& c,  // mom,
                               BaseMatrixT& d,  // lr,
                               T p1,        // learningRate,
                               T p2,        // momentum,
                               T p3) {      // decayRate
  applyQuaternary(quaternary::SgdUpdate<T>(p1, p2, p3), b, c, d);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(ApplyL1, ONE_PARAMETER, T lambda = p * b;
                                  a = (a > lambda)
                                          ? (a - lambda)
                                          : (a < -lambda) ? (a + lambda) : 0);
template<class T>
void BaseMatrixT<T>::applyL1(BaseMatrixT& lr, T learningRate, T decayRate) {
  applyBinary(binary::ApplyL1<T>(learningRate * decayRate), lr);
}

template<>
void BaseMatrixT<real>::applyL1(BaseMatrixT& lr,
                                real learningRate,
                                real decayRate) {
  if (useGpu_) {
    applyBinary(binary::ApplyL1<real>(learningRate * decayRate), lr);
  } else {
    simd::decayL1(this->data_, this->data_, lr.data_, learningRate * decayRate,
                  height_ * width_);
  }
}

DEFINE_MATRIX_UNARY_PARAMETER_OP(ApplyL1, ONE_PARAMETER, T lambda = p;
                                 a = (a > lambda)
                                         ? (a - lambda)
                                         : (a < -lambda) ? (a + lambda) : 0);
template<class T>
void BaseMatrixT<T>::applyL1(T learningRate, T decayRate) {
  applyUnary(unary::ApplyL1<T>(learningRate * decayRate));
}

template<>
void BaseMatrixT<real>::applyL1(real learningRate, real decayRate) {
  if (useGpu_) {
    applyUnary(unary::ApplyL1<real>(learningRate * decayRate));
  } else {
    simd::decayL1(this->data_, this->data_, learningRate * decayRate,
                  height_ * width_);
  }
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(ApplyL2, ONE_PARAMETER,
                                  a *= (1.0f / (1.0f + p * b)));
template<class T>
void BaseMatrixT<T>::applyL2(BaseMatrixT& lr, T learningRate, T decayRate) {
  if (useGpu_) {
    applyBinary(binary::ApplyL2<T>(learningRate * decayRate), lr);
  } else {
    size_t size = this->height_ * this->width_;
    T decay = learningRate * decayRate;
    for (size_t j = 0; j < size; ++j) {
      this->data_[j] *= 1.0f / (1.0f + decay * lr.data_[j]);
    }
  }
}

template<class T>
void BaseMatrixT<T>::applyL2(T learningRate, T decayRate) {
  BaseMatrixT<T>::mulScalar(1.0f / (1.0f + learningRate * decayRate));
}

DEFINE_MATRIX_BINARY_OP(DotMul, a *= b);
template<class T>
void BaseMatrixT<T>::dotMul(BaseMatrixT& b) {
  applyBinary(binary::DotMul<T>(), b);
}

DEFINE_MATRIX_TERNARY_OP(DotMul, a = b * c);
template<class T>
void BaseMatrixT<T>::dotMul(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::DotMul<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(DotDiv, a = (b == 0.0) ? 0.0 : b / c);
template<class T>
void BaseMatrixT<T>::dotDiv(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::DotDiv<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(DotDiv2P, TWO_PARAMETER,
                                   a = (b + p1) / (c + p2));
template<class T>
void BaseMatrixT<T>::dotDiv(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2) {
  applyTernary(ternary::DotDiv2P<T>(p1, p2), b, c);
}

DEFINE_MATRIX_QUATERNARY_OP(RankLoss, const T THRESHOLD = 40.0; a = b - c;
                            a = (a > THRESHOLD)
                                    ? THRESHOLD
                                    : ((a < -THRESHOLD) ? (-THRESHOLD) : a);
                            a = log(1 + exp(a)) - a * d);
template<>
void BaseMatrixT<real>::rankLoss(BaseMatrixT& b,
                                 BaseMatrixT& c,
                                 BaseMatrixT& d) {
  applyQuaternary(quaternary::RankLoss<real>(), b, c, d);
}

DEFINE_MATRIX_QUATERNARY_OP(RankLossBp, const T THRESHOLD = 40.0; a = b - c;
                            a = (a > THRESHOLD)
                                    ? THRESHOLD
                                    : ((a < -THRESHOLD) ? (-THRESHOLD) : a);
                            a = exp(a); a = (a / (1 + a) - d));
template<>
void BaseMatrixT<real>::rankLossBp(BaseMatrixT& b,
                                   BaseMatrixT& c,
                                   BaseMatrixT& d) {
  applyQuaternary(quaternary::RankLossBp<real>(), b, c, d);
}

/* this = log(1 + exp(b)) - c * b */
DEFINE_MATRIX_TERNARY_OP(LogisticRegressionLoss, const T THRESHOLD = 40.0;
                         T x = (b > THRESHOLD) ? THRESHOLD : (b < -THRESHOLD)
                                                                 ? -THRESHOLD
                                                                 : b;
                         a = log(1 + exp(x)) - c * x);
template<>
void BaseMatrixT<real>::logisticRegressionLoss(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::LogisticRegressionLoss<real>(), b, c);
}

/* this = exp(b)/(1+exp(b)) - c */
DEFINE_MATRIX_TERNARY_OP(LogisticRegressionLossBp, const T THRESHOLD = 40.0;
                         T x = (b > THRESHOLD) ? THRESHOLD : (b < -THRESHOLD)
                                                                 ? -THRESHOLD
                                                                 : b;
                         x = exp(x); a = x / (1 + x) - c);
template<>
void BaseMatrixT<real>::logisticRegressionLossBp(BaseMatrixT& b,
                                                 BaseMatrixT& c) {
  applyTernary(ternary::LogisticRegressionLossBp<real>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(BiggerThan, a = (b > c) ? 1.0f : 0.0f);
template<class T>
void BaseMatrixT<T>::biggerThan(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::BiggerThan<T>(), b, c);
}

DEFINE_MATRIX_QUATERNARY_OP(
    BiggerThan, a = ((b > c && d > 0.5f) || (b < c && d < 0.5f)) ? 1.0f : 0.0f);
template<class T>
void BaseMatrixT<T>::biggerThan(BaseMatrixT& b,
                                BaseMatrixT& c,
                                BaseMatrixT& d) {
  applyQuaternary(quaternary::BiggerThan<T>(), b, c, d);
}

DEFINE_MATRIX_TERNARY_OP(Max, a = (b > c) ? b : c);
template<class T>
void BaseMatrixT<T>::max2(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::Max<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(BinaryClassificationError, ONE_PARAMETER,
                                   c += ((a > p) == (b > p)) ? 0.0f : 1.0f);
template<class T>
void BaseMatrixT<T>::binaryClassificationError2(size_t destCol, BaseMatrixT& b,
                                                BaseMatrixT& c, T p) {
  CHECK(!useGpu_) << "do not support gpu";
  MatrixOffset offset(0, 0, 0, 0, destCol, 0);
  int numRows = b.height_;
  int numCols = b.width_;
  b.applyTernary(ternary::BinaryClassificationError<T>(p), c, *this, numRows,
                 numCols, offset, false_type(), true_type() /*cAsColVector*/);
}

template<>
void BaseMatrixT<real>::binaryClassificationError(size_t destCol,
                                                  BaseMatrixT& b,
                                                  BaseMatrixT& c,
                                                  real p) {
  MatrixOffset offset(destCol, 0, 0, 0, 0, 0);
  int numRows = b.height_;
  int numCols = b.width_;
  aggregate(aggregate::sum(), base::binary::classificationError(p),
            base::binary::add(), b, c, numRows, numCols, offset, false_type(),
            true_type() /*aAsColVector*/);
}

DEFINE_MATRIX_QUATERNARY_PARAMETER_OP(Add3, THREE_PARAMETER,
                                      a = p1 * b + p2 * c + p3 * d);
template<class T>
void BaseMatrixT<T>::add3(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d, T p1,
                          T p2, T p3) {
  applyQuaternary(quaternary::Add3<T>(p1, p2, p3), b, c, d);
}

DEFINE_MATRIX_TERNARY_OP(DotMulSquare, a = b * c * c);
template<class T>
void BaseMatrixT<T>::dotMulSquare(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::DotMulSquare<T>(), b, c);
}

DEFINE_MATRIX_TERNARY_OP(DotSquareSquare, a = b * b * c * c);
template<class T>
void BaseMatrixT<T>::dotSquareSquare(BaseMatrixT& b, BaseMatrixT& c) {
  applyTernary(ternary::DotSquareSquare<T>(), b, c);
}

DEFINE_MATRIX_BINARY_OP(DotMulSquare, a *= b * b);
template<class T>
void BaseMatrixT<T>::dotMulSquare(BaseMatrixT& b) {
  applyBinary(binary::DotMulSquare<T>(), b);
}

DEFINE_MATRIX_BINARY_OP(DotSquareMul, a = a * a * b);
template<class T>
void BaseMatrixT<T>::dotSquareMul(BaseMatrixT& b) {
  applyBinary(binary::DotSquareMul<T>(), b);
}

DEFINE_MATRIX_QUATERNARY_PARAMETER_OP(AddSquareSum, THREE_PARAMETER,
                                      T tmp = p1 * b + p2 * c + p3 * d;
                                      a += tmp * tmp);
template<class T>
void BaseMatrixT<T>::addSquareSum(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT d,
                                  T p1, T p2, T p3) {
  applyQuaternary(quaternary::AddSquareSum<T>(p1, p2, p3), b, c, d);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(AddSquare, ONE_PARAMETER, a += p * b * b);
template<class T>
void BaseMatrixT<T>::addSquare(BaseMatrixT& b, T p) {
  applyBinary(binary::AddSquare<T>(p), b);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(DecayAddSquare, TWO_PARAMETER,
                                  a = p1 * a + p2 * b * b);
template<class T>
void BaseMatrixT<T>::decayAddSquare(BaseMatrixT& b, T p1, T p2) {
  applyBinary(binary::DecayAddSquare<T>(p1, p2), b);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(DecayAddSquareMul, TWO_PARAMETER,
                                   a = p1 * a + p2 * b * b * c * c);
template<class T>
void BaseMatrixT<T>::decayAddSquareMul(BaseMatrixT& b, BaseMatrixT& c, T p1,
                                       T p2) {
  applyTernary(ternary::DecayAddSquareMul<T>(p1, p2), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(ReciprocalSum, THREE_PARAMETER,
                                   a = 1 / (p1 * b + p2 * c + p3));
template<class T>
void BaseMatrixT<T>::reciprocalSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2,
                                   T p3) {
  applyTernary(ternary::ReciprocalSum<T>(p1, p2, p3), b, c);
}

DEFINE_MATRIX_BINARY_PARAMETER_OP(Reciprocal2, TWO_PARAMETER,
                                  a = 1 / (p1 * b + p2));
template<class T>
void BaseMatrixT<T>::reciprocal2(BaseMatrixT& b, T p1, T p2) {
  applyBinary(binary::Reciprocal2<T>(p1, p2), b);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(DotMulSquareSum, TWO_PARAMETER,
                                   T tmp = p1 * b + p2 * c;
                                   a *= tmp * tmp);
template<class T>
void BaseMatrixT<T>::dotMulSquareSum(BaseMatrixT& b, BaseMatrixT& c, T p1,
                                     T p2) {
  applyTernary(ternary::DotMulSquareSum<T>(p1, p2), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(DotSquareSum, TWO_PARAMETER,
                                   T tmp = p1 * b + p2 * c;
                                   a = tmp * tmp);
template<class T>
void BaseMatrixT<T>::dotSquareSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2) {
  applyTernary(ternary::DotSquareSum<T>(p1, p2), b, c);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(DotMulSum, TWO_PARAMETER,
                                   a *= p1 * b + p2 * c);
template<class T>
void BaseMatrixT<T>::dotMulSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2) {
  applyTernary(ternary::DotMulSum<T>(p1, p2), b, c);
}

DEFINE_MATRIX_BINARY_OP(CopyAndClear, b = a; a = 0);
template<class T>
void BaseMatrixT<T>::copyAndClear(BaseMatrixT& b) {
  applyBinary(binary::CopyAndClear<T>(), b);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(AddDotMul, TWO_PARAMETER,
                                   a = p1 * a + p2 * b * c);
template<class T>
void BaseMatrixT<T>::addDotMul(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2) {
  applyTernary(ternary::AddDotMul<T>(p1, p2), b, c);
}

DEFINE_MATRIX_BINARY_OP(Assign, a = b;);
template<class T>
void BaseMatrixT<T>::assign(BaseMatrixT& b) {
  if (useGpu_) {
    applyBinary(binary::Assign<T>(), b);
  } else {  // cpu version
    CHECK_EQ(this->height_, b.height_);
    CHECK_EQ(this->width_, b.width_);
    memcpy(data_, b.data_, sizeof(T) * height_ * width_);
  }
}

template<class T>
void BaseMatrixT<T>::assignAtOffset(BaseMatrixT& b, int64_t columnOffset) {
  if (columnOffset + b.width_ <= width_) {
    int numRows = height_;
    int numCols = b.width_;
    MatrixOffset offset(columnOffset, 0, 0, 0);
    applyBinary(binary::Assign<T>(), b, numRows, numCols, offset);
  } else if (columnOffset + width_ <= b.width_) {
    int numRows = height_;
    int numCols = width_;
    MatrixOffset offset(0, 0, columnOffset, 0);
    applyBinary(binary::Assign<T>(), b, numRows, numCols, offset);
  } else {
    LOG(FATAL) << "Wrong argument "
               << " a.width=" << width_ << " b.width=" << b.width_
               << " columnOffset=" << columnOffset;
  }
}

DEFINE_MATRIX_BINARY_OP(DeepSwap, T tmp = a; a = b; b = tmp);
template<class T>
void BaseMatrixT<T>::deepSwap(BaseMatrixT& b) {
    applyBinary(binary::DeepSwap<T>(), b);
}

template<>
void BaseMatrixT<real>::rowDotMul(size_t destCol,
                                  BaseMatrixT& b,
                                  BaseMatrixT& c) {
  int numRows = b.height_;
  int numCols = b.width_;
  MatrixOffset offset(destCol, 0, 0, 0, 0, 0);
  aggregate(aggregate::sum(), base::binary::mul(), base::binary::add(), b, c,
            numRows, numCols, offset, false_type(),
            true_type() /*aAsColVector*/);
}

template<class T>
void BaseMatrixT<T>::rowDotMul2(size_t destCol,
                                BaseMatrixT& b,
                                BaseMatrixT& c) {
  CHECK(!useGpu_) << "do not support gpu";

  size_t height = this->height_;
  CHECK_LT(destCol, this->width_);
  CHECK_EQ(height, b.height_);
  CHECK_EQ(height, c.height_);
  CHECK_EQ(b.width_, c.width_);
  size_t width = b.width_;
  T* A = this->data_;
  const T* B = b.data_;
  const T* C = c.data_;
  for (size_t i = 0; i < height;
       ++i, A += this->width_, B += width, C += width) {
    for (size_t j = 0; j < width; ++j) {
      A[destCol] += B[j] * C[j];
    }
  }
}

template<>
void BaseMatrixT<real>::addDotMulVMM(BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  int numRows = b.height_;
  int numCols = b.width_;
  aggregate(aggregate::sum(), base::binary::mul(), base::binary::add(), b, c,
            numRows, numCols, offset, true_type() /*aAsRowVector*/,
            false_type());
}

template<class T>
void BaseMatrixT<T>::addDotMulVMM2(BaseMatrixT& b, BaseMatrixT& c) {
  CHECK(!useGpu_) << "do not support gpu";

  CHECK_EQ(height_, 1LU);
  CHECK_EQ(b.height_, c.height_);
  CHECK_EQ(width_, b.width_);
  CHECK_EQ(width_, c.width_);
  size_t height = b.height_;
  size_t width = b.width_;
  T* A = this->data_;
  const T* B = b.data_;
  const T* C = c.data_;
  for (size_t i = 0; i < height; ++i, B += width, C += width) {
    for (size_t j = 0; j < width; ++j) {
      A[j] += B[j] * C[j];
    }
  }
}

DEFINE_MATRIX_TERNARY_OP(addDotMulMMV, a += b * c);
template<class T>
void BaseMatrixT<T>::addDotMulMMV(BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::addDotMulMMV<T>(), b, c, numRows, numCols, offset,
               true_type() /*cAsRowVector*/, false_type());
}

template<class T>
void BaseMatrixT<T>::addDotMulMMV2(BaseMatrixT& b, BaseMatrixT& c) {
  CHECK(!useGpu_) << "do not support gpu";

  CHECK_EQ(c.height_, 1LU);
  CHECK_EQ(height_, b.height_);
  CHECK_EQ(width_, b.width_);
  CHECK_EQ(width_, c.width_);
  size_t height = height_;
  size_t width = width_;
  T* A = this->data_;
  const T* B = b.data_;
  const T* C = c.data_;
  for (size_t i = 0; i < height; ++i, A += width, B += width) {
    for (size_t j = 0; j < width; ++j) {
      A[j] += B[j] * C[j];
    }
  }
}

template<class T>
void BaseMatrixT<T>::rowScale(size_t cCol, BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, cCol, 0);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::DotMul<T>(), b, c, numRows, numCols, offset,
    false_type(), true_type() /*cAsColVector*/);
}

template<class T>
void BaseMatrixT<T>::rowScale2(size_t cCol, BaseMatrixT& b, BaseMatrixT& c) {
  CHECK(!useGpu_) << "do not support gpu";

  size_t height = this->height_;
  size_t width = this->width_;
  CHECK_EQ(height, b.height_);
  CHECK_EQ(width, b.width_);
  CHECK_LT(cCol, c.width_);
  CHECK_EQ(height, c.height_);
  T* A = this->data_;
  const T* B = b.data_;
  const T* C = c.data_;
  for (size_t i = 0; i < height; ++i, A += width, B += width, C += c.width_) {
    for (size_t j = 0; j < width; ++j) {
      A[j] = B[j] * C[cCol];
    }
  }
}

template<class T>
void BaseMatrixT<T>::colScale(size_t cRow, BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, 0, cRow);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::DotMul<T>(), b, c, numRows, numCols, offset,
               true_type() /* cAsRowVector */, false_type() /* cAsColVector */);
}

template<class T>
void BaseMatrixT<T>::addColScale(size_t cRow, BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, 0, cRow);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::addDotMulMMV<T>(), b, c, numRows, numCols, offset,
               true_type() /* cAsRowVector */, false_type() /* cAsColVector */);
}

template<class T>
void BaseMatrixT<T>::addRowScale(size_t cCol, BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, cCol, 0);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::addDotMulMMV<T>(), b, c, numRows, numCols, offset,
               false_type(), true_type() /*cAsColVector*/);
}

DEFINE_MATRIX_TERNARY_PARAMETER_OP(RowAdd, ONE_PARAMETER, a = b + p * c);
template<class T>
void BaseMatrixT<T>::rowAdd(size_t cCol, BaseMatrixT& b, BaseMatrixT& c, T p) {
  MatrixOffset offset(0, 0, 0, 0, cCol, 0);
  int numRows = height_;
  int numCols = width_;
  applyTernary(ternary::RowAdd<T>(p), b, c, numRows, numCols, offset,
    false_type(), true_type() /*cAsColVector*/);
}

DEFINE_MATRIX_TERNARY_OP(RowPow, a = pow(b, c));
template<>
void BaseMatrixT<real>::rowPow(size_t cCol, BaseMatrixT& b, BaseMatrixT& c) {
  if (useGpu_) {
    MatrixOffset offset(0, 0, 0, 0, cCol, 0);
    int numRows = height_;
    int numCols = width_;
    applyTernary(ternary::RowPow<real>(), b, c, numRows, numCols, offset,
                 false_type(), true_type() /*cAsColVector*/);
  } else {
    size_t height = this->height_;
    size_t width = this->width_;
    CHECK_EQ(height, b.height_);
    CHECK_EQ(width, b.width_);
    CHECK_LT(cCol, c.width_);
    CHECK_EQ(height, c.height_);
    real* A = this->data_;
    const real* B = b.data_;
    const real* C = c.data_;
    for (size_t i = 0; i < height; ++i, A += width, B += width, C += c.width_) {
      vPow(width, B, C[cCol], A);
    }
  }
}

template<class T>
void BaseMatrixT<T>::mulRowVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::DotMul<T>(), b, numRows, numCols, offset,
              true_type() /* bAsRowVector */, false_type());
}

DEFINE_MATRIX_BINARY_OP(DotDiv, a /= b);
template<class T>
void BaseMatrixT<T>::divRowVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::DotDiv<T>(), b, numRows, numCols, offset,
              true_type() /* bAsRowVector */, false_type());
}

template<class T>
void BaseMatrixT<T>::mulColVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::DotMul<T>(), b, numRows, numCols, offset,
              false_type(), true_type() /* bAsColVector */);
}

template<class T>
void BaseMatrixT<T>::divColVector(BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0);
  int numRows = height_;
  int numCols = width_;
  applyBinary(binary::DotDiv<T>(), b, numRows, numCols, offset,
              false_type(), true_type() /* bAsColVector */);
}

template<>
template <class Agg>
int BaseMatrixT<real>::applyRow(Agg agg, BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  size_t numRows = b.height_;
  size_t numCols = b.width_;
  CHECK_EQ(height_, numRows);
  CHECK_EQ(width_, 1UL);
  aggregate(agg, base::unary::identity(), base::binary::second(), b, numRows,
            numCols, offset, false_type(), true_type() /*aAsColVector*/);

  return 0;
}

template<>
template <class Agg, class Saver>
int BaseMatrixT<real>::applyRow(Agg agg, Saver sv, BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  size_t numRows = b.height_;
  size_t numCols = b.width_;
  CHECK_EQ(height_, numRows);
  CHECK_EQ(width_, 1UL);
  aggregate(agg, base::unary::identity(), sv, b, numRows, numCols, offset,
            false_type(), true_type() /*aAsColVector*/);

  return 0;
}

template<>
template <class Agg>
int BaseMatrixT<real>::applyRow(
     Agg agg, real scaleDest, real scaleAgg, BaseMatrixT& b) {
  if (scaleDest != 0) {
    applyRow(agg, base::binary::add2(scaleDest, scaleAgg), b);
  } else {
    applyRow(agg, base::binary::second(), b);
    if (scaleAgg != 1) {
      mulScalar(scaleAgg);
    }
  }
  return 0;
}

template<>
template <class Agg, class Op, class Saver>
int BaseMatrixT<real>::applyRow(Agg agg, Op op, Saver sv,
                                BaseMatrixT& b, BaseMatrixT& c) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  size_t numRows = b.height_;
  size_t numCols = b.width_;
  CHECK_EQ(height_, numRows);
  CHECK_EQ(width_, 1UL);
  CHECK_EQ(c.height_, numRows);
  CHECK_EQ(c.width_, numCols);
  aggregate(agg, op, sv,
            b, c, numRows, numCols, offset,
            false_type(), true_type() /*aAsColVector*/);
  return 0;
}

template<>
template <class Agg, class Op>
int BaseMatrixT<real>::applyRow(Agg agg, Op op, real scaleDest, real scaleAgg,
                                BaseMatrixT& b, BaseMatrixT& c) {
  if (scaleDest != 0) {
    applyRow(agg, op, base::binary::add2(scaleDest, scaleAgg), b, c);
  } else {
    applyRow(agg, op, base::binary::second(), b, c);
    if (scaleAgg != 1) {
      mulScalar(scaleAgg);
    }
  }
  return 0;
}

template<>
template <class Agg>
int BaseMatrixT<real>::applyCol(Agg agg, BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  size_t numRows = b.height_;
  size_t numCols = b.width_;
  CHECK_EQ(width_, numCols);
  CHECK_EQ(height_, 1UL);
  aggregate(agg, base::unary::identity(), base::binary::second(), b, numRows,
            numCols, offset, true_type() /*aAsRowVector*/, false_type());

  return 0;
}

template<>
template <class Agg, class Saver>
int BaseMatrixT<real>::applyCol(Agg agg, Saver sv, BaseMatrixT& b) {
  MatrixOffset offset(0, 0, 0, 0, 0, 0);
  size_t numRows = b.height_;
  size_t numCols = b.width_;
  CHECK_EQ(width_, numCols);
  CHECK_EQ(height_, 1UL);
  aggregate(agg, base::unary::identity(), sv, b, numRows, numCols, offset,
            true_type() /*aAsRowVector*/, false_type());

  return 0;
}

template<>
template <class Agg>
int BaseMatrixT<real>::applyCol(
     Agg agg, real scaleDest, real scaleAgg, BaseMatrixT& b) {
  if (scaleDest != 0) {
    applyCol(agg, base::binary::add2(scaleDest, scaleAgg), b);
  } else {
    applyCol(agg, base::binary::second(), b);
    if (scaleAgg != 1) {
      mulScalar(scaleAgg);
    }
  }
  return 0;
}

template<>
void BaseMatrixT<real>::sumRows(BaseMatrixT& b, real scaleSum, real scaleDest) {
  applyRow(aggregate::sum(), scaleDest, scaleSum, b);
}

template<>
void BaseMatrixT<real>::maxRows(BaseMatrixT& b) {
  applyRow(aggregate::max(), b);
}

template<>
void BaseMatrixT<real>::minRows(BaseMatrixT& b) {
  applyRow(aggregate::min(), b);
}

template<>
void BaseMatrixT<real>::maxCols(BaseMatrixT& b) {
  applyCol(aggregate::max(), b);
}

template<>
void BaseMatrixT<real>::minCols(BaseMatrixT& b) {
  applyCol(aggregate::min(), b);
}

template<>
void BaseMatrixT<real>::sumCols(BaseMatrixT& b, real scaleSum, real scaleDest) {
  applyCol(aggregate::sum(), scaleDest, scaleSum, b);
}

template<>
void BaseMatrixT<real>::sumOfSquaredDiffs(
    BaseMatrixT& b, BaseMatrixT& c, real scaleSum, real scaleDest) {
  applyRow(aggregate::sum(), base::binary::squaredDiff(),
           scaleDest, scaleSum, b, c);
}

template<>
void BaseMatrixT<real>::sumOfProducts(
    BaseMatrixT& b, BaseMatrixT& c, real scaleSum, real scaleDest) {
  applyRow(aggregate::sum(), base::binary::mul(),
           scaleDest, scaleSum, b, c);
}

template class BaseMatrixT<real>;
template class BaseMatrixT<int>;
}  // namespace paddle
