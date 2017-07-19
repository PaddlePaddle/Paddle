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
#include <stdint.h>
#include <cstddef>
#include "TensorExpression.h"
#include "paddle/utils/Common.h"

namespace paddle {

/*
 * nvcc currently does not support C++11,
 * so I realized false_type and true_type.
 */
template <class T, T v>
struct bool_constant {
  static const T value = v;
};
typedef bool_constant<bool, false> false_type;
typedef bool_constant<bool, true> true_type;

/**
 * @brief   Calculate matrix element address.
 *
 * For instance, address of A[i][j] = i * ld + j.
 *
 */
#define CAL_MATRIX_START_ADDRESS(address, height, width, ld, col, row) \
  CHECK_LE(col, width);                                                \
  CHECK_LE(row, height);                                               \
  address += row * ld + col;

class MatrixOffset {
public:
  size_t aCol_;
  size_t aRow_;
  size_t bCol_;
  size_t bRow_;
  size_t cCol_;
  size_t cRow_;
  size_t dCol_;
  size_t dRow_;
  MatrixOffset(size_t aCol = 0,
               size_t aRow = 0,
               size_t bCol = 0,
               size_t bRow = 0,
               size_t cCol = 0,
               size_t cRow = 0,
               size_t dCol = 0,
               size_t dRow = 0)
      : aCol_(aCol),
        aRow_(aRow),
        bCol_(bCol),
        bRow_(bRow),
        cCol_(cCol),
        cRow_(cRow),
        dCol_(dCol),
        dRow_(dRow) {}
};

template <class T>
class BaseMatrixT : public TensorExpression<BaseMatrixT<T>, T> {
public:
  size_t height_, width_;
  size_t stride_;
  T* data_;
  bool trans_;
  bool useGpu_;

public:
  virtual ~BaseMatrixT() {}
  BaseMatrixT(size_t height, size_t width, T* data, bool trans, bool useGpu)
      : height_(height),
        width_(width),
        stride_(width),
        data_(data),
        trans_(trans),
        useGpu_(useGpu) {}

  /**
   * @note This constructor is for temporarily making a matrix with different
   *       useGpu flag as the original matrix so that mixed gpu/cpu operations
   *       can be performed successfully.
   */
  BaseMatrixT(BaseMatrixT& mat, bool useGpu)
      : height_(mat.height_),
        width_(mat.width_),
        stride_(mat.stride_),
        data_(mat.data_),
        trans_(mat.trans_),
        useGpu_(useGpu) {}

  BaseMatrixT(size_t height,
              size_t width,
              size_t stride,
              T* data,
              bool trans,
              bool use_gpu)
      : height_(height),
        width_(width),
        stride_(stride),
        data_(data),
        trans_(trans),
        useGpu_(use_gpu) {
    /* CHECK_LE(width_, stride_); */
  }

  /// caller should make sure that the size of data is at least height*width
  void setData(T* data) { data_ = data; }

  /**
   * unary operator: element wise op(a).
   *
   * @code
   * for 0 <= i < this->height_ & for 0 <= j < this->width_.
   * @endcode
   */
  template <class Op>
  int applyUnary(Op op);

  /**
   * unary operator: element wise op(a).
   *
   * @code
   * for 0 <= i < numRows & for 0 <= j < numCols.
   * While matrix start address is:
   *  A = this->data_ + offset.aRow_*ld + offset.aCol_;
   * @endcode
   */
  template <class Op>
  int applyUnary(Op op, int numRows, int numCols, MatrixOffset& offset);

  /**
   * binary operator: element wise op(a, b).
   *
   * @code
   * for 0 <= i < this->height_ & for 0 <= j < this->width_.
   * While this->height_ == b.height_ && this->width_ == b.width_.
   * @endcode
   */
  template <class Op>
  int applyBinary(Op op, BaseMatrixT& b);

  /**
   * binary operator: element wise op(a, b)
   *
   * @code
   * for 0 <= i < numRows & for 0 <= j < numCols.
   * While matrix start address is:
   *   A = this->data_ + offset.aRow_*lda + offset.aCol_;
   *   B = b->data_ + offset.bRow_*ldb + offset.bCol_;
   *
   * if (bAsRowVector == false_type && bAsColVector == false_type)
   *   op(A[i * lda + j], B[i * ldb + j])
   *
   * if (bAsRowVector == true_type && bAsColVector == false_type)
   *   op(A[i * lda + j], B[j])
   *
   * if (bAsRowVector == false_type && bAsColVector == true_type)
   *   op(A[i * lda + j], B[i * ldb])
   *
   * if (bAsRowVector == true_type && bAsColVector == true_type)
   *   op(A[i * lda + j], B[0])
   * @endcode
   */
  template <class Op, class bAsRowVector, class bAsColVector>
  int applyBinary(Op op,
                  BaseMatrixT& b,
                  int numRows,
                  int numCols,
                  MatrixOffset& offset,
                  bAsRowVector,
                  bAsColVector);

  template <class Op>
  int applyBinary(
      Op op, BaseMatrixT& b, int numRows, int numCols, MatrixOffset& offset);

  /**
   * ternary operator: element wise op(a, b, c).
   *
   * @code
   * for 0 <= i < this->height_ & for 0 <= j < this->width_.
   *
   * While this->height_ == b.height_ && this->width_ == b.width_
   *    && this->height_ == c.height_ && this->width_ == c.width_
   * @endcode
   */
  template <class Op>
  int applyTernary(Op op, BaseMatrixT& b, BaseMatrixT& c);

  /**
   * ternary operator: element wise op(a, b, c).
   *
   * @code
   *  for 0 <= i < numRows & for 0 <= j < numCols.
   *  While matrix start address is:
   *
   *    A = this->data_ + offset.aRow_*lda + offset.aCol_;
   *    B = b->data_ + offset.bRow_*ldb + offset.bCol_;
   *    C = c->data_ + offset.cRow_*ldc + offset.cCol_;
   *
   *    if (cAsRowVector == false_type && cAsColVector == false_type)
   *      op(A[i*lda + j], B[i*ldb + j], C[i*ldc + j])
   *
   *    if (cAsRowVector == true_type && cAsColVector == false_type)
   *      op(A[i*lda + j], B[i*ldb + j], C[j])
   *
   *    if (cAsRowVector == false_type && cAsColVector == true_type)
   *      op(A[i*lda + j], B[i*ldb + j], C[i*ldc])
   *
   *    if (cAsRowVector == 1 && cAsColVector == 1)
   *      op(A[i*lda + j], B[i*ldb + j], C[0])
   * @endcode
   */
  template <class Op, class cAsRowVector, class cAsColVector>
  int applyTernary(Op op,
                   BaseMatrixT& b,
                   BaseMatrixT& c,
                   int numRows,
                   int numCols,
                   MatrixOffset& offset,
                   cAsRowVector,
                   cAsColVector);

  template <class Op>
  int applyTernary(Op op,
                   BaseMatrixT& b,
                   BaseMatrixT& c,
                   int numRows,
                   int numCols,
                   MatrixOffset& offset);

  /**
   * quaternary operator: element wise op(a, b, c, d).
   *
   * @code
   * for 0 <= i < this->height_ & for 0 <= j < this->width_.
   *
   * While this->height_ == b.height_ && this->width_ == b.width_
   *    && this->height_ == c.height_ && this->width_ == c.width_
   *    && this->height_ == d.height_ && this->width_ == d.width_
   * @endcode
   */
  template <class Op>
  int applyQuaternary(Op op, BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d);

  /**
   * quaternary operator: element wise op(a, b, c, d).
   *
   * @code
   * for 0 <= i < numRows & for 0 <= j < numCols.
   * While matrix start address is:
   *    A = this->data_ + offset.aRow_*lda + offset.aCol_;
   *    B = b->data_ + offset.bRow_*ldb + offset.bCol_;
   *    C = c->data_ + offset.cRow_*ldc + offset.cCol_;
   *    D = d->data_ + offset.dRow_*ldd + offset.dCol_;
   * @endcode
   */
  template <class Op>
  int applyQuaternary(Op op,
                      BaseMatrixT& b,
                      BaseMatrixT& c,
                      BaseMatrixT& d,
                      int numRows,
                      int numCols,
                      MatrixOffset& offset);

  /**
   * a aggregate expression that apply each row(or column) of matrix b.
   * op and sv is element wise operator.
   *
   * @code
   * if (aAsRowVector == true_type && aAsColVector == false_type)
   *  for each column j & 0 <= i < numRows, do:
   *    dst = agg(op(b[i*ldb + j]))
   *    a[j] = sv(a[j], dst)
   *
   * if (aAsRowVector == false_type && aAsColVector == true_type)
   *  for each row i & 0 <= j < numCols, do:
   *    dst = agg(op(b[i*ldb + j]))
   *    a[i] = sv(a[i], dst)
   * @endcode
   */
  template <class Agg,
            class Op,
            class Saver,
            class aAsRowVector,
            class aAsColVector>
  int aggregate(Agg agg,
                Op op,
                Saver sv,
                BaseMatrixT& b,
                int numRows,
                int numCols,
                MatrixOffset& offset,
                aAsRowVector,
                aAsColVector);

  /**
   * a aggregate expression that apply each row(or column) of matrix b and c.
   *
   * op and sv is element wise operator.
   *
   * @code
   * if (aAsRowVector == true_type && aAsColVector == false_type)
   *   for each column j & 0 <= i < numRows, do:
   *     dst = agg(op(b[i*ldb + j], c[i*ldc + j]))
   *     a[j] = sv(a[j], dst)
   *
   * if (aAsRowVector == false_type && aAsColVector == true_type)
   *   for each row i & 0 <= j < numCols, do:
   *     dst = agg(op(b[i*ldb + j], c[i*ldc + j]))
   *     a[i] = sv(a[i], dst)
   * @endcode
   */
  template <class Agg,
            class Op,
            class Saver,
            class aAsRowVector,
            class aAsColVector>
  int aggregate(Agg agg,
                Op op,
                Saver sv,
                BaseMatrixT& b,
                BaseMatrixT& c,
                int numRows,
                int numCols,
                MatrixOffset& offset,
                aAsRowVector,
                aAsColVector);

  /**
   * a aggregate expression that apply each row of matrix b.
   *
   * @code
   * for each row i & 0 <= j < b.width_, do:
   *   this[i] = agg(b[i*ldb + j])
   * @endcode
   */
  template <class Agg>
  int applyRow(Agg agg, BaseMatrixT& b);

  /**
   * a aggregate expression that apply each row of matrix b.
   *
   * @code
   * for each row i & 0 <= j < b.width_, do:
   *   dst = agg(op(b[i*ldb + j], c[i*ldc + j])
   *   this[i] = sv(this[i], dst)
   * @endcode
   */
  template <class Agg, class Op, class Saver>
  int applyRow(Agg agg, Op op, Saver sv, BaseMatrixT& b, BaseMatrixT& c);

  // Same as the above with the special handing of sv=add2(scaleDest, scaleAgg)
  template <class Agg, class Op>
  int applyRow(Agg agg,
               Op op,
               real scaleDest,
               real scaleAgg,
               BaseMatrixT& b,
               BaseMatrixT& c);

  /**
   * a aggregate expression that apply each row of matrix b.
   *
   * @code
   * for each row i & 0 <= j < b.width_, do:
   *   dst = agg(b[i*ldb + j])
   *   this[i] = sv(this[i], dst)
   * @endcode
   */
  template <class Agg, class Saver>
  int applyRow(Agg agg, Saver sv, BaseMatrixT& b);

  // Same as the above with the special handing of sv=add2(scaleDest, scaleAgg)
  template <class Agg>
  int applyRow(Agg agg, real scaleDest, real scaleAgg, BaseMatrixT& b);

  /**
   * a aggregate expression that apply each column of matrix b.
   *
   * @code
   * for each column j & 0 <= i < b.height_, do:
   *   this[j] = agg(b[i*ldb + j])
   * @endcode
   */
  template <class Agg>
  int applyCol(Agg agg, BaseMatrixT& b);

  /**
   * a aggregate expression that apply each column of matrix b.
   *
   * @code
   * for each column j & 0 <= i < b.height_, do:
   *   dst = agg(b[i*ldb + j])
   *   this[j] = sv(this[j], dst)
   * @endcode
   */
  template <class Agg, class Saver>
  int applyCol(Agg agg, Saver sv, BaseMatrixT& b);

  // Same as the above with the special handing of sv=add2(scaleDest, scaleAgg)
  template <class Agg>
  int applyCol(Agg agg, real scaleDest, real scaleAgg, BaseMatrixT& b);

  bool useGpu() const { return useGpu_; }

  const T* rowBuf(size_t row) const { return data_ + width_ * row; }

  T* rowBuf(size_t row) { return data_ + width_ * row; }

  /**
   * @brief   unary operator.
   *
   */
  void neg();
  void exp2();
  void pow2(T p);
  void log2();
  void sqrt2();
  void square2();
  void reciprocal2();
  void abs2();
  void sign2();
  void zero();

  /**
   * @code
   * this(row, col + columnOffset) = 0 for 0 <= col < numColumns
   * @endcode
   */
  void zeroAtOffset(int64_t columnOffset, int64_t numColumns);
  void one();
  void subScalar(T p);
  void mulScalar(T p);
  void divScalar(T p);

  /**
   * @code
   * this = p
   * @endcode
   */
  void assign(T p);

  /**
   * @code
   * swap(this, b)
   * example: swap two Matrices
   * MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
   * MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
   * cpuA->deepSwap(*cpuB);
   * @endcode
   */
  void deepSwap(BaseMatrixT& b);

  /**
   * @code
   * this = this + p
   * @endcode
   */
  void add(T p);

  /**
   * @code
   * this = this*p1 + p2
   * @endcode
   */
  void add(T p1, T p2);

  /**
   * this = this < low ? low : this
   *
   * this = this > high ? high : this
   */
  void clip(T p1, T p2);

  /**
   * @code
   * a = a > p ? 1.0f : 0.0f
   * @endcode
   */
  void biggerThanScalar(T p);

  /**
   * @code
   * a = a > p ? a : p
   * @endcode
   */
  void downClip(T p);

  /**
   * @code
   * this = b
   * @endcode
   */
  void assign(BaseMatrixT& b);

  /**
   * @code
   * If b.width + columOffset <= this.width
   *  this(row, col + columnOffset) = b(row, col) for 0 <= col < b.width
   *
   * If this.width + columnOffset <= b.width
   *  this(row, col) = b(row, col + columnOffset) for 0 <= col < this.width
   *
   * Otherwise, FATAL
   * @endcode
   */
  void assignAtOffset(BaseMatrixT& b, int64_t columnOffset);

  /// this = this + b
  void add(BaseMatrixT& b);

  /**
   * @code
   * If b.width + columOffset <= this.width
   *  this(row, col + columnOffset) += b(row, col) for 0 <= col < b.width
   *
   * If this.width + columnOffset <= b.width
   *  this(row, col) += b(row, col + columnOffset) for 0 <= col < this.width
   *
   * Otherwise, FATAL
   * @endcode
   */
  void addAtOffset(BaseMatrixT& b, int64_t columnOffset);

  void addColVector(BaseMatrixT& b);
  void addRowVector(BaseMatrixT& b);
  void addBias(BaseMatrixT& b, T scale);

  void mulRowVector(BaseMatrixT& b);
  void divRowVector(BaseMatrixT& b);

  void mulColVector(BaseMatrixT& b);
  void divColVector(BaseMatrixT& b);

  void addP2P(BaseMatrixT& b);

  /**
   * @code
   * this = this + b*p
   * @endcode
   */
  void add(BaseMatrixT& b, T p);

  /**
   * @code
   * this = p1*this + p2*b
   * @endcode
   */
  void add(BaseMatrixT& b, T p1, T p2);

  /**
   * @code
   * this = this - b
   * @endcode
   */
  void sub(BaseMatrixT& b);

  /**
   * @code
   * this = this - b*p
   * @endcode
   */
  void sub(BaseMatrixT& b, T p);

  /**
   * @code
   * b = max(0, this)
   * @endcode
   */
  void relu(BaseMatrixT& b);
  void reluDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = log(1.0 + exp(this))
   * @endcode
   */
  void softrelu(BaseMatrixT& b);
  void softreluDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = min(max(this, p1), p2)
   * @endcode
   */
  void brelu(BaseMatrixT& b);
  void breluDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = this * this
   * @endcode
   */
  void square2(BaseMatrixT& b);
  void squareDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = tanh(this)
   * @endcode
   */
  void tanh(BaseMatrixT& b);
  void tanhDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = p1 * tanh(p2 * this)
   * @endcode
   */
  void scaledTanh(BaseMatrixT& b, T p1, T p2);
  void scaledTanhDerivative(BaseMatrixT& b, T p1, T p2);

  /**
   * @code
   * b = 1.0f / this
   * @endcode
   */
  void reciprocal2(BaseMatrixT& b);
  void reciprocalDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = this > 0.0f ? this : -this
   * @endcode
   */
  void abs2(BaseMatrixT& b);
  void absDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = 1.0f / (1.0f + exp(-this))
   * @endcode
   */
  void sigmoid(BaseMatrixT& b);
  void sigmoidDerivative(BaseMatrixT& b);

  /**
   * @code
   * b = a
   * @endcode
   */
  void expDerivative(BaseMatrixT& b);

  void sign2(BaseMatrixT& b);

  void exp2(BaseMatrixT& b);
  void pow2(BaseMatrixT& b, T p);
  void log2(BaseMatrixT& b);
  void sqrt2(BaseMatrixT& b);
  void addScalar(BaseMatrixT& b, T p);
  void subScalar(BaseMatrixT& b, T p);
  void mulScalar(BaseMatrixT& b, T p);
  void divScalar(BaseMatrixT& b, T p);
  void scalarDiv(BaseMatrixT& b, T p);

  /**
   * @code
   * this = 1.0f / sqrt(b)
   * @endcode
   */
  void invSqrt(BaseMatrixT& b);

  /// this = (b == value)
  void isEqualTo(BaseMatrixT& b, T value);

  /**
   * @brief   ternary operator.
   */
  void softCrossEntropy(BaseMatrixT& b, BaseMatrixT& c);
  void softCrossEntropyBp(BaseMatrixT& b, BaseMatrixT& c);
  void binaryLabelCrossEntropy(BaseMatrixT& b, BaseMatrixT& c);
  void binaryLabelCrossEntropyBp(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = b + c
   * @endcode
   */
  void add(BaseMatrixT& b, BaseMatrixT& c);
  /**
   * @code
   * this = b*p1 + c*p2
   * @endcode
   */
  void add(BaseMatrixT& b, T p1, BaseMatrixT& c, T p2);
  /**
   * @code
   * this = b - c
   * @endcode
   */
  void sub(BaseMatrixT& b, BaseMatrixT& c);
  /**
   * @code
   * this = b*p1 - c*p2
   * @endcode
   */
  void sub(BaseMatrixT& b, T p1, BaseMatrixT& c, T p2);

  /**
   * @code
   * this = this + b + c
   * @endcode
   */
  void add2(BaseMatrixT& b, BaseMatrixT& c);
  /**
   * @code
   * this = this*p1 + b*p2 + c*p3
   * @endcode
   */
  void add2(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2, T p3);

  /**
   * @code
   * this = a*p1 + b*p2 + c*p3
   * @endcode
   */
  void add3(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d, T p1, T p2, T p3);

  /**
   * @code
   *   c = p2 * c - p1 *  (b + p3 * this)
   *   this += mom
   * @endcode
   */
  void sgdUpdate(BaseMatrixT& b,  //  grad
                 BaseMatrixT& c,  //  mom
                 T p1,            //  learningRate,
                 T p2,            //  momentum,
                 T p3);           //  decayRate

  /**
   * @code
   *   c = p2 * c - p1 * d * (b + p3 * this)
   *   this += mom
   * @endcode
   */
  void sgdUpdate(BaseMatrixT& b,  // grad,
                 BaseMatrixT& c,  // mom,
                 BaseMatrixT& d,  // lr,
                 T p1,            // learningRate,
                 T p2,            // momentum,
                 T p3);           // decayRate

  /// apply L1/L2 to *this*
  virtual void applyL1(T learningRate, T decayRate);
  void applyL1(BaseMatrixT& lr, T learningRate, T decayRate);
  void applyL2(T learningRate, T decayRate);
  void applyL2(BaseMatrixT& lr, T learningRate, T decayRate);

  /**
   * @code
   * this *= b
   * @endcode
   */
  void dotMul(BaseMatrixT& b);

  /**
   * @code
   * this = b * c
   * @endcode
   */
  void dotMul(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = b / c
   * @endcode
   */
  void dotDiv(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = (b + p1) / (c + p2)
   * @endcode
   */
  void dotDiv(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this = log(1 + exp(b - c)) - d * (b - c)
   * @endcode
   */
  void rankLoss(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d);
  void rankLossBp(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d);

  /**
   * @code
   * this = log(1 + exp(b)) - c * b
   * @endcode
   */
  void logisticRegressionLoss(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this += exp(b)/(1+exp(b)) - c
   * @endcode
   */
  void logisticRegressionLossBp(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = b > c ? 1.0 : 0.0
   * @endcode
   */
  void biggerThan(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = ((b>c && d>0.5) || (b<c && d<0.5)) ? 1 : 0)
   * @endcode
   */
  void biggerThan(BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT& d);

  /**
   * @code
   * this = b>c ? b : c
   * @endcode
   */
  void max2(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this[destCol] += (b>p1 == c>p1) ? 0 : 1)
   * @endcode
   */
  void binaryClassificationError(size_t destCol,
                                 BaseMatrixT& b,
                                 BaseMatrixT& c,
                                 T p);
  void binaryClassificationError2(size_t destCol,
                                  BaseMatrixT& b,
                                  BaseMatrixT& c,
                                  T p);

  /**
   * @code
   * this = this * b * b
   * @endcode
   */
  void dotMulSquare(BaseMatrixT& b);

  /**
   * @code
   * this = this * this * b
   * @endcode
   */
  void dotSquareMul(BaseMatrixT& b);

  /**
   * @code
   * this = b * c * c
   * @endcode
   */
  void dotMulSquare(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = b * b * c * c
   * @endcode
   */
  void dotSquareSquare(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = this * (p1*b + p2*c)^2
   * @endcode
   */
  void dotMulSquareSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this = (p1*b + p2*c)^2
   * @endcode
   */
  void dotSquareSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this=  this * (p1*b + p2*c)
   * @endcode
   */
  void dotMulSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this += sqr(p1*b + p2*c + p3*d)
   * @endcode
   */
  void addSquareSum(
      BaseMatrixT& b, BaseMatrixT& c, BaseMatrixT d, T p1, T p2, T p3);

  /**
   * @code
   * this += p * sqr(b)
   * @endcode
   */
  void addSquare(BaseMatrixT& b, T p);

  /**
   * @code
   * this = p1 * this + p2 * sqr(b)
   * @endcode
   */
  void decayAddSquare(BaseMatrixT& b, T p1, T p2);

  /**
   * @code
   * this = p1 * this + p2 * sqr(b * c)
   * @endcode
   */
  void decayAddSquareMul(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this = 1 / (p1 * b + p2)
   * @endcode
   */
  void reciprocal2(BaseMatrixT& b, T p1, T p2);

  /**
   * @code
   * this = 1 / (p1 * b + p2 * c + p3)
   * @endcode
   */
  void reciprocalSum(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2, T p3);

  /**
   * @code
   * b = this; this = 0
   * @endcode
   */
  void copyAndClear(BaseMatrixT& b);

  /**
   * @code
   * this_row[destCol] += dotprod(b_row, c_row)
   * @endcode
   */
  void rowDotMul(size_t destCol, BaseMatrixT& b, BaseMatrixT& c);
  void rowDotMul2(size_t destCol, BaseMatrixT& b, BaseMatrixT& c);

  /**
   * this is vector (one row matrix)
   *
   * @code
   *   for each row i, do:
   *      this_row += dotmul(b_row_i, c_row_i)
   * @endcode
   */
  void addDotMulVMM(BaseMatrixT& b, BaseMatrixT& c);
  void addDotMulVMM2(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * c is vector (one row matrix)
   *
   * @code
   * for each row i, do:
   *    this_row_i += dotmul(b_row_i, c_row)
   * @endcode
   */
  void addDotMulMMV(BaseMatrixT& b, BaseMatrixT& c);
  void addDotMulMMV2(BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this = p1 * this + p2 * b * c
   * @endcode
   */
  void addDotMul(BaseMatrixT& b, BaseMatrixT& c, T p1, T p2);

  /**
   * @code
   * this_row = b_row * c_row[cCol]
   * @endcode
   */
  void rowScale(size_t cCol, BaseMatrixT& b, BaseMatrixT& c);
  void rowScale2(size_t cCol, BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this_col = b_col * c_col[cRow]
   * @endcode
   */
  void colScale(size_t cRow, BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this_col += b_col * c_col[cRow]
   * @endcode
   */
  void addColScale(size_t cRow, BaseMatrixT& b, BaseMatrixT& c);

  /**
   * @code
   * this_row += b_row * c_row[cCol]
   * @endcode
   */
  void addRowScale(size_t cCol, BaseMatrixT& b, BaseMatrixT& c);

  /// calculate the sum of each row of the matrix b.
  /// this_i = scaleDest * this_i + scaleSum * \sum_j b_{ij}
  void sumRows(BaseMatrixT& b, T scaleSum, T scaleDest);

  /// calculate the maximum value of each row of the matrix b.
  void maxRows(BaseMatrixT& b);
  /// calculate the minimum value of each row of the matrix b.
  void minRows(BaseMatrixT& b);

  /// calculate the maximum value of each column of the matrix b.
  void maxCols(BaseMatrixT& b);
  /// calculate the minimum value of each column of the matrix b.
  void minCols(BaseMatrixT& b);

  /// calculate the sum of each column of the matrix b.
  /// this_i = scaleDest * this_i + scaleSum * \sum_j b_{ji}
  void sumCols(BaseMatrixT& b, T scaleSum, T scaleDest);

  /// this_i = scaleDest * this_i + scaleSum * \sum_j (b_{ij} - c_{ij})^2
  void sumOfSquaredDiffs(BaseMatrixT& b,
                         BaseMatrixT& c,
                         T scaleSum,
                         T scaleDest);

  /// this_i = scaleDest * this_i + scaleSum * \sum_j b_{ij} * c_{ij}
  void sumOfProducts(BaseMatrixT& b, BaseMatrixT& c, T scaleSum, T scaleDest);

  /**
   * @code
   * this_row = b_row + p * ones * c_row[cCol]
   * @endcode
   */
  void rowAdd(size_t cCol, BaseMatrixT& b, BaseMatrixT& c, T p);
  /**
   * @code
   * this_row = pow(b_row, c_row[cCol])
   * @endcode
   */
  void rowPow(size_t cCol, BaseMatrixT& b, BaseMatrixT& c);

  virtual bool isSparse() const { return false; }

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    if (useGpu_) {
      TensorGpuApply<T>(*this, expr);
    } else {
      TensorCpuApply<T>(*this, expr);
    }
  }

  template <typename ExpressionType>
  void operator+=(const ExpressionType& expr) {
    (*this) = (*this) + expr;
  }
  template <typename ExpressionType>
  void operator-=(const ExpressionType& expr) {
    (*this) = (*this) - expr;
  }
  template <typename ExpressionType>
  void operator*=(const ExpressionType& expr) {
    (*this) = (*this) * expr;
  }
  template <typename ExpressionType>
  void operator/=(const ExpressionType& expr) {
    (*this) = (*this) / expr;
  }
};

typedef BaseMatrixT<real> BaseMatrix;
typedef BaseMatrixT<int> IBaseMatrix;

}  // namespace paddle
