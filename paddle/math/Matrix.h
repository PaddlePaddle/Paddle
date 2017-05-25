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
#include <memory>
#include <thread>

#include "paddle/utils/Logging.h"
#include "paddle/utils/ThreadLocal.h"

#include <hl_gpu.h>

#include "BaseMatrix.h"
#include "MemoryHandle.h"
#include "Vector.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/ThreadLocal.h"

namespace paddle {

/// TODO(tianbing), move to paddle/function/TensorType.h
enum SparseValueType { NO_VALUE = 0, FLOAT_VALUE = 1 };

/**
 * @brief  matrix sparse_format .
 *
 * nnz represents nonzero number in sparse matrix.
 *
 * SPARSE_CSR: row major matrix. length of row is height_ + 1, each element
 * represents row start index in Matrix. length of col and value are nnz.
 *
 * SPARSE_CSC: col major matrix. length of col is width_ + 1, each element
 * represents col start index in Matrix. length of col and value are nnz.
 *
 * @code
 * for example: [0, 1, 0, 2, 0;
 *               1, 0, 0, 0, 0;
 *               0, 0, 0, 2, 5];
 * SPARSE_CSR row   [0, 2, 3, 5];
 *            col   [1, 3, 0, 3, 4];
 *            value [1, 2, 1, 2, 5]
 * SPARSE_CSC col   [0, 1, 2, 2, 4, 5];
 *            row   [1, 0, 0, 2, 2];
 *            value [1, 1, 2, 2, 5]
 * @endcode
 */
/// TODO(tianbing), move to paddle/function/TensorType.h
enum SparseFormat { SPARSE_CSR = 0, SPARSE_CSC = 1 };

class Matrix;
class GpuMatrix;
class CpuMatrix;
class CpuSparseMatrix;
class GpuSparseMatrix;
typedef std::shared_ptr<Matrix> MatrixPtr;
typedef std::shared_ptr<GpuMatrix> GpuMatrixPtr;
typedef std::shared_ptr<CpuMatrix> CpuMatrixPtr;
typedef std::shared_ptr<GpuSparseMatrix> GpuSparseMatrixPtr;
typedef std::shared_ptr<CpuSparseMatrix> CpuSparseMatrixPtr;

/**
 * Copy or assignemnt constructor will share the data as opposed to making a
 * copy of the original data. To make a copy of the orinal data, use copyFrom()
 * instead.
 */
class Matrix : public BaseMatrix {
protected:
  Matrix(MemoryHandlePtr memHandle,
         size_t height,
         size_t width,
         bool trans,
         bool use_gpu);

  Matrix(real* data, size_t height, size_t width, bool trans, bool use_gpu);

  Matrix(real* data,
         size_t height,
         size_t width,
         size_t stride,
         bool trans,
         bool use_gpu);

  static ThreadLocal<MatrixPtr> tmpMat_;

public:
  size_t elementCnt_;  // maximal number of elements which can be held in data_
  MemoryHandlePtr memoryHandle_;

public:
  virtual ~Matrix() {}

  static MatrixPtr create(MemoryHandlePtr memHandle,
                          size_t height,
                          size_t width,
                          bool trans = false);
  static MatrixPtr create(size_t height,
                          size_t width,
                          bool trans = false,
                          bool useGpu = false);
  static MatrixPtr create(real* data,
                          size_t height,
                          size_t width,
                          bool trans = false,
                          bool useGpu = false);
  static MatrixPtr create(real* data,
                          size_t height,
                          size_t width,
                          size_t stride,
                          bool trans = false,
                          bool useGpu = false);

  static MatrixPtr createSparseMatrix(size_t height,
                                      size_t width,
                                      size_t nnz,
                                      SparseValueType valueType = FLOAT_VALUE,
                                      bool trans = false,
                                      bool useGpu = false);
  static MatrixPtr createSparseMatrix(size_t height,
                                      size_t width,
                                      size_t nnz,
                                      SparseValueType valueType = FLOAT_VALUE,
                                      SparseFormat foramt = SPARSE_CSR,
                                      bool trans = false,
                                      bool useGpu = false);

  static MatrixPtr createSparseMatrix(real* data,
                                      int* row,
                                      int* col,
                                      size_t height,
                                      size_t width,
                                      size_t nnz, /* used to allocate space */
                                      SparseValueType valueType, /*value type*/
                                      SparseFormat format,
                                      bool trans,
                                      bool useGpu);

  static void resizeOrCreateSparseMatrix(
      MatrixPtr& matrix,
      size_t height,
      size_t width,
      size_t nnz,
      SparseValueType valueType = FLOAT_VALUE,
      SparseFormat foramt = SPARSE_CSR,
      bool trans = false,
      bool useGpu = false);

  static void resizeOrCreate(MatrixPtr& a,
                             size_t height,
                             size_t width,
                             bool trans = false,
                             bool useGpu = false);

  /**
   * @brief  set the data buffer used to hold the matrix data.
   *
   * caller should make sure that the size of data is at least
   * sizeof(real)*height*width.
   */
  void setData(real* data) {
    BaseMatrix::setData(data);
    memoryHandle_.reset();
  }

  /// the data should be contiguous
  void setData(real* data, size_t newHeight, size_t newWidth) {
    setData(data);
    height_ = newHeight;
    width_ = newWidth;
    elementCnt_ = newHeight * newWidth;
    stride_ = width_;
  }

  size_t getWidth() const { return width_; }
  size_t getHeight() const { return height_; }
  size_t getStride() const { return stride_; }
  size_t getElementCnt() const { return elementCnt_; }
  virtual real* getData() { return data_; }
  virtual const real* getData() const { return data_; }
  bool isTransposed() const { return trans_; }
  bool isContiguous() const { return stride_ == width_ || height_ == 1; }

  // If sparse matrix, need to dynamic_cast to CpuSparseMatrix/GpuSparseMatrix
  // befor call the following functions.
  // Declare these functions in the base class just easy to call them.
  // And these declarations should be moved to base class of sparse matrix
  // if refactor sparse matrix
  virtual int* getRows() const {
    LOG(FATAL) << "Not implemented";
    return nullptr;  //! suppress warning for no return value.
  }

  virtual int* getCols() const {
    LOG(FATAL) << "Not implemented";
    return nullptr;  //! suppress warning for no return value.
  }

  virtual SparseFormat getFormat() const {
    LOG(FATAL) << "Not implemented";
    return SPARSE_CSR;  //! suppress warning for no return value.
  }

  virtual SparseValueType getValueType() const {
    LOG(FATAL) << "Not implemented";
    return NO_VALUE;  //! suppress warning for no return value.
  }

  /**
   * @brief matrix elment-wise add
   *
   * Named add3 just because add/add2 has been used in BaseMatrix.cu
   * and they are not virtual function.
   */
  virtual void add3(MatrixPtr b) { LOG(FATAL) << "Not implemented"; }

  MemoryHandlePtr getMemoryHandle() const { return memoryHandle_; }

  virtual void zeroMem() { LOG(FATAL) << "Not implemented"; }

  virtual void resetOne() { LOG(FATAL) << "Not implemented"; }

  void setDiag(real value);

  virtual void copyFrom(const Matrix& src) { LOG(FATAL) << "Not implemented"; }

  virtual void trimFrom(const CpuSparseMatrix& src) {
    LOG(FATAL) << "Not implemented";
  }

  // asynchronous copy
  virtual void copyFrom(const Matrix& src, hl_stream_t stream) {
    LOG(FATAL) << "Not implemented";
  }

  MatrixPtr subMatrix(size_t startRow,
                      size_t endRow,
                      size_t startCol,
                      size_t endCol);

  MatrixPtr subRowMatrix(size_t startRow, size_t endRow) {
    return subMatrix(startRow, endRow, 0, getWidth());
  }

  MatrixPtr subColMatrix(size_t startCol, size_t endCol) {
    return subMatrix(0, getHeight(), startCol, endCol);
  }

  virtual MatrixPtr subMatrix(size_t startRow, size_t numRows) {
    CHECK_LE(startRow + numRows, getHeight());
    return Matrix::create(getData() + startRow * getWidth(),
                          numRows,
                          getWidth(),
                          trans_,
                          useGpu_);
  }
  virtual MatrixPtr subMatrix(size_t startRow, size_t numRows, MatrixPtr dest) {
    CHECK_LE(startRow + numRows, getHeight());
    CHECK_EQ(useGpu_, dest->useGpu_);
    dest->setData(this->rowBuf(startRow), numRows, getWidth());
    return dest;
  }

  /**
   * If this is GpuMatrix, src is assumed to be CPU memory
   *
   * If this is CpuMatrix, src is assumed to be CPU memory
   */
  virtual void copyFrom(const real* src, size_t size) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void copyFrom(const real* src, const int64_t* seq) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief convert a int vector to a real matrix.
   *
   * (1) source and dest are both in CPU.
   *
   * (2) sizes are exactly match.
   */
  virtual void copyFrom(const IVector& src) {
    LOG(FATAL) << "copy data from int vector only available on CpuMatrix.";
  }

  virtual void copyByRowIndex(Matrix& b, const IVector& rowIndex) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief Create a matrix with the same type (GpuMatrix, CpuMatrix,
   *        NonValueSparseMatrix, etc.) as this.
   *
   * If height and width is zero, the new matrix will have the same size
   * as this, otherwise the new matrix will have the specified size.
   *
   */
  virtual MatrixPtr clone(size_t height = 0,
                          size_t width = 0,
                          bool useGpu = false) {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  virtual real* getRowBuf(size_t row) {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  virtual real getElement(size_t x, size_t y) const {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  virtual real getSum() {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  virtual void accumulateColSum(Matrix& src) {
    LOG(FATAL) << "Not implemented";
  }

  virtual real getAbsSum() {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  /**
   * @note Original data may not be preserved after resize().
   */
  virtual void resize(size_t newHeight, size_t newWidth) = 0;

  /**
   * @note This should only be used for sparse matrix.
   */
  virtual void resize(size_t newHeight,
                      size_t newWidth,
                      size_t newNnz, /* total item used to allocate space */
                      SparseValueType valueType,
                      SparseFormat format) = 0;

  /**
   * @brief This should only be used for sparse matrix.
   *
   * Currently must be called for each row in order.
   * The matrix is not valid until setRow is called for the last row.
   */
  virtual void setRow(size_t row,
                      size_t colNum,
                      const unsigned int* cols,
                      const real* values) = 0;

  virtual MatrixPtr getTranspose() = 0;

  /**
   * @brief  hard transpose.
   *
   * allocate matTrans' memory outside, then set memAlloc as false;
   * else set as true.
   */
  virtual void transpose(MatrixPtr& matTrans, bool memAlloc) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief  rotate 90 degrees in clock-wise if clockWise=true;
   *         otherwise rotate in anti clock-wise
   * clock-wise:
   * \f[
   *   y(j,i) = x(M-i-1,j)
   * \f]
   * anti clock-wise:
   * \f[
   *   y(j,i) = x(i, N-1-j)
   * \f]
   * where \f$x\f$ is (M x N) input, and \f$y\f$ is (N x M) output.
   *
   * allocate matRot' memory outside, then set memAlloc as false;
   * else set as true.
   */
  virtual void rotate(MatrixPtr& matRot, bool memAlloc, bool clockWise) {
    LOG(FATAL) << "Not implemented";
  }

  virtual MatrixPtr getInverse() {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  /**
   * @brief  inverse.
   *
   * if allocate matInv's memory outside, then set memAlloc as false;
   * else set as true.
   */
  virtual void inverse(MatrixPtr& matInv, bool memAlloc) {
    LOG(FATAL) << "Not implemented";
  }

public:
  /// Only set all variables to 0 or NULL but not free them.
  virtual void clear() {
    height_ = 0;
    width_ = 0;
    data_ = NULL;
  }

  void reshape(size_t height, size_t width);

  /// add b to each sample of this.
  virtual void addBias(Matrix& b, real scale) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void addSharedBias(Matrix& b, real scale) {
    LOG(FATAL) << "Not implemented";
  }

  void addBias(Matrix& b, real scale, bool sharedBias) {
    if (!sharedBias) {
      addBias(b, scale);
    } else {
      addSharedBias(b, scale);
    }
  }

  /// add each sample from a to this.
  virtual void collectBias(Matrix& a, real scale) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void collectSharedBias(Matrix& a, real scale) {
    LOG(FATAL) << "Not implemented";
  }

  void collectBias(Matrix& a, real scale, bool sharedBias) {
    if (!sharedBias) {
      collectBias(a, scale);
    } else {
      collectSharedBias(a, scale);
    }
  }

  virtual void sequenceAvgForward(Matrix& a,
                                  const IVector& startsPos,
                                  int mode) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void sequenceAvgBackward(Matrix& a,
                                   const IVector& startsPos,
                                   int mode) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * this = scaleAB*(a*b) + scaleT*this
   * @endcode
   */
  virtual void mul(const Matrix& a,
                   const Matrix& b,
                   real scaleAB,
                   real scaleT) {
    LOG(FATAL) << "Not implemented";
  }

  /// Add a vector (column) b to matrix a, column by column.
  virtual void addColumnVector(const Matrix& b) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * For j < codeLength:
   *   this(i, j) += vec(index(i, j), 0)
   * where index(i, j) = ((codes(i) + numClasses) >> (j + 1)) - 1
   * @endcode
   */
  virtual void addByBitCode(size_t numClasses,
                            const IVector& codes,
                            const Matrix& vec) {
    (void)numClasses;
    (void)codes;
    (void)vec;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength:
   *   vec(index(i, j), 0) += this(i, j)
   * where index is same as the index for addByBitCode
   * @endcode
   */
  virtual void addByBitCodeBackward(size_t numClasses,
                                    const IVector& codes,
                                    Matrix& vec) {
    (void)numClasses;
    (void)codes;
    (void)vec;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength:
   *   this(i, j) += <mat.row(index(i, j)), input.row(i)>
   * where index is same as the index for addByBitCode
   * @endcode
   */
  virtual void mulByBitCode(size_t numClasses,
                            const IVector& codes,
                            const Matrix& mat,
                            const Matrix& input) {
    (void)numClasses;
    (void)codes;
    (void)mat;
    (void)input;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength:
   *   mat.row(index(i, j)) += this(i, j) * input.row(i)
   * where index is same as the index for addByBitCode
   * @endcode
   */
  virtual void mulByBitCodeBackwardWeight(size_t numClasses,
                                          const IVector& codes,
                                          Matrix& mat,
                                          const Matrix& input) {
    (void)numClasses;
    (void)codes;
    (void)mat;
    (void)input;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength:
   *   input.row(i) += this(i, j) * mat.row(index(i, j))
   * where index is same as the index for addByBitCode
   * @endcode
   */
  virtual void mulByBitCodeBackwardError(size_t numClasses,
                                         const IVector& codes,
                                         const Matrix& mat,
                                         Matrix& input) {
    (void)numClasses;
    (void)codes;
    (void)mat;
    (void)input;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength
   *   sum(i, 0) = scaleSum * \sum_j  bit(i, j) * this(i, j)
   * where bit(i, j) = ((codes(i) + numClasses) & 2^j) ? 1 : 0
   * @endcode
   */
  virtual void sumByBitCode(size_t numClasses,
                            IVector& codes,
                            Matrix& sum,
                            real scaleSum) {
    (void)numClasses;
    (void)codes;
    (void)sum;
    (void)scaleSum;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * For j < codeLength
   *  this(i, j) -= bit(i, j)
   * where bit(i, j) is same as that for sumByBitCode
   * @endcode
   */
  virtual void subByBitCode(size_t numClasses_, IVector& codes) {
    (void)numClasses_;
    (void)codes;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * add the sum of each row of this to mat
   */
  virtual void rowSum(Matrix& sum) {
    (void)sum;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * set the max of each row of this to mat
   */
  virtual void rowMax(Matrix& max) {
    (void)max;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * set the max of each column of this to mat
   */
  virtual void colMax(Matrix& max) { LOG(FATAL) << "not implemented"; }

  /**
   * @brief Get the top k elements of each column of this matrix.
   *
   * The row ids and values of these elements are stored in
   * maxIds and max respectively. where k is the size of maxIds.
   * And note that the top k elements are not sorted.
   */
  virtual void colMax(IVector& maxIds, Matrix& maxVal) {
    LOG(FATAL) << "not implemented";
  }

  virtual void maxoutForward(Matrix& a,
                             IVector& id,
                             size_t channels,
                             size_t groups) {
    LOG(FATAL) << "not implemented";
  }

  virtual void maxoutBackward(Matrix& a,
                              IVector& id,
                              size_t channels,
                              size_t groups) {
    LOG(FATAL) << "not implemented";
  }

  virtual void rowMaxId(IVector& maxIds) { LOG(FATAL) << "Not implemented"; }

  /**
   * @brief Get the top k elements of each row of this matrix.
   *
   * The column ids and values of these elements are stored in
   * maxIds and max respectively. where k is the size of maxIds.
   * And note that the top k elements are not sorted.
   */
  virtual void rowMax(IVector& maxIds, Matrix& max) {
    LOG(FATAL) << "Not implemented";
  }

  /// normalize each row so that the sum of each row is 1.
  virtual void rowNormalizeL1(Matrix& out) {
    (void)out;
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   *  this = a*b
   * @endcode
   */
  virtual void mul(const Matrix& a, const Matrix& b) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * this = scaleAB*(this*b) +  scaleT*this
   * @endcode
   */
  virtual void rightMul(Matrix& b, real scaleAB, real scaleT) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * this = this* b
   * @endcode
   */
  virtual void rightMul(Matrix& b) { LOG(FATAL) << "Not implemented"; }

  /**
   * @code
   * this = scaleAB*(a*this) +  scaleT*this
   * @endcode
   */
  virtual void leftMul(Matrix& a, real scaleAB, real scaleT) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * this = a*this)
   * @endcode
   */
  virtual void leftMul(Matrix& a) { LOG(FATAL) << "Not implemented"; }

  /// merge the element for each col.
  virtual void colMerge(Matrix& src) { LOG(FATAL) << "Not implemented"; }

  /// copy -log(output[label]) to this->data[i].
  virtual void oneHotCrossEntropy(Matrix& output, IVector& label) {
    LOG(FATAL) << "Not implemented";
  }

  /// calculate the error of outputV according to label.
  virtual void oneHotCrossEntropyBp(Matrix& outputV, IVector& label) {
    LOG(FATAL) << "Not implemented";
  }

  /// copy -log(output[label]) to this->data[i].
  virtual void oneHotCrossEntropyWithSelfNorm(Matrix& output,
                                              IVector& label,
                                              real alpha) {
    LOG(FATAL) << "Not implemented";
  }

  /// calculate the error of outputV according to label.
  virtual void oneHotCrossEntropyWithSelfNormBp(Matrix& outputV,
                                                IVector& label,
                                                real alpha) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * \f[
   *  a[i] = \sum_{j=-(N-1)/2}^{(N-1)/2} b_{i+j} * c_{j}
   * \f]
   *
   * b contains M elements,
   * c contains N elements (N is odd),
   * b's index arithmetic is computed modulo M,
   * c's index arithmetic is computed modulo N.
   */
  virtual void circularConv(Matrix& b, Matrix& c) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void circularConvDerivative(Matrix& output,
                                      Matrix& prevOut1,
                                      Matrix& prevOut2,
                                      Matrix& prevGrad1,
                                      Matrix& prevGrad2) {
    LOG(FATAL) << "Not implemented";
  }

  /* output_ij = exp(this_{ij}) / (sum_j exp(this_ij)) */
  virtual void softmax(Matrix& output) {
    (void)output;
    LOG(FATAL) << "Not implemeted";
  }
  virtual void sequenceSoftmax(Matrix& output, const IVector& index) {
    (void)output;
    LOG(FATAL) << "Not implemeted";
  }

  virtual void softmaxBackward(Matrix& outputV) {
    (void)outputV;
    LOG(FATAL) << "Not implemeted";
  }

  /*
    sum_i = sum_j this_ij * output_ij
    this_ij = output_ij* (this_ij - sum_i)
  */
  virtual void softmaxDerivative(Matrix& output, Matrix& sftmaxSum) {
    LOG(FATAL) << "Not implemented";
  }

  /// calculate the sum of squares diff cost.
  virtual void sumOfSquares(Matrix& output, Matrix& label) {
    LOG(FATAL) << "Not implemented";
  }

  /// gradient of sumOfSquares.
  virtual void sumOfSquaresBp(Matrix& outputV, Matrix& label) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void smoothL1(Matrix& output, Matrix& label, real destScale) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void smoothL1Bp(Matrix& outputV, Matrix& label, real destScale) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void tanh(Matrix& output) { LOG(FATAL) << "Not implemented"; }

  virtual void tanhDerivative(Matrix& output) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void softrelu(Matrix& output) { LOG(FATAL) << "Not implemented"; }

  virtual void softreluDerivative(Matrix& output) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void scaledTanh(Matrix& output, real p1, real p2) {
    LOG(FATAL) << "Not implemented";
  }

  /// print out the values of elements to os
  virtual void print(std::ostream& os) const {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * print a part of the matrix
   * from the (top,left) value to the (height, width) value (not included)
   */
  virtual void print(std::ostream& os, size_t height, size_t width) const {
    LOG(FATAL) << "Not implemented";
  }

  /// print one row to os
  virtual void printOneRow(std::ostream& os, size_t idx) const {
    LOG(FATAL) << "Not implemented";
  }

  virtual void check(std::ostream& os, Matrix& refMat, bool printDiff = true) {}

  virtual real getMin() {
    LOG(FATAL) << "Not implemented";
    return 0;
  }
  virtual real getMax() {
    LOG(FATAL) << "Not implemented";
    return 0;
  }

  virtual void randomizeUniform() { LOG(FATAL) << "Not implemented"; }

  /**
   * @brief  calulate the error of classification
   *
   * output[i] = 1 if row i is an error.
   *
   * output[i] = 0 if row i is correct.
   *
   */
  virtual void classificationError(Matrix& output,
                                   IVector& label,
                                   size_t topkSize = 1) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * This function is used to calculate the convolution:
   *
   * It will expand a feature matrix according to the
   * convolution filters
   */
  virtual void convExpand(Matrix& feature,
                          int feaImgHeight,
                          int feaImgWidth,
                          int channels,
                          int blockH,
                          int blockW,
                          int strideH,
                          int strideW,
                          int paddingH,
                          int paddingW,
                          int outputH,
                          int outputW) {
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * This function is the reverse implementation of convExpand:
   *
   * Its function is to restore a expanded-matrix into a feature matrix
   */
  virtual void convShrink(Matrix& expandColMat,
                          int thisImgHeight,
                          int thisImgWidth,
                          int channels,
                          int blockH,
                          int blockW,
                          int strideH,
                          int strideW,
                          int paddingH,
                          int paddingW,
                          int outputH,
                          int outputW,
                          real alpha = 1.0f,
                          real beta = 0.0f) {
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * Pooling forward operation, pick out the largest element
   * in the sizeX of value
   */
  virtual void maxPoolForward(Matrix& inputMat,
                              size_t imgSizeH,
                              size_t imgSizeW,
                              size_t channels,
                              size_t sizeX,
                              size_t sizeY,
                              size_t strideH,
                              size_t strideW,
                              size_t outputH,
                              size_t outputW,
                              size_t paddingH,
                              size_t paddingW) {
    LOG(FATAL) << "Not implemeted";
  }

  /// Pooling backward operation.
  virtual void maxPoolBackward(Matrix& image,
                               size_t imgSizeH,
                               size_t imgSizeW,
                               Matrix& outGrad,
                               Matrix& outV,
                               size_t sizeX,
                               size_t sizeY,
                               size_t strideH,
                               size_t strideW,
                               size_t outputH,
                               size_t outputW,
                               real scaleTargets,
                               real scaleOutput,
                               size_t paddingH,
                               size_t paddingW) {
    LOG(FATAL) << "Not implemeted";
  }

  /// Pooling forward operation, caculate the average of sizeX elements.
  virtual void avgPoolForward(Matrix& input,
                              size_t imgSizeH,
                              size_t imgSizeW,
                              size_t channels,
                              size_t sizeX,
                              size_t sizeY,
                              size_t strideH,
                              size_t strideW,
                              size_t outputH,
                              size_t outputW,
                              size_t paddingH,
                              size_t paddingW) {
    LOG(FATAL) << "Not implemeted";
  }

  virtual void avgPoolBackward(Matrix& input,
                               size_t imgSizeH,
                               size_t imgSizeW,
                               size_t sizeX,
                               size_t sizeY,
                               size_t strideH,
                               size_t strideW,
                               size_t outputH,
                               size_t outputW,
                               real scaleTargets,
                               real scaleOutput,
                               size_t paddingH,
                               size_t paddingW) {
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * Input: one or more sequences. Each sequence contains some instances.
   *
   * Output: output size is the number of input sequences (NOT input
   * instances).
   *
   * output[i] is set to max_input[i].
   */
  virtual void maxSequenceForward(Matrix& input,
                                  const IVector& sequence,
                                  IVector& index) {
    LOG(FATAL) << "Not implemeted";
  }

  virtual void maxSequenceBackward(Matrix& outputGrad,
                                   const IVector& sequence,
                                   IVector& index) {
    LOG(FATAL) << "Not implemeted";
  }

  /**
   * @code
   * this.row[i] += table.row[ids[i]]
   * if ids[i] == -1, it will be ignored
   * @endcode
   */
  virtual void selectRows(Matrix& table, IVector& ids) {
    (void)table;
    (void)ids;
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * this[i] = table[i, id[i]]
   * @endcode
   */
  virtual void selectElements(Matrix& table, IVector& ids) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * table.row[ids[i]] += this.row[i]
   * if ids[i] == -1, it will be ignored
   * @endcode
   */
  virtual void addToRows(Matrix& table, IVector& ids) {
    (void)table;
    (void)ids;
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @code
   * table[i, id[i]] += this[i]
   * @endcode
   */
  virtual void addElements(Matrix& table, IVector& ids) {
    LOG(FATAL) << "Not implemented";
  }
  /**
   * @brief  cross entropy for multi binary labels
   *
   * @code
   * this[i] = -sum(label[i][j]*log(output[i][j])
   *           + (1-label[i][j])*log(1-output[i][j]))
   * @endcode
   */
  virtual void multiBinaryLabelCrossEntropy(Matrix& output, Matrix& label) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief  The gradient of cross entropy for multi binary labels on output
   *
   * @code
   * this[i][j] = -label[i][j]/output[i][j]
   *              + (1-label[i][j])/(1-output[i][j])
   * @endcode
   */
  virtual void multiBinaryLabelCrossEntropyBp(Matrix& output, Matrix& label) {
    LOG(FATAL) << "Not implemented";
  }

  /**
   * @brief  Calculate the classification error for multi binary labels
   *
   * @code
   * this[i] = sum((output[i][j] >= threshold && label[i][j] == 0)
   *            || (output[i][j] < threshold && label[i][j] == 1))
   *            / output->getWidth()
   * @endcode
   */
  virtual void classificationErrorMulti(Matrix& output,
                                        Matrix& label,
                                        real threshold) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void paramReluForward(Matrix& data, Matrix& W) {
    LOG(FATAL) << "Not implemented";
  }
  virtual void paramReluBackwardW(Matrix& oGrad, Matrix& data) {
    LOG(FATAL) << "Not implemented";
  }
  virtual void paramReluBackwardDiff(Matrix& oGrad, Matrix& data, Matrix& W) {
    LOG(FATAL) << "Not implemented";
  }

  virtual void bilinearForward(const Matrix& in,
                               const size_t inImgH,
                               const size_t inImgW,
                               const size_t outImgH,
                               const size_t outImgW,
                               const size_t numChannels,
                               const real ratioH,
                               const real ratioW) {
    LOG(FATAL) << "Not implemented";
  }
  virtual void bilinearBackward(const Matrix& out,
                                const size_t outImgH,
                                const size_t outImgW,
                                const size_t inImgH,
                                const size_t inImgW,
                                const size_t numChannels,
                                const real ratioH,
                                const real ratioW) {
    LOG(FATAL) << "Not implemented";
  }

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    if (useGpu_) {
      TensorGpuApply<real>(*this, expr);
    } else {
      TensorCpuApply<real>(*this, expr);
    }
  }

  bool isEmpty() const { return data_ == nullptr; }

  explicit operator bool() const { return !isEmpty(); }
};

inline std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
  mat.print(os);
  return os;
}

class GpuMatrix : public Matrix {
public:
  GpuMatrix();

  GpuMatrix(size_t height, size_t width, bool trans = false);
  GpuMatrix(real* data, size_t height, size_t width, bool trans = false)
      : Matrix(data, height, width, trans, true) {}
  GpuMatrix(real* data,
            size_t height,
            size_t width,
            size_t stride,
            bool trans = false)
      : Matrix(data, height, width, stride, trans, true) {}
  GpuMatrix(GpuMemHandlePtr dataHandle,
            size_t height,
            size_t width,
            bool trans = false)
      : Matrix(dataHandle, height, width, trans, true) {}
  ~GpuMatrix();

  void zeroMem();
  void resetOne();
  void setDiag(real value);

  void resize(size_t newHeight, size_t newWidth);
  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format) {
    LOG(FATAL) << "Only Support Sparse Matrix";
  }
  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values) {
    LOG(FATAL) << "Only Support Sparse Matrix";
  }

  /**
   * Copy the data from cpu_memory buffer
   */
  void copyFrom(const real* hostSrc, size_t size);

  void copyFrom(const real* hostSrc, const int64_t* seq);

  void copyFrom(const Matrix& src, hl_stream_t stream);

  void copyFrom(const Matrix& src);

  void copyFrom(const IVector& src);

  void copyByRowIndex(Matrix& b, const IVector& rowIndex);

  MatrixPtr clone(size_t height, size_t width, bool useGpu = false);

  real getElement(size_t x, size_t y) const;

  real* getRow(size_t row) { return BaseMatrix::rowBuf(row); }
  virtual real* getRowBuf(size_t row) { return getRow(row); }

  real getSum();
  void accumulateColSum(Matrix& src);
  real getAbsSum();

  real getMin();
  real getMax();

  MatrixPtr getTranspose();
  void transpose(MatrixPtr& matTrans, bool memAlloc);
  void rotate(MatrixPtr& matRot, bool memAlloc, bool clockWise);

  MatrixPtr getInverse();
  void inverse(MatrixPtr& matInv, bool memAlloc);

  /// add b to each sample of this.
  void addBias(Matrix& b, real scale);
  void addSharedBias(Matrix& b, real scale);

  /**
   * @code
   * add each sample from a to this.
   * @endcode
   */
  void collectBias(Matrix& a, real scale);
  void collectSharedBias(Matrix& a, real scale);

  void sequenceAvgForward(Matrix& a, const IVector& startsPos, int mode);
  void sequenceAvgBackward(Matrix& a, const IVector& startsPos, int mode);

  /**
   * @code
   * this.row[i] += table.row[ids[i]]
   * @endcode
   */
  virtual void selectRows(Matrix& table, IVector& ids);

  /**
   * @code
   * this[i] = table[i, id[i]]
   * @endcode
   */
  virtual void selectElements(Matrix& table, IVector& ids);

  /**
   * @code
   * table.row[ids[i]] += this.row[i]
   * @endcode
   */
  virtual void addToRows(Matrix& table, IVector& ids);

  void addColumnVector(const Matrix& b);

  /**
   * @code
   * this = scaleAB*(a*b) + scaleT*this
   * @endcode
   */
  void mul(const Matrix& a, const Matrix& b, real scaleAB, real scaleT);

  /**
   * @code
   * this = a*b
   * @endcode
   */
  void mul(const Matrix& a, const Matrix& b);

  void mul(const GpuMatrix& a, const GpuMatrix& b, real scaleAB, real scaleT);

  void mul(const GpuSparseMatrix& a,
           const GpuMatrix& b,
           real scaleAB,
           real scaleT);

  void mul(const GpuMatrix& a,
           const GpuSparseMatrix& b,
           real scaleAB,
           real scaleT);

  /**
   * @code
   * this = scaleAB*(this*b) +  scaleT*this
   * @endcode
   */
  void rightMul(Matrix& b, real scaleAB, real scaleT);

  /**
   * @code
   * this = this* b
   * @endcode
   */
  void rightMul(Matrix& b);

  /**
   * @code
   * this = scaleAB*(a*this) +  scaleT*this
   * @endcode
   */
  void leftMul(Matrix& a, real scaleAB, real scaleT);

  /**
   * @code
   * this = a*this
   * @endcode
   */
  void leftMul(Matrix& a);

  void colMerge(Matrix& src);
  void rowSum(Matrix& sum);
  void rowMax(Matrix& max);
  void rowMax(IVector& maxIds, Matrix& max);
  void colMax(Matrix& max);
  void colMax(IVector& maxIds, Matrix& max);
  void maxoutForward(Matrix& a, IVector& id, size_t channels, size_t groups);
  void maxoutBackward(Matrix& a, IVector& id, size_t channels, size_t groups);

  void oneHotCrossEntropy(Matrix& output, IVector& label);
  void oneHotCrossEntropyBp(Matrix& outputV, IVector& label);
  void oneHotCrossEntropyWithSelfNorm(Matrix& output,
                                      IVector& label,
                                      real alpha);
  void oneHotCrossEntropyWithSelfNormBp(Matrix& outputV,
                                        IVector& label,
                                        real alpha);

  void softmax(Matrix& output);
  void sequenceSoftmax(Matrix& output, const IVector& index);
  void softmaxBackward(Matrix& outputV);
  void softmaxDerivative(Matrix& output, Matrix& sftmaxSum);

  /// calculate the sum of squares diff cost.
  void sumOfSquares(Matrix& output, Matrix& label);

  /// gradient of sumOfSquares.
  void sumOfSquaresBp(Matrix& outputV, Matrix& label);
  void tanh(Matrix& output);
  void tanhDerivative(Matrix& output);
  void softrelu(Matrix& output);
  void softreluDerivative(Matrix& output);
  void scaledTanh(Matrix& output, real p1, real p2);

  virtual void print(std::ostream& os) const;
  virtual void print(std::ostream& os, size_t height, size_t width) const;

  void paramReluForward(Matrix& data, Matrix& W);
  void paramReluBackwardW(Matrix& oGrad, Matrix& data);
  void paramReluBackwardDiff(Matrix& oGrad, Matrix& data, Matrix& W);

  void check(std::ostream& os, Matrix& refMat, bool printDiff = true);
  void randomizeUniform();

  void classificationError(Matrix& output, IVector& label, size_t topkSize = 1);

  void convExpand(Matrix& feature,
                  int feaImgHeight,
                  int feaImgWidth,
                  int channels,
                  int blockH,
                  int blockW,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  int outputH,
                  int outputW);

  void convShrink(Matrix& expandColMat,
                  int thisImgHeight,
                  int thisImgWidth,
                  int channels,
                  int blockH,
                  int blochW,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingWreal,
                  int outputH,
                  int outputW,
                  real alpha = 1.0f,
                  real beta = 0.0f);

  void maxPoolForward(Matrix& inputMat,
                      size_t imgSizeH,
                      size_t imgSizeW,
                      size_t channels,
                      size_t sizeX,
                      size_t sizeY,
                      size_t strideH,
                      size_t strideW,
                      size_t outputH,
                      size_t outputW,
                      size_t paddingH,
                      size_t paddingW);

  void maxPoolBackward(Matrix& image,
                       size_t imgSizeH,
                       size_t imgSizeW,
                       Matrix& outGrad,
                       Matrix& outV,
                       size_t sizeX,
                       size_t sizeY,
                       size_t strideH,
                       size_t strideW,
                       size_t outputH,
                       size_t outputW,
                       real scaleTargets,
                       real scaleOutput,
                       size_t paddingH,
                       size_t paddingW);

  void avgPoolForward(Matrix& input,
                      size_t imgSizeH,
                      size_t imgSizeW,
                      size_t channels,
                      size_t sizeX,
                      size_t sizeY,
                      size_t strideH,
                      size_t strideW,
                      size_t outputH,
                      size_t outputW,
                      size_t paddingH,
                      size_t paddingW);

  void avgPoolBackward(Matrix& input,
                       size_t imgSizeH,
                       size_t imgSizeW,
                       size_t sizeX,
                       size_t sizeY,
                       size_t strideH,
                       size_t strideW,
                       size_t outputH,
                       size_t outputW,
                       real scaleTargets,
                       real scaleOutput,
                       size_t paddingH,
                       size_t paddingW);

  void maxSequenceForward(Matrix& input,
                          const IVector& sequence,
                          IVector& index);

  void maxSequenceBackward(Matrix& outputGrad,
                           const IVector& sequence,
                           IVector& index);

  void bilinearForward(const Matrix& in,
                       const size_t inImgH,
                       const size_t inImgW,
                       const size_t outImgH,
                       const size_t outImgW,
                       const size_t numChannels,
                       const real ratioH,
                       const real ratioW);

  void bilinearBackward(const Matrix& out,
                        const size_t outImgH,
                        const size_t outImgW,
                        const size_t inImgH,
                        const size_t inImgW,
                        const size_t numChannels,
                        const real ratioH,
                        const real ratioW);

  void multiBinaryLabelCrossEntropy(Matrix& output, Matrix& label);

  void multiBinaryLabelCrossEntropyBp(Matrix& output, Matrix& label);

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorGpuApply<real>(*this, expr);
  }
};

class CpuMatrix : public Matrix {
public:
  CpuMatrix(size_t height, size_t width, bool trans = false);
  CpuMatrix(real* data, size_t height, size_t width, bool trans = false)
      : Matrix(data, height, width, trans, false) {}
  CpuMatrix(real* data,
            size_t height,
            size_t width,
            size_t stride,
            bool trans = false)
      : Matrix(data, height, width, stride, trans, false) {}

  CpuMatrix(CpuMemHandlePtr dataHandle,
            size_t height,
            size_t width,
            bool trans = false)
      : Matrix(dataHandle, height, width, trans, false) {}

  ~CpuMatrix();

  void zeroMem();
  void resetOne();
  void setDiag(real value);

  void resize(size_t newHeight, size_t newWidth);
  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format) {
    LOG(FATAL) << "Only Support Sparse Matrix";
  }
  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values) {
    LOG(FATAL) << "Only Support Sparse Matrix";
  }

  real getElement(size_t x, size_t y) const;
  real getSum();
  void accumulateColSum(Matrix& src);
  real getAbsSum();

  MatrixPtr getTranspose();
  void transpose(MatrixPtr& matTrans, bool memAlloc);
  void rotate(MatrixPtr& matRot, bool memAlloc, bool clockWise);

  MatrixPtr getInverse();
  void inverse(MatrixPtr& matInv, bool memAlloc);

  void copyFrom(const Matrix& src);

  void copyFrom(const Matrix& src, hl_stream_t stream);

  void copyFrom(const real* cpuSrc, size_t size);

  void copyFrom(const real* cpuSrc, const int64_t* seq);

  void copyFrom(const IVector& src);

  void copyFrom(CpuSparseMatrix& src);

  void copyByRowIndex(Matrix& b, const IVector& rowIndex);

  MatrixPtr clone(size_t height, size_t width, bool useGpu = false);

  void convExpand(Matrix& feature,
                  int feaImgHeight,
                  int feaImgWidth,
                  int channels,
                  int blcokH,
                  int blockW,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  int outputH,
                  int outputW);

  void convShrink(Matrix& expandFeat,
                  int thisImgHeight,
                  int thisImgWidth,
                  int channels,
                  int blockH,
                  int blockW,
                  int strideH,
                  int strideW,
                  int paddingH,
                  int paddingW,
                  int outputH,
                  int outputW,
                  real alpha = 1.0f,
                  real beta = 0.0f);

  void maxPoolForward(Matrix& inputMat,
                      size_t imgSizeH,
                      size_t imgSizeW,
                      size_t channels,
                      size_t sizeX,
                      size_t sizeY,
                      size_t strideH,
                      size_t strideW,
                      size_t outputH,
                      size_t outputW,
                      size_t paddingH,
                      size_t paddingW);

  void maxPoolBackward(Matrix& image,
                       size_t imgSizeH,
                       size_t imgSizeW,
                       Matrix& outGrad,
                       Matrix& outV,
                       size_t sizeX,
                       size_t sizeY,
                       size_t strideH,
                       size_t strideW,
                       size_t outputH,
                       size_t outputW,
                       real scaleTargets,
                       real scaleOutput,
                       size_t paddingH,
                       size_t paddingW);

  void avgPoolForward(Matrix& input,
                      size_t imgSizeH,
                      size_t imgSizeW,
                      size_t channels,
                      size_t sizeX,
                      size_t sizeY,
                      size_t strideH,
                      size_t strideW,
                      size_t outputH,
                      size_t outputW,
                      size_t paddingH,
                      size_t paddingW);

  void avgPoolBackward(Matrix& input,
                       size_t imgSizeH,
                       size_t imgSizeW,
                       size_t sizeX,
                       size_t sizeY,
                       size_t strideH,
                       size_t strideW,
                       size_t outputH,
                       size_t outputW,
                       real scaleTargets,
                       real scaleOutput,
                       size_t paddingH,
                       size_t paddingW);

  void maxSequenceForward(Matrix& input,
                          const IVector& sequence,
                          IVector& index);

  void maxSequenceBackward(Matrix& outputGrad,
                           const IVector& sequence,
                           IVector& index);

  real* getRow(size_t row) { return BaseMatrix::rowBuf(row); }
  virtual real* getRowBuf(size_t row) { return getRow(row); }

public:
  /// add b to each sample of this.
  void addBias(Matrix& b, real scale);
  void addSharedBias(Matrix& b, real scale);

  /// add each sample of a to this.
  void collectBias(Matrix& a, real scale);
  void collectSharedBias(Matrix& a, real scale);

  void sequenceAvgForward(Matrix& a, const IVector& startsPos, int mode);
  void sequenceAvgBackward(Matrix& a, const IVector& startsPos, int mode);

  /**
   * @code
   * this.row[i] += table.row[ids[i]]
   * @endcode
   */
  virtual void selectRows(Matrix& table, IVector& ids);

  /**
   * @code
   * table.row[ids[i]] += this.row[i]
   * @endcode
   */
  virtual void addToRows(Matrix& table, IVector& ids);

  /**
   * @code
   * this[i] = table[i, id[i]]
   * @endcode
   */
  virtual void selectElements(Matrix& table, IVector& ids);

  /**
   * @code
   * table[i, id[i]] += this[i]
   * @endcode
   */
  virtual void addElements(Matrix& table, IVector& ids);

  /**
   * use abstract getRow() to get row from table.
   *
   * Define table as template instead of virtual class for performance sake.
   * internal used by above two virtual funcs.
   */
  template <typename TableMatType>
  void selectRowsImp(TableMatType& table, IVector& ids);
  template <typename TableMatType>
  void addToRowsImp(TableMatType& table, IVector& ids);

  void addColumnVector(const Matrix& b);

  void mul(const Matrix& a, const Matrix& b, real scaleAB, real scaleT);
  void mul(CpuMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

  void mul(CpuMatrix* a, CpuSparseMatrix* b, real scaleAB, real scaleT);

  static void mul(CpuMatrix* a,
                  CpuMatrix* b,
                  CpuSparseMatrix* c,
                  real scaleAB,
                  real scaleT);

  /**
   * c = a * b
   *
   * use abstract getRow() to get row from B,C.
   * Define B,C as template instead of virtual class for performance sake.
   */
  template <typename MatBType, typename MatCType>
  static void mul(
      CpuSparseMatrix* a, MatBType* b, MatCType* c, real scaleAB, real scaleT);

  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);

  void mul(const Matrix& a, const Matrix& b);

  void rightMul(Matrix& b, real scaleAB, real scaleT);
  void rightMul(Matrix& b);

  void leftMul(Matrix& a, real scaleAB, real scaleT);
  void leftMul(Matrix& a);
  void colMerge(Matrix& src);
  void rowSum(Matrix& sum);
  void rowMaxId(IVector& maxIds);
  void rowMax(Matrix& max);
  void rowMax(IVector& maxIds, Matrix& maxVal);
  void colMax(Matrix& max);
  void colMax(IVector& maxIds, Matrix& maxVal);
  void maxoutForward(Matrix& a, IVector& id, size_t channels, size_t groups);
  void maxoutBackward(Matrix& a, IVector& id, size_t channels, size_t groups);
  void rowNormalizeL1(Matrix& out);

  void oneHotCrossEntropy(Matrix& output, IVector& label);
  void oneHotCrossEntropyBp(Matrix& outputV, IVector& label);
  void oneHotCrossEntropyWithSelfNorm(Matrix& output,
                                      IVector& label,
                                      real alpha);
  void oneHotCrossEntropyWithSelfNormBp(Matrix& outputV,
                                        IVector& label,
                                        real alpha);

  void circularConv(Matrix& b, Matrix& c);
  void circularConvDerivative(Matrix& output,
                              Matrix& prevOut1,
                              Matrix& prevOut2,
                              Matrix& prevGrad1,
                              Matrix& prevGrad2);

  void softmax(Matrix& output);
  void sequenceSoftmax(Matrix& output, const IVector& index);
  void softmaxDerivative(Matrix& output, Matrix& sftmaxSum);

  /// calculate the sum of squares diff cost.
  void sumOfSquares(Matrix& output, Matrix& label);

  /// gradient of sumOfSquares.
  void sumOfSquaresBp(Matrix& outputV, Matrix& label);

  void smoothL1(Matrix& output, Matrix& label, real destScale);
  void smoothL1Bp(Matrix& output, Matrix& label, real destScale);

  void tanh(Matrix& output);
  void tanhDerivative(Matrix& output);

  void softrelu(Matrix& output);
  void softreluDerivative(Matrix& output);
  void scaledTanh(Matrix& output, real p1, real p2);

  void print(std::ostream& os) const;
  void print(std::ostream& os, size_t height, size_t width) const;
  void printOneRow(std::ostream& os, size_t idx) const;

  void paramReluForward(Matrix& data, Matrix& W);
  void paramReluBackwardW(Matrix& oGrad, Matrix& data);
  void paramReluBackwardDiff(Matrix& oGrad, Matrix& data, Matrix& W);

  void check(std::ostream& os, Matrix& refMat, bool printDiff = true);

  real getMin();
  real getMax();

  void randomizeUniform();

  void classificationError(Matrix& output, IVector& label, size_t topkSize = 1);

  void addByBitCode(size_t numClasses, const IVector& codes, const Matrix& vec);

  void addByBitCodeBackward(size_t numClasses,
                            const IVector& codes,
                            Matrix& vec);

  void mulByBitCode(size_t numClasses,
                    const IVector& codes,
                    const Matrix& mat,
                    const Matrix& input);

  void mulByBitCodeBackwardWeight(size_t numClasses,
                                  const IVector& codes,
                                  Matrix& mat,
                                  const Matrix& input);

  void mulByBitCodeBackwardError(size_t numClasses,
                                 const IVector& codes,
                                 const Matrix& mat,
                                 Matrix& input);

  void sumByBitCode(size_t numClasses,
                    IVector& codes,
                    Matrix& sum,
                    real scaleSum);

  void subByBitCode(size_t numClasses_, IVector& codes);

  void multiBinaryLabelCrossEntropy(Matrix& output, Matrix& label);
  void multiBinaryLabelCrossEntropyBp(Matrix& output, Matrix& label);
  void classificationErrorMulti(Matrix& output, Matrix& label, real threshold);

  void bilinearForward(const Matrix& in,
                       const size_t inImgH,
                       const size_t inImgW,
                       const size_t outImgH,
                       const size_t outImgW,
                       const size_t numChannels,
                       const real ratioH,
                       const real ratioW);

  void bilinearBackward(const Matrix& out,
                        const size_t outImgH,
                        const size_t outImgW,
                        const size_t inImgH,
                        const size_t inImgW,
                        const size_t numChannels,
                        const real ratioH,
                        const real ratioW);

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorCpuApply<real>(*this, expr);
  }
};

class SharedCpuMatrix : public CpuMatrix {
public:
  /* blockNum is number of partitions of the matrix  */
  SharedCpuMatrix(int blockNum, size_t height, size_t width, bool trans = false)
      : CpuMatrix(height, width, trans) {
    initShared(blockNum);
  }
  SharedCpuMatrix(
      int blockNum, real* data, size_t height, size_t width, bool trans = false)
      : CpuMatrix(data, height, width, trans) {
    initShared(blockNum);
  }

  SharedCpuMatrix(int blockNum,
                  CpuMemHandlePtr dataHandle,
                  size_t height,
                  size_t width,
                  bool trans = false)
      : CpuMatrix(dataHandle, height, width, trans) {
    initShared(blockNum);
  }

  SharedCpuMatrix(CpuMemHandlePtr dataHandle,
                  size_t height,
                  size_t width,
                  bool trans = false)
      : CpuMatrix(dataHandle, height, width, trans) {
    initBlock(1);
  }

  ~SharedCpuMatrix() {}

public:
  virtual void mul(CpuSparseMatrix* a, CpuMatrix* b, real scaleAB, real scaleT);
  virtual void add(Matrix& b, real p1, real p2);
  virtual void add(real p1, real p2);

private:
  using Matrix::mul;
  void initShared(int blockNum);
  void initBlock(int blockNum);

  int blockNum_;
  std::vector<std::unique_ptr<std::mutex>> blockLocks_;
  ThreadLocal<CpuMatrixPtr> localBuf_;
  ThreadLocal<std::vector<int>> localBufRows_;
  ThreadLocal<std::vector<int>> blockSeq_;
};

typedef struct { unsigned int col; } sparse_non_value_t;

typedef struct {
  unsigned int col;
  float value;
} sparse_float_value_t;

}  // namespace paddle
#include "ExecViaCpu.h"
