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

#include "Matrix.h"
#include "MathFunctions.h"
#include "SparseMatrix.h"
#include "SparseRowMatrix.h"

#include <float.h>
#include <algorithm>
#include <cmath>

#include <string.h>
#include "hl_cnn.h"
#include "hl_gpu.h"
#include "hl_table_apply.h"
#include "hl_top_k.h"
#include "paddle/utils/Logging.h"

#include "paddle/utils/ThreadLocal.h"

#include "SIMDFunctions.h"

namespace paddle {

inline real _pow(real a, real beta) { return std::pow(a, beta); }

inline real _square(real a) { return a * a; }

inline real _safelog(real a) { return a > 0.0f ? std::log(a) : -40.0f; }

Matrix::Matrix(MemoryHandlePtr memHandle,
               size_t height,
               size_t width,
               bool trans,
               bool use_gpu)
    : BaseMatrix(
          height,
          width,
          memHandle ? (reinterpret_cast<real*>(memHandle->getBuf())) : nullptr,
          trans,
          use_gpu) {
  elementCnt_ = width * height;
  memoryHandle_ = memHandle;
}

Matrix::Matrix(
    real* data, size_t height, size_t width, bool trans, bool use_gpu)
    : BaseMatrix(height, width, data, trans, use_gpu) {
  elementCnt_ = width * height;
}

Matrix::Matrix(real* data,
               size_t height,
               size_t width,
               size_t stride,
               bool trans,
               bool use_gpu)
    : BaseMatrix(height, width, stride, data, trans, use_gpu) {
  elementCnt_ = width * height;
}

MatrixPtr Matrix::createSparseMatrix(real* data,
                                     int* row,
                                     int* col,
                                     size_t height,
                                     size_t width,
                                     size_t nnz, /* used to allocate space */
                                     SparseValueType valueType, /*value type*/
                                     SparseFormat format,
                                     bool trans,
                                     bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuSparseMatrix>(
        data, row, col, height, width, nnz, valueType, format, trans);
  } else {
    return std::make_shared<CpuSparseMatrix>(
        data, row, col, height, width, nnz, valueType, format, trans);
  }
}

MatrixPtr Matrix::createSparseMatrix(size_t height,
                                     size_t width,
                                     size_t nnz, /* used to allocate space */
                                     SparseValueType valueType, /*value type*/
                                     SparseFormat format,
                                     bool trans,
                                     bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuSparseMatrix>(
        height, width, nnz, valueType, format, trans);
  } else {
    return std::make_shared<CpuSparseMatrix>(
        height, width, nnz, valueType, format, trans);
  }
}

MatrixPtr Matrix::create(MemoryHandlePtr memHandle,
                         size_t height,
                         size_t width,
                         bool trans) {
  if (auto gpuHandle = std::dynamic_pointer_cast<GpuMemoryHandle>(memHandle)) {
    return std::make_shared<GpuMatrix>(gpuHandle, height, width, trans);
  } else if (auto cpuHandle =
                 std::dynamic_pointer_cast<CpuMemoryHandle>(memHandle)) {
    return std::make_shared<CpuMatrix>(cpuHandle, height, width, trans);
  } else {
    LOG(FATAL) << "Wrong";
    return nullptr;
  }
}

MatrixPtr Matrix::create(size_t height, size_t width, bool trans, bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuMatrix>(height, width, trans);
  } else {
    return std::make_shared<CpuMatrix>(height, width, trans);
  }
}

MatrixPtr Matrix::create(
    real* data, size_t height, size_t width, bool trans, bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuMatrix>(data, height, width, trans);
  } else {
    return std::make_shared<CpuMatrix>(data, height, width, trans);
  }
}

MatrixPtr Matrix::create(real* data,
                         size_t height,
                         size_t width,
                         size_t stride,
                         bool trans,
                         bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuMatrix>(data, height, width, stride, trans);
  } else {
    return std::make_shared<CpuMatrix>(data, height, width, stride, trans);
  }
}

MatrixPtr Matrix::createSparseMatrix(size_t height,
                                     size_t width,
                                     size_t nnz,
                                     SparseValueType valueType,
                                     bool trans,
                                     bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuSparseMatrix>(
        height, width, nnz, valueType, SPARSE_CSR, trans);
  } else {
    return std::make_shared<CpuSparseMatrix>(
        height, width, nnz, valueType, SPARSE_CSR, trans);
  }
}

void Matrix::resizeOrCreate(
    MatrixPtr& matrix, size_t height, size_t width, bool trans, bool useGpu) {
  if (!matrix) {
    matrix = Matrix::create(height, width, trans, useGpu);
  } else {
    CHECK_EQ(matrix->useGpu(), useGpu);
    matrix->resize(height, width);
  }
}

void Matrix::resizeOrCreateSparseMatrix(MatrixPtr& matrix,
                                        size_t height,
                                        size_t width,
                                        size_t nnz,
                                        SparseValueType valueType,
                                        SparseFormat format,
                                        bool trans,
                                        bool useGpu) {
  if (!matrix) {
    matrix = Matrix::createSparseMatrix(
        height, width, nnz, valueType, format, trans, useGpu);
  } else {
    CHECK(dynamic_cast<CpuSparseMatrix*>(matrix.get()) ||
          dynamic_cast<GpuSparseMatrix*>(matrix.get()));
    CHECK_EQ(matrix->useGpu(), useGpu);
    matrix->resize(height, width, nnz, valueType, format);
  }
}

void Matrix::reshape(size_t height, size_t width) {
  CHECK(isContiguous());
  CHECK(height_ * width_ == height * width);
  height_ = height;
  width_ = width;
  stride_ = width_;
}

MatrixPtr Matrix::subMatrix(size_t startRow,
                            size_t endRow,
                            size_t startCol,
                            size_t endCol) {
  CHECK_LE(startRow, endRow);
  CHECK_LE(endRow, getHeight());
  CHECK_LE(startCol, endCol);
  CHECK_LE(endCol, getWidth());

  return Matrix::create(getData() + startRow * getStride() + startCol,
                        endRow - startRow,
                        endCol - startCol,
                        getStride(),
                        trans_,
                        useGpu_);
}

void Matrix::setDiag(real value) {
  CHECK(data_ != NULL);
  CHECK_EQ(height_, width_);

  zeroMem();
  BaseMatrix diag(height_, 1, stride_ + 1, data_, false, useGpu_);
  diag.assign(value);
}

GpuMatrix::GpuMatrix(size_t height, size_t width, bool trans)
    : Matrix(std::make_shared<GpuMemoryHandle>(height * width * sizeof(real)),
             height,
             width,
             trans,
             true) {}

GpuMatrix::~GpuMatrix() {}

void GpuMatrix::zeroMem() {
  CHECK(data_ != NULL);
  zero();
}

void GpuMatrix::resetOne() {
  CHECK(data_ != NULL);
  one();
}

void GpuMatrix::resize(size_t newHeight, size_t newWidth) {
  size_t newSize = newHeight * newWidth;
  if (NULL == memoryHandle_.get() ||
      newSize * sizeof(real) > memoryHandle_->getAllocSize()) {
    memoryHandle_ = std::make_shared<GpuMemoryHandle>(newSize * sizeof(real));
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
  }
  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newSize;
  stride_ = width_;
}

real GpuMatrix::getElement(size_t x, size_t y) const {
  real elem = 0;
  hl_memcpy_device2host(&elem, &data_[x * stride_ + y], sizeof(real));
  return elem;
}

real GpuMatrix::getSum() {
  CHECK(isContiguous());
  real sum = 0.0f;
  hl_vector_sum(data_, &sum, height_ * width_);
  return sum;
}

real GpuMatrix::getMin() {
  CHECK(isContiguous());
  auto vec = GpuVector(height_ * width_, data_);
  return vec.getMin();
}

real GpuMatrix::getMax() {
  CHECK(isContiguous());
  auto vec = GpuVector(height_ * width_, data_);
  return vec.getMax();
}

void GpuMatrix::accumulateColSum(Matrix& src) {
  CHECK_EQ(getWidth(), src.getWidth());
  CHECK_EQ(getHeight(), (size_t)1);
  sumCols(src, 1.0, 1.0);
}

real GpuMatrix::getAbsSum() {
  CHECK(isContiguous());
  real sum = 0.0f;
  hl_vector_abs_sum(data_, &sum, height_ * width_);
  return sum;
}

void GpuMatrix::copyFrom(const Matrix& src) {
  CHECK(isContiguous());
  CHECK(src.isContiguous());
  CHECK(elementCnt_ == src.getElementCnt());

  if (typeid(src) == typeid(CpuMatrix)) {
    hl_memcpy_host2device(
        data_, const_cast<real*>(src.getData()), sizeof(real) * elementCnt_);
  } else if (typeid(src) == typeid(GpuMatrix)) {
    hl_memcpy_device2device(
        data_, const_cast<real*>(src.getData()), sizeof(real) * elementCnt_);
  } else {
    LOG(FATAL) << "Wrong";
  }
}

void GpuMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  CHECK(isContiguous());
  CHECK(src.isContiguous());
  CHECK(elementCnt_ == src.getElementCnt());
  hl_memcpy_async(this->getData(),
                  const_cast<real*>(src.getData()),
                  sizeof(real) * elementCnt_,
                  stream);
}

void GpuMatrix::copyFrom(const real* hostSrc, size_t size) {
  CHECK(isContiguous());
  CHECK(size <= elementCnt_);
  hl_memcpy_host2device(data_, const_cast<real*>(hostSrc), sizeof(real) * size);
}

void GpuMatrix::copyFrom(const real* hostSrc, const int64_t* seq) {
  LOG(FATAL) << "not implemented";
}

void GpuMatrix::copyFrom(const IVector& src) {
  CHECK(isContiguous());
  CpuMatrix matrix(src.getSize(), 1, false);
  matrix.copyFrom(src);
  copyFrom(matrix);
}

void GpuMatrix::copyByRowIndex(Matrix& b, const IVector& rowIndex) {
  size_t height = getHeight();
  size_t width = getWidth();
  CHECK_EQ(b.getWidth(), width);
  real* dst = getData();
  real* src = b.getData();
  const int* index = rowIndex.getData();
  hl_sequence2batch_copy(dst, src, index, width, height, true);
}

MatrixPtr GpuMatrix::clone(size_t height, size_t width, bool useGpu) {
  CHECK(isContiguous());

  if (height == 0 && width == 0) {
    height = height_;
    width = width_;
  }

  CHECK(width && height);

  if (useGpu) {
    return std::make_shared<GpuMatrix>(height, width);
  } else {
    return std::make_shared<CpuMatrix>(height, width);
  }
}

MatrixPtr GpuMatrix::getTranspose() {
  if (memoryHandle_.get() != NULL) {
    MatrixPtr copy_T(
        new GpuMatrix(std::dynamic_pointer_cast<GpuMemoryHandle>(memoryHandle_),
                      height_,
                      width_,
                      true));
    return copy_T;
  } else {
    MatrixPtr copy_T(new GpuMatrix(data_, height_, width_, true));
    return copy_T;
  }
}

void GpuMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  if (memAlloc) {
    matTrans = std::make_shared<GpuMatrix>(width_, height_);
  } else {
    CHECK(matTrans != NULL);
    CHECK_EQ(matTrans->getHeight(), width_);
    CHECK_EQ(matTrans->getWidth(), height_);
  }
  real* dataTrans = matTrans->getData();
  real* data = getData();
  int lda = getStride();
  int ldc = matTrans->getStride();

  hl_matrix_transpose(data, dataTrans, height_, width_, lda, ldc);
}

void GpuMatrix::rotate(MatrixPtr& matRot, bool memAlloc, bool clockWise) {
  if (memAlloc) {
    matRot = std::make_shared<GpuMatrix>(width_, height_);
  } else {
    CHECK(matRot != NULL);
    CHECK_EQ(matRot->getHeight(), width_);
    CHECK_EQ(matRot->getWidth(), height_);
  }

  real* dataRot = matRot->getData();
  real* data = getData();
  hl_matrix_rotate(data, dataRot, height_, width_, clockWise);
}

MatrixPtr GpuMatrix::getInverse() {
  MatrixPtr matInv;
  inverse(matInv, true);
  return matInv;
}

void GpuMatrix::inverse(MatrixPtr& matInv, bool memAlloc) {
  CHECK_EQ(height_, width_);

  if (memAlloc) {
    matInv = std::make_shared<GpuMatrix>(height_, width_);
  } else {
    CHECK(matInv != NULL);
  }

  real* data = getData();
  real* dataInv = matInv->getData();
  int lda = getStride();
  int ldc = matInv->getStride();

  hl_matrix_inverse(data, dataInv, height_, lda, ldc);
}

void GpuMatrix::addBias(Matrix& b, real scale) {
  CHECK(b.getHeight() == 1) << "the Bias should be a vector";
  BaseMatrix::addBias(b, scale);
}

void GpuMatrix::addSharedBias(Matrix& b, real scale) {
  CHECK(b.getHeight() == 1) << "the Bias should be a vector";
  CHECK_LE(b.getWidth(), getWidth());
  CHECK_EQ(getWidth() % b.getWidth(), 0UL);
  hl_matrix_add_shared_bias(
      getData(), b.getData(), b.getWidth(), getHeight(), getWidth(), scale);
}

void GpuMatrix::collectBias(Matrix& a, real scale) {
  CHECK_EQ(getHeight(), (size_t)1);
  CHECK_EQ(width_, a.getWidth());
  GpuSparseMatrix* sMatPtr = dynamic_cast<GpuSparseMatrix*>(&a);
  if (!sMatPtr) {
    sumCols(a, /* scaleSum= */ scale, /* scaleDest= */ 1);
  } else {
    real* data = getData();
    hl_sparse_matrix_s A_d = sMatPtr->sMatrix_.get();
    hl_sparse_matrix_column_sum(data, A_d, sMatPtr->getHeight(), width_, scale);
  }
}

void GpuMatrix::collectSharedBias(Matrix& a, real scale) {
  CHECK_EQ(getHeight(), (size_t)1);
  CHECK_EQ(a.getWidth() % getWidth(), 0UL);
  hl_matrix_collect_shared_bias(
      getData(), a.getData(), getWidth(), a.getHeight(), a.getWidth(), scale);
}

void GpuMatrix::sequenceAvgForward(Matrix& a,
                                   const IVector& startsPos,
                                   int mode) {
  size_t height = getHeight();
  size_t width = getWidth();
  CHECK_EQ(height, startsPos.getSize() - 1);
  CHECK_EQ(width, a.getWidth());
  real* dst = getData();
  real* src = a.getData();
  const int* starts = startsPos.getData();

  hl_sequence_avg_forward(dst, src, starts, height, width, mode);
}

void GpuMatrix::sequenceAvgBackward(Matrix& a,
                                    const IVector& startsPos,
                                    int mode) {
  size_t height = a.getHeight();
  size_t width = getWidth();
  CHECK_EQ(height, startsPos.getSize() - 1);
  CHECK_EQ(width, a.getWidth());
  real* dst = getData();
  real* src = a.getData();
  const int* starts = startsPos.getData();

  hl_sequence_avg_backward(dst, src, starts, height, width, mode);
}

/* this = scaleAB*(a*b) +  scaleT*this */
void GpuMatrix::mul(const GpuMatrix& a,
                    const GpuMatrix& b,
                    real scaleAB,
                    real scaleT) {
  CHECK(!isTransposed()) << "Not supported";

  if (!a.isTransposed() && !b.isTransposed()) {
    CHECK_EQ(width_, b.width_);
    CHECK_EQ(height_, a.height_);
    CHECK_EQ(a.width_, b.height_);
  } else if (a.isTransposed() && !b.isTransposed()) {
    CHECK_EQ(width_, b.width_);
    CHECK_EQ(height_, a.width_);
    CHECK_EQ(a.height_, b.height_);
  } else if (!a.isTransposed() && b.isTransposed()) {
    CHECK_EQ(width_, b.height_);
    CHECK_EQ(height_, a.height_);
    CHECK_EQ(a.width_, b.width_);
  } else {
    LOG(FATAL) << "Is not supported";
  }

  real* A_d = a.data_;
  real* B_d = b.data_;
  real* C_d = data_;
  int dimM = getHeight();
  int dimN = getWidth();
  int dimK = !a.isTransposed() ? a.width_ : a.height_;
  int lda = a.getStride();
  int ldb = b.getStride();
  int ldc = getStride();
  hl_trans_op_t transa = !a.isTransposed() ? HPPL_OP_N : HPPL_OP_T;
  hl_trans_op_t transb = !b.isTransposed() ? HPPL_OP_N : HPPL_OP_T;

  hl_matrix_mul(A_d,
                transa,
                B_d,
                transb,
                C_d,
                dimM,
                dimN,
                dimK,
                scaleAB,
                scaleT,
                lda,
                ldb,
                ldc);
}

void GpuMatrix::mul(const GpuSparseMatrix& a,
                    const GpuMatrix& b,
                    real scaleAB,
                    real scaleT) {
  CHECK(isContiguous());
  CHECK(b.isContiguous());
  CHECK(b.useGpu_ == true) << "Matrix type are not equal";
  CHECK(!trans_ && !b.trans_) << "not supported";

  if (!a.trans_) {
    CHECK(width_ == b.width_ && height_ == a.height_ && a.width_ == b.height_)
        << "Matrix dimensions are not equal";
  } else {
    CHECK(width_ == b.width_ && height_ == a.width_ && a.height_ == b.height_)
        << "Matrix dimensions are not equal";
  }
  hl_trans_op_t transA = a.trans_ ? HPPL_OP_T : HPPL_OP_N;
  hl_sparse_matrix_s A_d = a.sMatrix_.get();
  real* B_d = b.data_;
  real* C_d = data_;
  hl_matrix_csr_mul_dense(A_d,
                          transA,
                          B_d,
                          HPPL_OP_N,
                          C_d,
                          height_,
                          width_,
                          b.height_,
                          scaleAB,
                          scaleT);
}

void GpuMatrix::mul(const GpuMatrix& a,
                    const GpuSparseMatrix& b,
                    real scaleAB,
                    real scaleT) {
  CHECK(isContiguous());
  CHECK(a.isContiguous());
  CHECK(a.useGpu_ == true) << "Matrix type are not equal";

  hl_sparse_matrix_s B_d = b.sMatrix_.get();
  real* A_d = a.data_;
  real* C_d = data_;
  hl_trans_op_t transB = b.trans_ ? HPPL_OP_T : HPPL_OP_N;
  if (!b.trans_) {
    CHECK(width_ == b.width_ && height_ == a.height_ && a.width_ == b.height_)
        << "Matrix dimensions are not equal";
  } else {
    CHECK(width_ == b.height_ && height_ == a.height_ && a.width_ == b.width_)
        << "Matrix dimensions are not equal";
  }
  if (b.format_ == SPARSE_CSC) {
    hl_matrix_dense_mul_csc(A_d,
                            HPPL_OP_N,
                            B_d,
                            transB,
                            C_d,
                            height_,
                            width_,
                            a.width_,
                            scaleAB,
                            scaleT);
  } else {
    hl_matrix_dense_mul_csr(A_d,
                            HPPL_OP_N,
                            B_d,
                            transB,
                            C_d,
                            height_,
                            width_,
                            a.width_,
                            scaleAB,
                            scaleT);
  }
}

/* this = a*b */
void GpuMatrix::mul(const Matrix& a, const Matrix& b) { mul(a, b, 1.0, 0.0); }

void GpuMatrix::mul(const Matrix& a,
                    const Matrix& b,
                    real scaleAB,
                    real scaleT) {
  const auto a_ptr = dynamic_cast<const GpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const GpuMatrix*>(&b);
  const auto a_ptr_s = dynamic_cast<const GpuSparseMatrix*>(&a);
  const auto b_ptr_s = dynamic_cast<const GpuSparseMatrix*>(&b);

  if (a_ptr && b_ptr) {
    mul(*a_ptr, *b_ptr, scaleAB, scaleT);
  } else if (a_ptr_s && b_ptr) {
    mul(*a_ptr_s, *b_ptr, scaleAB, scaleT);
  } else if (a_ptr && b_ptr_s) {
    mul(*a_ptr, *b_ptr_s, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "Not supported";
  }
}

/* this = this* b */
void GpuMatrix::rightMul(Matrix& b) { rightMul(b, 1.0, 0.0); }

/* this = scaleAB*(this*b) +  scaleT*this */
void GpuMatrix::rightMul(Matrix& b, real scaleAB, real scaleT) {
  CHECK(dynamic_cast<GpuMatrix*>(&b));
  CHECK(!isTransposed()) << "Not supported";
  CHECK(!b.isTransposed()) << "Not supported";
  mul(*this, *dynamic_cast<GpuMatrix*>(&b), scaleAB, scaleT);
}

/* this = a*this */
void GpuMatrix::leftMul(Matrix& a) { leftMul(a, 1.0, 0.0); }

/* this = scaleAB*(a*this) +  scaleT*this */
void GpuMatrix::leftMul(Matrix& a, real scaleAB, real scaleT) {
  CHECK(dynamic_cast<GpuMatrix*>(&a));
  CHECK(!isTransposed()) << "Not supported";
  CHECK(!a.isTransposed()) << "Not supported";
  mul(*dynamic_cast<GpuMatrix*>(&a), *this, scaleAB, scaleT);
}

void GpuMatrix::selectRows(Matrix& table, IVector& ids) {
#ifndef PADDLE_ONLY_CPU
  CHECK(dynamic_cast<GpuMatrix*>(&table));
  CHECK(table.useGpu());
  CHECK(ids.useGpu());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), table.getWidth());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  real* a = getData();
  size_t tableSize = table.getHeight();
  int* index = ids.getData();

  hl_matrix_select_rows(a,
                        stride_,
                        table.getData(),
                        table.stride_,
                        index,
                        numSamples,
                        tableSize,
                        dim);
#endif
}

void GpuMatrix::addToRows(Matrix& table, IVector& ids) {
#ifndef PADDLE_ONLY_CPU
  CHECK(dynamic_cast<GpuMatrix*>(&table));
  CHECK(table.useGpu());
  CHECK(ids.useGpu());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), table.getWidth());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  real* a = getData();
  size_t tableSize = table.getHeight();
  int* index = ids.getData();

  hl_matrix_add_to_rows(table.getData(),
                        table.stride_,
                        a,
                        stride_,
                        index,
                        numSamples,
                        tableSize,
                        dim);
#endif
}

void GpuMatrix::colMerge(Matrix& src) {
  CHECK(src.height_ == height_);
  if (!trans_ && !src.trans_) {
    sumRows(src, /* scaleSum= */ 1, /* scaleDest= */ 0);
  } else {
    LOG(FATAL) << "Is not supported";
  }
}

void GpuMatrix::rowSum(Matrix& sum) {
  CHECK_EQ(sum.getHeight(), getHeight());
  CHECK_EQ(sum.getWidth(), (size_t)1);

  sum.sumRows(*this, /* scaleSum= */ 1, /* scaleDest= */ 0);
}

void GpuMatrix::rowMax(Matrix& max) {
  CHECK_EQ(max.getHeight(), getHeight());
  CHECK_EQ(max.getWidth(), (size_t)1);

  max.maxRows(*this);
}

void GpuMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
#ifndef PADDLE_ONLY_CPU
  CHECK(maxIds.useGpu() && maxVal.useGpu()) << "Matrix type are not equal";
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  CHECK_EQ(maxVal.getWidth(), beam);

  hl_matrix_top_k(maxVal.getData(),
                  maxVal.getStride(),
                  maxIds.getData(),
                  this->getData(),
                  this->getStride(),
                  this->getWidth(),
                  beam,
                  numSamples);
#endif
}

void GpuMatrix::colMax(Matrix& max) {
  CHECK_EQ(max.getWidth(), getWidth());
  CHECK_EQ(max.getHeight(), (size_t)1);

  max.maxCols(*this);
}

void GpuMatrix::colMax(IVector& maxIds, Matrix& maxVal) {
  LOG(FATAL) << "Is not supported";
}

void GpuMatrix::maxoutForward(Matrix& a,
                              IVector& id,
                              size_t channels,
                              size_t groups) {
  CHECK(dynamic_cast<GpuMatrix*>(&a));
  CHECK(dynamic_cast<GpuIVector*>(&id));
  CHECK_EQ(a.getHeight(), getHeight());

  size_t size = getWidth();
  size_t batchSize = getHeight();
  const real* input = a.getData();
  real* output = getData();
  int* idForGpu = id.getData();

  hl_maxout_forward(
      input, output, idForGpu, batchSize, size, size / channels, groups);
}

void GpuMatrix::maxoutBackward(Matrix& a,
                               IVector& id,
                               size_t channels,
                               size_t groups) {
  CHECK(dynamic_cast<GpuMatrix*>(&a));
  CHECK(dynamic_cast<GpuIVector*>(&id));
  CHECK_EQ(a.getHeight(), getHeight());

  size_t size = a.getWidth();
  size_t batchSize = getHeight();
  real* input = getData();
  const real* output = a.getData();
  const int* idForGpu = id.getData();

  hl_maxout_backward(
      input, output, idForGpu, batchSize, size, size / channels, groups);
}

/*calulate the error of classification */
void GpuMatrix::classificationError(Matrix& output,
                                    IVector& label,
                                    size_t topkSize) {
  auto gpuOutput = dynamic_cast<GpuMatrix*>(&output);
  auto gpuLabel = dynamic_cast<GpuIVector*>(&label);
  size_t numSamples = this->getHeight();
  GpuMatrixPtr gpuTopVal = std::make_shared<GpuMatrix>(numSamples, topkSize);
  GpuIVectorPtr gpuTopIds = std::make_shared<GpuIVector>(numSamples * topkSize);

  CHECK(gpuOutput && gpuLabel) << "Invalid argument pointer";
  CHECK(gpuTopVal && gpuTopIds) << "Allocate GPU memory failed";
  CHECK(gpuLabel->getSize() == numSamples) << "Vector size is not equal";
  CHECK(numSamples == gpuOutput->getHeight() && this->getWidth() == 1)
      << "Matrix dimensions are not equal";

  size_t dim = gpuOutput->getWidth();
  hl_matrix_classification_error(gpuTopVal->getData(),
                                 gpuTopVal->getStride(),
                                 gpuTopIds->getData(),
                                 gpuOutput->getData(),
                                 gpuOutput->getStride(),
                                 dim,
                                 topkSize,
                                 numSamples,
                                 gpuLabel->getData(),
                                 this->getData());
}

/* copy -log(output[i * width + label]) to this->data[i] */
void GpuMatrix::oneHotCrossEntropy(Matrix& output, IVector& label) {
  GpuMatrix* output_ptr = dynamic_cast<GpuMatrix*>(&output);
  GpuIVector* label_ptr = dynamic_cast<GpuIVector*>(&label);

  CHECK(output_ptr && label_ptr) << "Invalid argument pointer";

  CHECK(height_ == label.getSize() && width_ == 1 && height_ == output.height_)
      << "Matrix dimensions are not equal";

  real* A_d = output_ptr->data_;
  real* C_d = data_;
  int* label_d = label_ptr->getData();

  hl_matrix_cross_entropy(A_d, C_d, label_d, height_, output.width_);
}

/* calculate the error of outputV according to label */
void GpuMatrix::oneHotCrossEntropyBp(Matrix& outputV, IVector& label) {
  GpuMatrix* output_ptr = dynamic_cast<GpuMatrix*>(&outputV);
  GpuIVector* label_ptr = dynamic_cast<GpuIVector*>(&label);

  CHECK(output_ptr && label_ptr) << "Invalid argument pointer";

  CHECK(height_ == output_ptr->height_ && width_ == output_ptr->width_)
      << "Matrix dimensions are not equal";

  real* output_d = output_ptr->data_;
  real* grad_d = data_;
  int* label_d = label_ptr->getData();

  hl_matrix_cross_entropy_bp(grad_d, output_d, label_d, height_, width_);
}

void GpuMatrix::oneHotCrossEntropyWithSelfNorm(Matrix& output,
                                               IVector& label,
                                               real alpha) {
  LOG(FATAL) << "Not implemented";
}

void GpuMatrix::oneHotCrossEntropyWithSelfNormBp(Matrix& outputV,
                                                 IVector& label,
                                                 real alpha) {
  LOG(FATAL) << "Not implemented";
}

void GpuMatrix::softmax(Matrix& output) {
  CHECK(output.useGpu()) << "Matrix type are not equal";

  size_t height = getHeight();
  size_t width = getWidth();
  CHECK(height == output.getHeight() && width == output.getWidth())
      << "Matrix dimensions are not equal";

  real* inputData = getData();
  real* outputData = output.getData();
  hl_matrix_softmax(inputData, outputData, height, width);
}

void GpuMatrix::sequenceSoftmax(Matrix& output, const IVector& index) {
  CHECK_EQ(getWidth(), 1UL);
  CHECK_EQ(output.getWidth(), 1UL);
  CHECK(isContiguous());

  real* inputData = getData();
  real* outputData = output.getData();
  auto starts = index.getData();
  int numSequences = index.getSize() - 1;
  hl_sequence_softmax_forward(inputData, outputData, starts, numSequences);
}

void GpuMatrix::softmaxDerivative(Matrix& output, Matrix& sftmaxSum) {
  CHECK(output.useGpu_ == true && sftmaxSum.useGpu_ == true)
      << "Matrix type are not equal";

  CHECK(height_ == output.height_ && width_ == output.width_ &&
        height_ == sftmaxSum.height_)
      << "Matrix dimensions are not equal";

  real* output_d = output.data_;
  real* sftmaxSum_d = sftmaxSum.data_;
  real* grad_d = data_;
  hl_matrix_softmax_derivative(grad_d, output_d, sftmaxSum_d, height_, width_);
}

void GpuMatrix::softmaxBackward(Matrix& outputV) {
  CHECK(outputV.useGpu()) << "Matrix type are not equal";

  size_t height = getHeight();
  size_t width = getWidth();
  CHECK(height == outputV.getHeight() && width == outputV.getWidth())
      << "Matrix dimensions are not equal";

  real* output_grad = getData();
  real* output_value = outputV.getData();
  hl_softmax_backward(output_value, output_grad, height, width);
}

void GpuMatrix::sumOfSquares(Matrix& output, Matrix& label) {
  CHECK_EQ(label.getHeight(), height_);
  CHECK_EQ(output.getHeight(), height_);
  CHECK_EQ(label.getWidth(), output.getWidth());
  CHECK_EQ((size_t)1, width_);

  auto labelptr = dynamic_cast<GpuSparseMatrix*>(&label);
  if (labelptr) {
    LOG(FATAL) << "not supported: GpuSparseMatrix as label";
  }

  BaseMatrix::sumOfSquaredDiffs(output,
                                label,
                                /* scaleSum= */ 1,
                                /* scaleDest= */ 1);
}

void GpuMatrix::sumOfSquaresBp(Matrix& outputV, Matrix& label) {
  add2(outputV, label, 1, 2, -2);
}

void GpuMatrix::tanh(Matrix& output) { BaseMatrix::tanh(output); }

void GpuMatrix::tanhDerivative(Matrix& output) {
  BaseMatrix::tanhDerivative(output);
}

void GpuMatrix::softrelu(Matrix& output) { BaseMatrix::softrelu(output); }

void GpuMatrix::softreluDerivative(Matrix& output) {
  BaseMatrix::softreluDerivative(output);
}

void GpuMatrix::scaledTanh(Matrix& output, real p1, real p2) {
  BaseMatrix::scaledTanh(output, p1, p2);
}

void GpuMatrix::randomizeUniform() {
  CHECK(isContiguous());
  real* data = data_;
  size_t size = height_ * width_;

  hl_rand(data, size);
}

void GpuMatrix::print(std::ostream& os) const {
  CHECK(isContiguous());
  CpuMatrix cpuMat(getHeight(), getWidth());
  cpuMat.copyFrom(*this);
  cpuMat.print(os);
}

void GpuMatrix::print(std::ostream& os, size_t height, size_t width) const {
  CHECK(isContiguous());
  CpuMatrix cpuMat(getHeight(), getWidth());
  cpuMat.copyFrom(*this);
  cpuMat.print(os, height, width);
}

void GpuMatrix::check(std::ostream& os, Matrix& refMat, bool printDiff) {
  CHECK(isContiguous());
  CHECK(height_ == refMat.getHeight());
  CHECK(width_ == refMat.getWidth());
  CpuMatrix cpuRef(height_, width_);
  GpuMatrix gpuRef(height_, width_);
  cpuRef.copyFrom(refMat);
  gpuRef.copyFrom(*this);
  size_t diffCnt = 0;
  for (size_t i = 0; i < height_; ++i) {
    for (size_t j = 0; j < width_; ++j) {
      real a = gpuRef.getElement(i, j);
      real b = cpuRef.getElement(i, j);
      if (fabs(a - b) > 0.00001) {
        ++diffCnt;
        if (printDiff) {
          os << "ref= " << a << "  check= " << b << std::endl;
        }
      }
    }
  }
  LOG(INFO) << "the  diffCnt is " << diffCnt;
}

void GpuMatrix::convExpand(Matrix& feature,
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
  CHECK(feature.useGpu_ == true) << "Matrix type are not equal";

  CHECK_EQ(size_t(feaImgHeight * feaImgWidth * channels),
           feature.getHeight() * feature.getWidth())
      << "Matrix dimensions are not equal";

  size_t elemCnt = outputH * outputW * blockH * blockW * channels;
  CHECK_EQ(elemCnt, height_ * width_) << "Matrix dimensions are not equal";

  hl_expand_feature2col(feature.getData(),
                        channels,
                        feaImgHeight,
                        feaImgWidth,
                        blockH,
                        blockW,
                        strideH,
                        strideW,
                        paddingH,
                        paddingW,
                        outputH,
                        outputW,
                        getData());
}

void GpuMatrix::convShrink(Matrix& expandFeat,
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
                           real alpha,
                           real beta) {
  CHECK(expandFeat.useGpu_ == true) << "Matrix type are not equal";
  CHECK_EQ(size_t(thisImgHeight * thisImgWidth * channels),
           getHeight() * getWidth())
      << "Matrix dimensions are not equal";

  size_t elemCnt = outputH * outputW * blockW * blockH * channels;
  CHECK(elemCnt == expandFeat.getHeight() * expandFeat.getWidth())
      << "Matrix dimensions are not equal";
  hl_shrink_col2feature(expandFeat.getData(),
                        channels,
                        thisImgHeight,
                        thisImgWidth,
                        blockH,
                        blockW,
                        strideH,
                        strideW,
                        paddingH,
                        paddingW,
                        outputH,
                        outputW,
                        getData(),
                        alpha,
                        beta);
}

void GpuMatrix::maxPoolForward(Matrix& inputMat,
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
  CHECK(inputMat.useGpu_ == true) << "Matrix type are not equal";

  real* inputData = inputMat.getData();
  size_t frameNum = inputMat.getHeight();
  size_t width = imgSizeW;
  size_t height = imgSizeH;
  CHECK(height * width * channels == inputMat.getWidth());
  CHECK(height_ == inputMat.getHeight());
  CHECK(width_ == outputH * outputW * channels);

  hl_maxpool_forward(frameNum,
                     inputData,
                     channels,
                     height,
                     width,
                     outputH,
                     outputW,
                     sizeX,
                     sizeY,
                     strideH,
                     strideW,
                     paddingH,
                     paddingW,
                     data_,
                     getStride());
}

void GpuMatrix::maxPoolBackward(Matrix& inputMat,
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
  CHECK(inputMat.useGpu_ == true && outGrad.useGpu_ == true &&
        outV.useGpu_ == true)
      << "Matrix type are not equal";

  real* inputData = inputMat.getData();
  real* outData = outV.getData();
  real* outDiff = outGrad.getData();
  size_t frameNum = inputMat.getHeight();
  size_t channels = outV.getWidth() / outputH / outputW;
  size_t width = imgSizeW;
  size_t height = imgSizeH;
  CHECK(height * width * channels == inputMat.getWidth());
  CHECK(height_ == inputMat.getHeight());
  CHECK(width_ == width * height * channels);
  CHECK(outGrad.getHeight() == outV.getHeight() &&
        outGrad.getWidth() == outV.getWidth());

  hl_maxpool_backward(frameNum,
                      inputData,
                      outData,
                      outDiff,
                      channels,
                      height,
                      width,
                      outputH,
                      outputW,
                      sizeX,
                      sizeY,
                      strideH,
                      strideW,
                      paddingH,
                      paddingW,
                      scaleTargets,
                      scaleOutput,
                      data_,
                      outGrad.getStride());
}

void GpuMatrix::avgPoolForward(Matrix& inputMat,
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
  CHECK(inputMat.useGpu_ == true) << "Matrix type are not equal";

  real* inputData = inputMat.getData();
  size_t frameNum = inputMat.getHeight();
  size_t height = imgSizeH;
  size_t width = imgSizeW;
  CHECK(height * width * channels == inputMat.getWidth());
  CHECK(height_ == inputMat.getHeight());
  CHECK(width_ == outputH * outputW * channels);

  hl_avgpool_forward(frameNum,
                     inputData,
                     channels,
                     height,
                     width,
                     outputH,
                     outputW,
                     sizeX,
                     sizeY,
                     strideH,
                     strideW,
                     paddingH,
                     paddingW,
                     data_,
                     getStride());
}

void GpuMatrix::avgPoolBackward(Matrix& outGrad,
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
  CHECK(outGrad.useGpu_ == true) << "Matrix type are not equal";

  real* outDiff = outGrad.getData();
  size_t frameNum = outGrad.getHeight();
  size_t channels = outGrad.getWidth() / outputH / outputW;
  size_t height = imgSizeH;
  size_t width = imgSizeW;
  CHECK(height * width * channels == width_);
  CHECK(height_ == outGrad.getHeight());
  CHECK(outGrad.getWidth() == outputH * outputW * channels);

  hl_avgpool_backward(frameNum,
                      outDiff,
                      channels,
                      height,
                      width,
                      outputH,
                      outputW,
                      sizeX,
                      sizeY,
                      strideH,
                      strideW,
                      paddingH,
                      paddingW,
                      scaleTargets,
                      scaleOutput,
                      data_,
                      outGrad.getStride());
}

void GpuMatrix::maxSequenceForward(Matrix& input,
                                   const IVector& sequence,
                                   IVector& index) {
  CHECK(dynamic_cast<GpuMatrix*>(&input));
  CHECK(dynamic_cast<const GpuIVector*>(&sequence));
  CHECK(dynamic_cast<GpuIVector*>(&index));

  real* outData = getData();
  real* inputData = input.getData();
  const int* starts = sequence.getData();
  int* maxIndex = index.getData();
  size_t numSequences = getHeight();
  size_t dim = getWidth();

  CHECK_EQ(dim, input.getWidth());
  CHECK_EQ(numSequences, sequence.getSize() - 1);
  CHECK_EQ(numSequences * dim, index.getSize());

  hl_max_sequence_forward(
      inputData, starts, outData, maxIndex, numSequences, dim);
}

void GpuMatrix::maxSequenceBackward(Matrix& outputGrad,
                                    const IVector& sequence,
                                    IVector& index) {
  CHECK(dynamic_cast<GpuMatrix*>(&outputGrad));
  CHECK(dynamic_cast<const GpuIVector*>(&sequence));
  CHECK(dynamic_cast<GpuIVector*>(&index));

  real* inputGrad = getData();
  real* outGrad = outputGrad.getData();
  int* maxIndex = index.getData();
  size_t dim = getWidth();
  size_t numSequences = sequence.getSize() - 1;

  CHECK_EQ(dim, outputGrad.getWidth());
  CHECK_EQ(numSequences, outputGrad.getHeight());
  CHECK_EQ(numSequences * dim, index.getSize());

  hl_max_sequence_backward(outGrad, maxIndex, inputGrad, numSequences, dim);
}

void GpuMatrix::paramReluForward(Matrix& data, Matrix& W) {
  CHECK(data.useGpu_ == true && W.useGpu_ == true)
      << "Matrix type are not equal";
  real* input = data.getData();
  real* w = W.getData();
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = W.getHeight() * W.getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  real* output = getData();
  hl_param_relu_forward(output, input, w, numElements, numSamples, partial_sum);
}

void GpuMatrix::paramReluBackwardW(Matrix& oGrad, Matrix& data) {
  CHECK(oGrad.useGpu_ == true && data.useGpu_ == true)
      << "Matrix type are not equal";
  real* ograd = oGrad.getData();
  real* input = data.getData();
  real* wgrad = data_;
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = this->getHeight() * this->getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  hl_param_relu_backward_w(
      wgrad, ograd, input, numElements, numSamples, partial_sum);
}

void GpuMatrix::paramReluBackwardDiff(Matrix& oGrad, Matrix& data, Matrix& W) {
  real* diff = data_;
  real* input = data.getData();
  real* ograd = oGrad.getData();
  real* w = W.getData();
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = W.getHeight() * W.getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  hl_param_relu_backward_diff(
      ograd, input, w, diff, numElements, numSamples, partial_sum);
}

void GpuMatrix::addColumnVector(const Matrix& b) {
  BaseMatrix::addColVector(const_cast<Matrix&>(b));
}

void GpuMatrix::bilinearForward(const Matrix& in,
                                const size_t inImgH,
                                const size_t inImgW,
                                const size_t outImgH,
                                const size_t outImgW,
                                const size_t numChannels,
                                const real ratioH,
                                const real ratioW) {
  CHECK(dynamic_cast<const GpuMatrix*>(&in));

  const size_t outputW = getWidth();
  const size_t outputH = getHeight();
  const size_t inputW = in.getWidth();
  const size_t inputH = in.getHeight();

  real* outData = getData();
  const real* inData = in.getData();

  if (inImgH == outImgW && inImgW == outImgW) {
    this->copyFrom(in);
  } else {
    hl_bilinear_forward(inData,
                        inImgH,
                        inImgW,
                        inputH,
                        inputW,
                        outData,
                        outImgH,
                        outImgW,
                        outputH,
                        outputW,
                        numChannels,
                        ratioH,
                        ratioW);
  }
}

void GpuMatrix::bilinearBackward(const Matrix& out,
                                 const size_t outImgH,
                                 const size_t outImgW,
                                 const size_t inImgH,
                                 const size_t inImgW,
                                 const size_t numChannels,
                                 const real ratioH,
                                 const real ratioW) {
  CHECK(dynamic_cast<const GpuMatrix*>(&out));

  const size_t inputW = getWidth();
  const size_t inputH = getHeight();
  const size_t outputW = out.getWidth();
  const size_t outputH = out.getHeight();

  real* inGrad = getData();
  const real* outGrad = out.getData();

  if (outImgH == inImgH && outImgW == inImgW) {
    this->add(const_cast<Matrix&>(out));
  } else {
    hl_bilinear_backward(inGrad,
                         inImgH,
                         inImgW,
                         inputH,
                         inputW,
                         outGrad,
                         outImgH,
                         outImgW,
                         outputH,
                         outputW,
                         numChannels,
                         ratioH,
                         ratioW);
  }
}

void GpuMatrix::multiBinaryLabelCrossEntropy(Matrix& output, Matrix& label) {
  GpuMatrix* outputPtr = dynamic_cast<GpuMatrix*>(&output);
  auto labelPtr = dynamic_cast<GpuSparseMatrix*>(&label);

  CHECK(outputPtr && labelPtr) << "Invalid argument pointer";
  CHECK(labelPtr->format_ == SPARSE_CSR) << "Matrix format not supported";
  CHECK(height_ == outputPtr->height_ && width_ == 1 &&
        outputPtr->width_ == labelPtr->getWidth() &&
        outputPtr->height_ == labelPtr->getHeight())
      << "Matrix dimensions are not equal";

  real* output_d = outputPtr->data_;
  real* entropy_d = data_;
  hl_sparse_matrix_s mat_d = labelPtr->sMatrix_.get();
  hl_matrix_multi_binary_cross_entropy(
      output_d, entropy_d, mat_d, height_, outputPtr->width_);
}

void GpuMatrix::multiBinaryLabelCrossEntropyBp(Matrix& output, Matrix& label) {
  GpuMatrix* outputPtr = dynamic_cast<GpuMatrix*>(&output);
  auto labelPtr = dynamic_cast<GpuSparseMatrix*>(&label);

  CHECK(outputPtr && labelPtr) << "Invalid argument pointer";
  CHECK(labelPtr->format_ == SPARSE_CSR) << "Matrix format not supported";
  CHECK(height_ == outputPtr->height_ && width_ == outputPtr->width_ &&
        outputPtr->width_ == labelPtr->getWidth() &&
        outputPtr->height_ == labelPtr->getHeight())
      << "Matrix dimensions are not equal";

  real* output_d = outputPtr->data_;
  real* grad_d = data_;
  hl_sparse_matrix_s mat_d = labelPtr->sMatrix_.get();
  hl_matrix_multi_binary_cross_entropy_bp(
      output_d, grad_d, mat_d, height_, width_);
}

/**
 * CpuMatrix
 */

CpuMatrix::CpuMatrix(size_t height, size_t width, bool trans)
    : Matrix(std::make_shared<CpuMemoryHandle>(height * width * sizeof(real)),
             height,
             width,
             trans,
             false) {}

CpuMatrix::~CpuMatrix() {}

void CpuMatrix::zeroMem() {
  CHECK(data_ != NULL);
  if (isContiguous()) {
    memset(data_, 0, height_ * width_ * sizeof(real));
  } else {
    BaseMatrix::zero();
  }
}
void CpuMatrix::resetOne() {
  CHECK(data_ != NULL);
  BaseMatrix::one();
}

void CpuMatrix::copyFrom(const Matrix& src) {
  CHECK(isContiguous());
  if (typeid(src) == typeid(GpuMatrix)) {
    CHECK(src.isContiguous());
    CHECK(elementCnt_ == src.getElementCnt());
    hl_memcpy_device2host(
        data_, const_cast<real*>(src.getData()), sizeof(real) * elementCnt_);
  } else if (typeid(src) == typeid(CpuMatrix) ||
             typeid(src) == typeid(SharedCpuMatrix)) {
    CHECK(src.isContiguous());
    CHECK(elementCnt_ == src.getElementCnt());
    memcpy(data_, src.getData(), sizeof(real) * elementCnt_);
  } else if (typeid(src) == typeid(CpuSparseMatrix)) {
    CHECK_GE(elementCnt_, src.getElementCnt());
    copyFrom(dynamic_cast<CpuSparseMatrix&>(const_cast<Matrix&>(src)));
  } else {
    LOG(FATAL) << "Wrong";
  }
}

void CpuMatrix::copyFrom(CpuSparseMatrix& src) {
  CHECK(isContiguous());
  CHECK(height_ == src.getHeight());
  CHECK(width_ == src.getWidth());
  memset(data_, 0, sizeof(real) * height_ * width_);
  if (src.getValueType() == FLOAT_VALUE) {
    if (src.getFormat() == SPARSE_CSC) {
      int* rows = src.getRows();
      real* vals = src.getValue();
      for (size_t i = 0; i < width_; i++) {
        for (size_t j = src.getColStartIdx(i); j < src.getColStartIdx(i + 1);
             j++) {
          data_[rows[j] * width_ + i] = vals[j];
        }
      }
    } else {
      int* cols = src.getCols();
      real* vals = src.getValue();
      for (size_t i = 0; i < height_; i++) {
        for (size_t j = src.getRowStartIdx(i); j < src.getRowStartIdx(i + 1);
             j++) {
          data_[i * width_ + cols[j]] = vals[j];
        }
      }
    }
  } else {
    if (src.getFormat() == SPARSE_CSC) {
      int* rows = src.getRows();
      for (size_t i = 0; i < width_; i++) {
        for (size_t j = src.getColStartIdx(i); j < src.getColStartIdx(i + 1);
             j++) {
          data_[rows[j] * width_ + i] = 1.0;
        }
      }
    } else {
      int* cols = src.getCols();
      for (size_t i = 0; i < height_; i++) {
        for (size_t j = src.getRowStartIdx(i); j < src.getRowStartIdx(i + 1);
             j++) {
          data_[i * width_ + cols[j]] = 1.0;
        }
      }
    }
  }
}

void CpuMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  CHECK(isContiguous());
  CHECK(src.isContiguous());
  CHECK(elementCnt_ == src.getElementCnt());
  if (typeid(src) == typeid(GpuMatrix)) {
    hl_memcpy_async(this->getData(),
                    const_cast<real*>(src.getData()),
                    sizeof(real) * elementCnt_,
                    stream);
  } else if (typeid(src) == typeid(CpuMatrix)) {
    memcpy(data_, src.getData(), sizeof(real) * elementCnt_);
  } else {
    LOG(FATAL) << "Wrong";
  }
}

void CpuMatrix::copyFrom(const real* cpuSrc, size_t size) {
  CHECK(isContiguous());
  CHECK(size <= elementCnt_);
  memcpy(data_, cpuSrc, sizeof(real) * size);
}

void CpuMatrix::copyFrom(const real* cpuSrc, const int64_t* seq) {
  CHECK(isContiguous());
  for (size_t i = 0; i < height_; i++) {
    memcpy(data_ + i * width_, cpuSrc + seq[i] * width_, sizeof(real) * width_);
  }
}

void CpuMatrix::copyFrom(const IVector& src) {
  CHECK(isContiguous());
  CHECK(elementCnt_ == src.getSize())
      << "the src and dst should have same size.";
  const int* cpuSrc = NULL;
  IVectorPtr tmp;
  if (src.useGpu()) {
    CpuIVector tmp(src.getSize());
    tmp.copyFrom(src);
    cpuSrc = tmp.getData();
  } else {
    cpuSrc = src.getData();
  }
  for (size_t i = 0; i < elementCnt_; ++i) {
    data_[i] = cpuSrc[i];
  }
}

void CpuMatrix::copyByRowIndex(Matrix& b, const IVector& rowIndex) {
  size_t height = getHeight();
  size_t width = getWidth();
  CHECK_EQ(b.getWidth(), width);
  const int* index = rowIndex.getData();
  for (size_t i = 0; i < height; i++) {
    CHECK_LT(static_cast<size_t>(index[i]), b.getHeight());
    real* src = b.getData() + index[i] * width;
    real* dst = getData() + i * width;
    memcpy(dst, src, sizeof(real) * width);
  }
}

MatrixPtr CpuMatrix::clone(size_t height, size_t width, bool useGpu) {
  CHECK(isContiguous());

  if (height == 0 && width == 0) {
    height = height_;
    width = width_;
  }

  CHECK(width && height);

  if (useGpu) {
    return std::make_shared<GpuMatrix>(height, width);
  } else {
    return std::make_shared<CpuMatrix>(height, width);
  }
}

void CpuMatrix::resize(size_t newHeight, size_t newWidth) {
  size_t newSize = newHeight * newWidth;
  if (NULL == memoryHandle_.get() ||
      newSize * sizeof(real) > memoryHandle_->getAllocSize()) {
    memoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize * sizeof(real));
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newSize;
  stride_ = width_;
}

real CpuMatrix::getElement(size_t x, size_t y) const {
  return data_[x * stride_ + y];
}

real CpuMatrix::getSum() {
  CHECK(isContiguous());
  double sum = 0;
  for (size_t i = 0; i < height_; ++i) {
    for (size_t j = 0; j < width_; ++j) {
      sum += data_[i * width_ + j];
    }
  }
  return sum;
}

void CpuMatrix::accumulateColSum(Matrix& src) {
  CHECK_EQ(getWidth(), src.getWidth());
  CHECK_EQ(getHeight(), (size_t)1);

  sumCols(src, /* scaleSum= */ 1, /* scaleDest= */ 1);
}

real CpuMatrix::getAbsSum() {
  CHECK(isContiguous());
  double sum = 0;
  for (size_t i = 0; i < height_; ++i) {
    for (size_t j = 0; j < width_; ++j) {
      sum += fabs(data_[i * width_ + j]);
    }
  }
  return sum;
}

MatrixPtr CpuMatrix::getTranspose() {
  if (memoryHandle_.get() != NULL) {
    return std::make_shared<CpuMatrix>(
        std::dynamic_pointer_cast<CpuMemoryHandle>(memoryHandle_),
        height_,
        width_,
        true);
  } else {
    MatrixPtr copy_T(new CpuMatrix(data_, height_, width_, true));
    return copy_T;
  }
}

void CpuMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  if (memAlloc) {
    matTrans = std::make_shared<CpuMatrix>(width_, height_);
  } else {
    CHECK(matTrans != NULL);
    CHECK_EQ(matTrans->getHeight(), width_);
    CHECK_EQ(matTrans->getWidth(), height_);
  }
  real* dataTrans = matTrans->getData();
  real* data = getData();
  int lda = getStride();
  int ldc = matTrans->getStride();

  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      dataTrans[j * ldc + i] = data[i * lda + j];
    }
  }
}

void CpuMatrix::rotate(MatrixPtr& matRot, bool memAlloc, bool clockWise) {
  if (memAlloc) {
    matRot = std::make_shared<CpuMatrix>(width_, height_);
  } else {
    CHECK(matRot != NULL);
    CHECK_EQ(matRot->getHeight(), width_);
    CHECK_EQ(matRot->getWidth(), height_);
  }
  real* dataRot = matRot->getData();
  real* data = getData();

  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      if (clockWise) {
        dataRot[j * height_ + i] = data[(height_ - i - 1) * width_ + j];
      } else {
        dataRot[j * height_ + i] = data[i * width_ + (width_ - j - 1)];
      }
    }
  }
}

MatrixPtr CpuMatrix::getInverse() {
  MatrixPtr matInv;
  inverse(matInv, true);
  return matInv;
}

void CpuMatrix::inverse(MatrixPtr& matInv, bool memAlloc) {
  CHECK_EQ(height_, width_);

  if (memAlloc) {
    matInv = std::make_shared<CpuMatrix>(height_, width_);
  } else {
    CHECK(matInv != NULL);
  }

  CHECK_EQ(height_, matInv->getHeight());
  CHECK_EQ(width_, matInv->getWidth());
  matInv->copyFrom(*this);

  real* data = getData();
  real* dataInv = matInv->getData();
  int ldc = matInv->getStride();

  if (height_ == 1) {
    CHECK_NE(*data, 0);
    *dataInv = 1.0 / (*data);
    return;
  }

  /* Compute the LU decomposition of the matrix */
  std::vector<int> ipiv(height_);
  CBLAS_ORDER order = (matInv->isTransposed() ? CblasColMajor : CblasRowMajor);
  int info = getrf<real>(order, height_, height_, dataInv, ldc, ipiv.data());
  CHECK_EQ(info, 0);

  /* Compute the inverse of the matrix given its LU decompsotion */
  info = getri<real>(order, height_, dataInv, ldc, ipiv.data());
  CHECK_EQ(info, 0);
}

void CpuMatrix::convExpand(Matrix& feature,
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
  CHECK(feature.useGpu_ == false) << "Matrix type are not equal";

  CHECK_EQ(size_t(feaImgHeight * feaImgWidth * channels),
           feature.getHeight() * feature.getWidth())
      << "Matrix dimensions are not equal";

  size_t elemCnt = outputH * outputW * blockH * blockW * channels;
  CHECK_EQ(elemCnt, height_ * width_) << "Matrix dimensions are not equal";

  int channelsCol = channels * blockH * blockW;
  real* srcData = feature.getData();
  for (int c = 0; c < channelsCol; ++c) {
    int wOffset = c % blockW;
    int hOffset = (c / blockW) % blockH;
    int c_im = c / blockH / blockW;
    for (int h = 0; h < outputH; ++h) {
      for (int w = 0; w < outputW; ++w) {
        // no c_im*height to Exclude the channel number
        int imgRowIdx = h * strideH + hOffset;
        int imgColIdx = w * strideW + wOffset;
        if ((imgRowIdx - paddingH) < 0 ||
            (imgRowIdx - paddingH) >= feaImgHeight ||
            (imgColIdx - paddingW) < 0 ||
            (imgColIdx - paddingW) >= feaImgWidth) {
          data_[(c * outputH + h) * outputW + w] = 0;
        } else {
          imgRowIdx += c_im * feaImgHeight - paddingH;
          imgColIdx -= paddingW;
          data_[(c * outputH + h) * outputW + w] =
              srcData[imgRowIdx * feaImgWidth + imgColIdx];
        }
      }
    }
  }
}

void CpuMatrix::convShrink(Matrix& expandFeat,
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
                           real alpha,
                           real beta) {
  CHECK(expandFeat.useGpu_ == false) << "Matrix type are not equal";
  CHECK_EQ(size_t(thisImgHeight * thisImgWidth * channels),
           getHeight() * getWidth())
      << "Matrix dimensions are not equal";

  size_t elemCnt = outputH * outputW * blockH * blockW * channels;

  CHECK(elemCnt == expandFeat.getHeight() * expandFeat.getWidth())
      << "Matrix dimensions are not equal";

  real* expandData = expandFeat.getData();
  int channelsCol = channels * blockH * blockW;
  for (int c = 0; c < channelsCol; ++c) {
    int wOffset = c % blockW;
    int hOffset = (c / blockW) % blockH;
    int c_im = c / blockW / blockH;
    for (int h = 0; h < outputH; ++h) {
      for (int w = 0; w < outputW; ++w) {
        int imRowIdx = h * strideH + hOffset;
        int imColIdx = w * strideW + wOffset;
        if ((imRowIdx - paddingH) >= 0 &&
            (imRowIdx - paddingH) < thisImgHeight &&
            (imColIdx - paddingW) >= 0 &&
            (imColIdx - paddingW) < thisImgWidth) {
          imRowIdx += c_im * thisImgHeight - paddingH;
          imColIdx -= paddingW;
          data_[imRowIdx * thisImgWidth + imColIdx] =
              alpha * expandData[(c * outputH + h) * outputW + w] +
              beta * data_[imRowIdx * thisImgWidth + imColIdx];
        }
      }
    }
  }
}

void CpuMatrix::maxPoolForward(Matrix& inputMat,
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
  real* inputData = inputMat.getData();
  real* outData = data_;
  size_t num = inputMat.getHeight();
  size_t inWidth = imgSizeW;
  size_t inHeight = imgSizeH;
  CHECK(inHeight * inWidth == inputMat.getWidth() / channels);
  CHECK_EQ(num, this->getHeight());
  CHECK_EQ(channels * outputH * outputW, this->getWidth());
  size_t outStride = getStride();

  /* initialize the data_ */
  for (size_t i = 0; i < height_; i++) {
    for (size_t j = 0; j < width_; j++) {
      outData[i * outStride + j] = -(real)FLT_MAX;
    }
  }

  /* pool max one by one */
  for (size_t n = 0; n < num; ++n) {  // frame by frame
    if (!isContiguous()) {
      outData = data_ + n * outStride;
    }
    for (size_t c = 0; c < channels; ++c) {  // channel by channel
      for (size_t ph = 0; ph < outputH; ++ph) {
        for (size_t pw = 0; pw < outputW; ++pw) {
          int hstart = ph * strideH - paddingH;
          int wstart = pw * strideW - paddingW;
          int hend = std::min(hstart + sizeY, inHeight);
          int wend = std::min(wstart + sizeX, inWidth);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              outData[ph * outputW + pw] = std::max(outData[ph * outputW + pw],
                                                    inputData[h * inWidth + w]);
            }
          }
        }
      }
      // compute offset
      inputData += inHeight * inWidth;
      outData += outputH * outputW;
    }
  }
}

void CpuMatrix::maxPoolBackward(Matrix& image,
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
  size_t num = image.getHeight();
  size_t channels = size_t(width_ / imgSizeH / imgSizeW);
  CHECK(image.getWidth() == imgSizeH * imgSizeW * channels);
  CHECK(image.getHeight() == height_ && image.getWidth() == width_);
  CHECK(outV.getHeight() == outGrad.getHeight() &&
        outV.getWidth() == outGrad.getWidth());

  real* tgtGrad = data_;
  real* inData = image.getData();
  real* otData = outV.getData();
  real* otGrad = outGrad.getData();

  size_t outStride = outV.getStride();
  real* origOutData = otData;
  real* origOutGrad = otGrad;

  for (size_t n = 0; n < num; ++n) {
    if (!outV.isContiguous()) {
      otData = origOutData + n * outStride;
      otGrad = origOutGrad + n * outStride;
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t ph = 0; ph < outputH; ++ph) {
        for (size_t pw = 0; pw < outputW; ++pw) {
          int hstart = ph * strideH - paddingH;
          int wstart = pw * strideW - paddingW;
          int hend = std::min(hstart + sizeY, imgSizeH);
          int wend = std::min(wstart + sizeX, imgSizeW);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              tgtGrad[h * imgSizeW + w] =
                  scaleTargets * tgtGrad[h * imgSizeW + w] +
                  scaleOutput * otGrad[ph * outputW + pw] *
                      (inData[h * imgSizeW + w] == otData[ph * outputW + pw]);
            }
          }
        }
      }
      // offset
      inData += imgSizeH * imgSizeW;
      tgtGrad += imgSizeH * imgSizeW;
      otData += outputH * outputW;
      otGrad += outputH * outputW;
    }
  }
}

void CpuMatrix::avgPoolForward(Matrix& input,
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
  // The main loop
  size_t num = input.getHeight();
  size_t inHeight = imgSizeH;
  size_t inWidth = imgSizeW;
  CHECK(inHeight * inWidth * channels == input.getWidth());
  CHECK(outputH * outputW * channels * num == height_ * width_);
  real* tgtData = data_;
  real* inData = input.getData();

  for (size_t n = 0; n < num; ++n) {
    if (!isContiguous()) {
      tgtData = data_ + n * getStride();
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t ph = 0; ph < outputH; ++ph) {
        for (size_t pw = 0; pw < outputW; ++pw) {
          int hstart = ph * strideH - paddingH;
          int wstart = pw * strideW - paddingW;
          int hend = std::min(hstart + sizeY, inHeight + paddingH);
          int wend = std::min(wstart + sizeX, inWidth + paddingW);
          int poolSize = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, static_cast<int>(inHeight));
          wend = std::min(wend, static_cast<int>(inWidth));

          CHECK(poolSize);
          tgtData[ph * outputW + pw] = 0;  // clear
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              tgtData[ph * outputW + pw] += inData[h * inWidth + w];
            }
          }
          tgtData[ph * outputW + pw] /= poolSize;
        }
      }
      // compute offset
      inData += inHeight * inWidth;
      tgtData += outputH * outputW;
    }
  }
}

void CpuMatrix::avgPoolBackward(Matrix& input,
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
  size_t num = input.getHeight();
  size_t channels = input.getWidth() / outputH / outputW;
  CHECK(imgSizeH * imgSizeW * channels == getWidth());
  real* inData = input.getData();
  real* outData = getData();

  for (size_t n = 0; n < num; ++n) {
    if (!input.isContiguous()) {
      inData = input.getData() + n * input.getStride();
    }
    for (size_t c = 0; c < channels; ++c) {
      for (size_t ph = 0; ph < outputH; ++ph) {
        for (size_t pw = 0; pw < outputW; ++pw) {
          int hstart = ph * strideH - paddingH;
          int wstart = pw * strideW - paddingW;
          int hend = std::min(hstart + sizeY, imgSizeH + paddingH);
          int wend = std::min(wstart + sizeX, imgSizeW + paddingW);
          int poolSize = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, static_cast<int>(imgSizeH));
          wend = std::min(wend, static_cast<int>(imgSizeW));
          CHECK(poolSize);

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              outData[h * imgSizeW + w] += inData[ph * outputW + pw] / poolSize;
            }
          }
        }
      }
      // offset
      outData += imgSizeH * imgSizeW;
      inData += outputH * outputW;
    }
  }
}

/**
 * Input: one or more sequences. Each sequence contains some instances.
 * Output: output size is the number of input sequences (NOT input instances).
 * output[i] is set to max_{for each instance in this sequence}{input[i]}
 */
void CpuMatrix::maxSequenceForward(Matrix& input,
                                   const IVector& sequence,
                                   IVector& index) {
  CHECK(dynamic_cast<CpuMatrix*>(&input));
  CHECK(dynamic_cast<const CpuIVector*>(&sequence));
  CHECK(dynamic_cast<CpuIVector*>(&index));

  real* outData = getData();
  real* inputData = input.getData();
  const int* starts = sequence.getData();
  int* maxIndex = index.getData();
  size_t numSequences = getHeight();
  size_t dim = getWidth();

  CHECK_EQ(dim, input.getWidth());
  CHECK_EQ(numSequences, sequence.getSize() - 1);
  CHECK_EQ(starts[numSequences], (int)input.getHeight());
  CHECK_EQ(numSequences * dim, index.getSize());

  for (size_t sequenceId = 0; sequenceId < numSequences; ++sequenceId) {
    // current sequence, loop for each input instance
    // (1) first instance: do not need compare, copy value to outV directly
    for (size_t k = 0; k < dim; ++k) {
      outData[sequenceId * dim + k] = inputData[starts[sequenceId] * dim + k];
      maxIndex[sequenceId * dim + k] = starts[sequenceId];
    }
    // (2) other instance in same sequence
    for (int insId = starts[sequenceId] + 1; insId < starts[sequenceId + 1];
         ++insId) {
      // insId is the index on all instances
      for (size_t k = 0; k < dim; ++k) {
        // for each dim
        if (inputData[insId * dim + k] > outData[sequenceId * dim + k]) {
          // update max value and record index
          outData[sequenceId * dim + k] = inputData[insId * dim + k];
          maxIndex[sequenceId * dim + k] = insId;
        }
      }
    }
  }
}

void CpuMatrix::maxSequenceBackward(Matrix& outputGrad,
                                    const IVector& sequence,
                                    IVector& index) {
  CHECK(dynamic_cast<CpuMatrix*>(&outputGrad));
  CHECK(dynamic_cast<const CpuIVector*>(&sequence));
  CHECK(dynamic_cast<CpuIVector*>(&index));

  real* inputGrad = getData();
  real* outGrad = outputGrad.getData();
  int* maxIndex = index.getData();
  size_t dim = getWidth();
  size_t numSequences = sequence.getSize() - 1;

  CHECK_EQ(dim, outputGrad.getWidth());
  CHECK_EQ(numSequences, outputGrad.getHeight());
  CHECK_EQ(numSequences * dim, index.getSize());

  for (size_t sequenceId = 0; sequenceId < numSequences; ++sequenceId) {
    // current sequence
    for (size_t j = 0; j < dim; ++j) {
      // each dim
      int insId = maxIndex[sequenceId * dim + j];
      inputGrad[insId * dim + j] += outGrad[sequenceId * dim + j];
    }
  }
}

inline void vecAddTo(real* a, const real* b, size_t len) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i] += b[i];
  }
}

inline void vecAddTo(real* a, const real* b, real scaleB, size_t len) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i] += scaleB * b[i];
  }
}

inline void colVecAddTo(
    real* a, const real* b, size_t len, size_t aWidth, size_t bWidth) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i * aWidth] += b[i * bWidth];
  }
}

inline void colVecAddTo(
    real* a, real* b, real c, size_t len, size_t aWidth, size_t bWidth) {
  for (unsigned int i = 0; i < len; ++i) {
    a[i * aWidth] += b[i * bWidth] * c;
  }
}

void CpuMatrix::addBias(Matrix& b, real scale) {
  CHECK(b.useGpu_ == false) << "Matrix type are not equal";

  CHECK_EQ(b.getHeight(), (size_t)1);
  CHECK_EQ(width_, b.getWidth());
  real* aData = getData();
  real* bData = b.getData();
  size_t numSamples = getHeight();
  size_t dim = getWidth();

  if (scale == 1 && getStride() % 32 == 0) {  // use libaddto
    // @TODO(yuyang18) Make input addr can be unaligned.
    // So merge this if and else
    CHECK_EQ((size_t)aData % 32, 0UL);
    CHECK_EQ((size_t)bData % 32, 0UL);
    for (size_t i = 0; i < numSamples; i++) {
      simd::addTo(aData + i * getStride(), bData, dim);
    }
  } else {
    for (size_t i = 0; i < numSamples; i++) {
      for (size_t j = 0; j < dim; j++) {
        aData[i * getStride() + j] += scale * bData[j];
      }
    }
  }
}

void CpuMatrix::addSharedBias(Matrix& b, real scale) {
  CHECK_EQ(b.getHeight(), (size_t)1);
  real* aData = getData();
  real* bData = b.getData();
  size_t numSamples = getHeight();
  size_t channel = b.getWidth();
  CHECK_EQ(getWidth() % channel, 0UL);
  size_t dim = getWidth() / channel;

  for (size_t i = 0; i < numSamples; i++) {
    for (size_t c = 0; c < channel; c++) {
      for (size_t j = 0; j < dim; j++) {
        aData[i * getStride() + c * dim + j] += scale * bData[c];
      }
    }
  }
}

void CpuMatrix::collectBias(Matrix& a, real scale) {
  CHECK_EQ(getHeight(), (size_t)1);
  CHECK_EQ(width_, a.getWidth());
  CpuSparseMatrix* aptr = dynamic_cast<CpuSparseMatrix*>(&a);
  if (!aptr) {
    sumCols(a, /* scaleSum= */ scale, /* scaleDest= */ 1);
  } else {
    size_t nnz = aptr->getElementCnt();
    int* cols = aptr->getCols();
    real* A = aptr->getValue();
    real* B = getData();
    for (size_t i = 0; i < nnz; i++) {
      B[cols[i]] += scale * A[i];
    }
  }
}

void CpuMatrix::collectSharedBias(Matrix& a, real scale) {
  CHECK_EQ(getHeight(), (size_t)1);
  real* B = getData();
  real* A = a.getData();
  size_t numSamples = a.getHeight();
  size_t channel = getWidth();
  CHECK_EQ(a.getWidth() % channel, 0UL);
  size_t dim = a.getWidth() / channel;
  for (size_t i = 0; i < numSamples; i++) {
    for (size_t c = 0; c < channel; c++) {
      for (size_t j = 0; j < dim; j++) {
        B[c] += scale * A[i * channel * dim + c * dim + j];
      }
    }
  }
}

void CpuMatrix::sequenceAvgForward(Matrix& a,
                                   const IVector& startsPos,
                                   int mode) {
  size_t height = getHeight();
  size_t width = getWidth();
  CHECK_EQ(height, startsPos.getSize() - 1);
  CHECK_EQ(width, a.getWidth());
  real* dst = getData();
  real* src = a.getData();
  const int* starts = startsPos.getData();
  MatrixPtr outMtx = Matrix::create(nullptr, 1, width, false, false);
  MatrixPtr dataMtx = Matrix::create(nullptr, 1, width, false, false);
  for (size_t i = 0; i < height; i++) {
    int sequenceLength = starts[i + 1] - starts[i];
    if (0 == sequenceLength) {
      // empty sequence
      continue;
    }
    outMtx->setData(dst + i * width);
    dataMtx->setData(src + starts[i] * width, sequenceLength, width);
    if (mode == 0) {
      // plain average
      outMtx->sumCols(*dataMtx,
                      (real)1 / (real)sequenceLength,
                      /* scaleDest= */ 1);
    } else if (mode == 1) {
      // sum instead of average
      outMtx->sumCols(*dataMtx, /* scaleSum= */ 1, /* scaleDest= */ 1);
    } else if (mode == 2) {
      // divide by square root of sequenceLength
      outMtx->sumCols(*dataMtx,
                      (real)1 / std::sqrt(sequenceLength),
                      /* scaleDest= */ 1);
    } else {
      LOG(FATAL) << "should not reach here";
    }
  }
}

void CpuMatrix::sequenceAvgBackward(Matrix& a,
                                    const IVector& startsPos,
                                    int mode) {
  size_t height = a.getHeight();
  size_t width = getWidth();
  CHECK_EQ(height, startsPos.getSize() - 1);
  CHECK_EQ(width, a.getWidth());
  real* dst = getData();
  real* src = a.getData();
  const int* starts = startsPos.getData();
  MatrixPtr outMtx = Matrix::create(nullptr, 1, width, false, false);
  MatrixPtr dataMtx = Matrix::create(nullptr, 1, width, false, false);
  for (size_t i = 0; i < height; ++i) {
    int sequenceLength = starts[i + 1] - starts[i];
    if (0 == sequenceLength) {
      // empty sequence
      continue;
    }
    outMtx->setData(dst + starts[i] * width, sequenceLength, width);
    dataMtx->setData(src + i * width);
    if (mode == 0) {
      // plain average
      outMtx->addBias(*dataMtx, 1.0f / sequenceLength);
    } else if (mode == 1) {
      // sum instead of average
      outMtx->addBias(*dataMtx, 1.0f);
    } else if (mode == 2) {
      // divide by square root of sequenceLength
      outMtx->addBias(*dataMtx, 1.0f / std::sqrt(sequenceLength));
    } else {
      LOG(FATAL) << "should not reach here";
    }
  }
}

/* this = scaleAB*(a*b) + scaleT*this*/
void CpuMatrix::mul(const Matrix& a,
                    const Matrix& b,
                    real scaleAB,
                    real scaleT) {
  CHECK(!isTransposed()) << "Not supported";
  const auto a_ptr = dynamic_cast<const CpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const CpuMatrix*>(&b);
  const auto a_ptr_s = dynamic_cast<const CpuSparseMatrix*>(&a);
  const auto b_ptr_s = dynamic_cast<const CpuSparseMatrix*>(&b);

  if (a_ptr && b_ptr) {
    mul((CpuMatrix*)a_ptr, (CpuMatrix*)b_ptr, scaleAB, scaleT);
  } else if (a_ptr_s && b_ptr) {
    mul((CpuSparseMatrix*)a_ptr_s, (CpuMatrix*)b_ptr, scaleAB, scaleT);
  } else if (a_ptr && b_ptr_s) {
    mul((CpuMatrix*)a_ptr, (CpuSparseMatrix*)b_ptr_s, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "Not supported";
  }
}

void CpuMatrix::mul(CpuSparseMatrix* a,
                    CpuMatrix* b,
                    real scaleAB,
                    real scaleT) {
  if (dynamic_cast<CacheRowCpuMatrix*>(b)) {
    return mul(a, dynamic_cast<CacheRowCpuMatrix*>(b), this, scaleAB, scaleT);
  } else if (dynamic_cast<SparseRowCpuMatrix*>(b)) {
    return mul(a, dynamic_cast<SparseRowCpuMatrix*>(b), this, scaleAB, scaleT);
  } else {
    return mul(a, b, this, scaleAB, scaleT);
  }
}

void CpuMatrix::mul(CpuMatrix* a, CpuMatrix* b, real scaleAB, real scaleT) {
  CHECK(!isTransposed()) << "Not supported";

  size_t a_col, b_col, a_row, b_row;
  CBLAS_TRANSPOSE a_trans, b_trans;
  if (!a->isTransposed()) {
    a_col = a->getWidth();
    a_row = a->getHeight();
    a_trans = CblasNoTrans;
  } else {
    a_col = a->getHeight();
    a_row = a->getWidth();
    a_trans = CblasTrans;
  }
  if (!b->isTransposed()) {
    b_col = b->getWidth();
    b_row = b->getHeight();
    b_trans = CblasNoTrans;
  } else {
    b_col = b->getHeight();
    b_row = b->getWidth();
    b_trans = CblasTrans;
  }

  CHECK_EQ(a_col, b_row);
  CHECK_EQ(a_row, getHeight());
  CHECK_EQ(b_col, getWidth());

  real* A = a->getData();
  real* B = b->getData();
  real* C = getData();

  int M = getHeight();
  int N = getWidth();
  int K = a_col;
  int lda = a->getStride();
  int ldb = b->getStride();
  int ldc = getStride();
  gemm<real>(
      a_trans, b_trans, M, N, K, scaleAB, A, lda, B, ldb, scaleT, C, ldc);
}

void CpuMatrix::mul(
    CpuMatrix* a, CpuMatrix* b, CpuSparseMatrix* c, real scaleAB, real scaleT) {
  CHECK(!c->isTransposed()) << "Not supported";
  CHECK_EQ(c->getValueType(), FLOAT_VALUE);

  real* A = a->getData();
  real* B = b->getData();
  real* C = c->getValue();
  int* rows = c->getRows();
  int* cols = c->getCols();
  size_t height = c->getHeight();
  size_t width = c->getWidth();
  if (scaleT == 0) {
    c->zeroMem();
  }

  if (!a->isTransposed() && !b->isTransposed()) {
    size_t m = a->getWidth();
    CHECK_EQ(b->getHeight(), m);
    CHECK_EQ(a->getHeight(), height);
    CHECK_EQ(b->getWidth(), width);
    if (c->getFormat() == SPARSE_CSC) {
      for (size_t i = 0; i < width; i++) {
        size_t start = c->getColStartIdx(i);
        size_t end = c->getColStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t rowIdx = rows[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[rowIdx * m + k] * B[k * width + i];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      for (size_t i = 0; i < height; i++) {
        size_t start = c->getRowStartIdx(i);
        size_t end = c->getRowStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[i * m + k] * B[k * width + colIdx];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    }
  } else if (a->isTransposed() && !b->isTransposed()) {
    size_t m = a->getHeight();
    CHECK_EQ(m, b->getHeight());
    CHECK_EQ(b->getWidth(), width);
    CHECK_EQ(a->getWidth(), height);

    if (c->getFormat() == SPARSE_CSC) {
      for (size_t i = 0; i < width; i++) {
        size_t start = c->getColStartIdx(i);
        size_t end = c->getColStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t rowIdx = rows[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[k * height + rowIdx] * B[k * width + i];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      for (size_t i = 0; i < height; i++) {
        int start = c->getRowStartIdx(i);
        int end = c->getRowStartIdx(i + 1);
        for (int j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[k * height + i] * B[k * width + colIdx];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    }
  } else if (!a->isTransposed() && b->isTransposed()) {
    size_t m = a->getWidth();
    CHECK_EQ(b->getWidth(), m);
    CHECK_EQ(a->getHeight(), height);
    CHECK_EQ(b->getHeight(), width);
    if (c->getFormat() == SPARSE_CSR) {
      for (size_t i = 0; i < height; i++) {
        size_t start = c->getRowStartIdx(i);
        size_t end = c->getRowStartIdx(i + 1);
        for (size_t j = start; j < end; j++) {
          real sum = 0;
          size_t colIdx = cols[j];
          for (size_t k = 0; k < m; k++) {
            sum += A[i * m + k] * B[colIdx * m + k];
          }
          C[j] = scaleAB * sum + scaleT * C[j];
        }
      }
    } else {
      LOG(FATAL) << "Not supported csc format "
                    "when a is not trans and b is trans";
    }
  } else {
    LOG(FATAL) << "Not supported";
  }
}

void CpuMatrix::mul(CpuMatrix* a,
                    CpuSparseMatrix* b,
                    real scaleAB,
                    real scaleT) {
  CHECK(!trans_) << "Not supported";
  CHECK(!a->isTransposed()) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1);

  // TODO(yuyang18): Maybe bug implementation here
  CHECK_EQ(scaleAB, static_cast<real>(1.0));

  real* A = a->getData();
  real* B = b->getValue();
  real* C = getData();
  int* rows = b->getRows();
  int* cols = b->getCols();

  if (scaleT == 0) {
    zeroMem();
  }
  if (b->getFormat() == SPARSE_CSC) {
    if (!b->isTransposed()) {
      size_t m = a->getWidth();
      CHECK_EQ(b->getHeight(), m);
      CHECK_EQ(a->getHeight(), height_);
      CHECK_EQ(b->getWidth(), width_);

      if (b->getValueType() == NO_VALUE) {
        for (size_t j = 0; j < b->getWidth(); ++j) {
          int start = b->getColStartIdx(j);
          int end = b->getColStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(C + j, A + rows[i], height_, width_, a->getWidth());
          }
        }
      } else if (b->getValueType() == FLOAT_VALUE) {
        for (size_t j = 0; j < b->getWidth(); ++j) {
          int start = b->getColStartIdx(j);
          int end = b->getColStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(
                C + j, A + rows[i], B[i], height_, width_, a->getWidth());
          }
        }
      }
    } else /*if (b->isTransposed())*/ {
      size_t m = a->getWidth();
      CHECK_EQ(b->getHeight(), width_);
      CHECK_EQ(a->getHeight(), height_);
      CHECK_EQ(b->getWidth(), m);
      if (b->getValueType() == NO_VALUE) {
        for (size_t i = 0; i < b->getWidth(); ++i) {
          int start = b->getColStartIdx(i);
          int end = b->getColStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(C + rows[j], A + i, height_, width_, a->getWidth());
          }
        }
      } else if (b->getValueType() == FLOAT_VALUE) {
        for (size_t i = 0; i < b->getWidth(); ++i) {
          int start = b->getColStartIdx(i);
          int end = b->getColStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(
                C + rows[j], A + i, B[j], height_, width_, a->getWidth());
          }
        }
      }
    }
  } else {
    if (!b->isTransposed()) {
      size_t m = a->getWidth();
      CHECK_EQ(b->getHeight(), m);
      CHECK_EQ(a->getHeight(), height_);
      CHECK_EQ(b->getWidth(), width_);

      if (b->getValueType() == NO_VALUE) {
        for (size_t j = 0; j < b->getHeight(); ++j) {
          int start = b->getRowStartIdx(j);
          int end = b->getRowStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(C + cols[i], A + j, height_, width_, a->getWidth());
          }
        }
      } else if (b->getValueType() == FLOAT_VALUE) {
        for (size_t j = 0; j < b->getHeight(); ++j) {
          int start = b->getRowStartIdx(j);
          int end = b->getRowStartIdx(j + 1);
          for (int i = start; i < end; ++i) {
            colVecAddTo(
                C + cols[i], A + j, B[i], height_, width_, a->getWidth());
          }
        }
      }
    } else /*if (b->isTransposed())*/ {
      size_t m = a->getWidth();
      CHECK_EQ(b->getHeight(), width_);
      CHECK_EQ(a->getHeight(), height_);
      CHECK_EQ(b->getWidth(), m);
      if (b->getValueType() == NO_VALUE) {
        for (size_t i = 0; i < b->getHeight(); ++i) {
          int start = b->getRowStartIdx(i);
          int end = b->getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(C + i, A + cols[j], height_, width_, a->getWidth());
          }
        }
      } else if (b->getValueType() == FLOAT_VALUE) {
        for (size_t i = 0; i < b->getHeight(); ++i) {
          int start = b->getRowStartIdx(i);
          int end = b->getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            colVecAddTo(
                C + i, A + cols[j], B[j], height_, width_, a->getWidth());
          }
        }
      }
    }
  }
}

void CpuMatrix::selectRows(Matrix& table, IVector& ids) {
  if (dynamic_cast<CacheRowCpuMatrix*>(&table)) {
    selectRowsImp(*dynamic_cast<CacheRowCpuMatrix*>(&table), ids);
  } else if (dynamic_cast<SparseRowCpuMatrix*>(&table)) {
    selectRowsImp(*dynamic_cast<SparseRowCpuMatrix*>(&table), ids);
  } else {
    CHECK(table.isContiguous());
    selectRowsImp(*dynamic_cast<CpuMatrix*>(&table), ids);
  }
}

void CpuMatrix::selectElements(Matrix& table, IVector& ids) {
  CHECK_EQ(table.getHeight(), ids.getSize());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), 1U);
  real* tableData = table.getData();
  int* idsData = ids.getData();
  for (size_t i = 0; i < table.getHeight(); i++) {
    data_[i] += tableData[i * table.getWidth() + idsData[i]];
  }
}

void CpuMatrix::addElements(Matrix& table, IVector& ids) {
  CHECK_EQ(table.getHeight(), ids.getSize());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), 1U);
  real* tableData = table.getData();
  int* idsData = ids.getData();
  for (size_t i = 0; i < table.getHeight(); i++) {
    tableData[i * table.getWidth() + idsData[i]] += data_[i];
  }
}

// this.row[i] += table.row[ids[i]]
template <typename TableMatType>
void CpuMatrix::selectRowsImp(TableMatType& table, IVector& ids) {
  CHECK(!table.useGpu());
  CHECK(!ids.useGpu());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), table.getWidth());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  real* a = getData();
  size_t tableSize = table.getHeight();
  int* index = ids.getData();

  for (size_t i = 0; i < numSamples; ++i) {
    if (index[i] == -1) continue;
    CHECK_LT(index[i], (int)tableSize);
    CHECK_GE(index[i], 0);
    vecAddTo(a + i * stride_, table.getRow(index[i]), dim);
  }
}

void CpuMatrix::addToRows(Matrix& table, IVector& ids) {
  if (dynamic_cast<CacheRowCpuMatrix*>(&table)) {
    addToRowsImp(*dynamic_cast<CacheRowCpuMatrix*>(&table), ids);
  } else if (dynamic_cast<SparseAutoGrowRowCpuMatrix*>(&table)) {
    addToRowsImp(*dynamic_cast<SparseAutoGrowRowCpuMatrix*>(&table), ids);
  } else if (dynamic_cast<SparseRowCpuMatrix*>(&table)) {
    addToRowsImp(*dynamic_cast<SparseRowCpuMatrix*>(&table), ids);
  } else {
    CHECK(table.isContiguous());
    addToRowsImp(*dynamic_cast<CpuMatrix*>(&table), ids);
  }
}

// table.row[ids[i]] += this.row[i]
template <typename TableMatType>
void CpuMatrix::addToRowsImp(TableMatType& table, IVector& ids) {
  CHECK(!table.useGpu());
  CHECK(!ids.useGpu());
  CHECK_EQ(getHeight(), ids.getSize());
  CHECK_EQ(getWidth(), table.getWidth());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  real* a = getData();
  size_t tableSize = table.getHeight();
  int* index = ids.getData();

  for (size_t i = 0; i < numSamples; ++i) {
    if (index[i] == -1) continue;
    CHECK_LT(index[i], (int)tableSize);
    CHECK_GE(index[i], 0);
    vecAddTo(table.getRow(index[i]), a + i * stride_, dim);
  }
}

static ThreadLocal<std::vector<const real*>> threadLocalColArray;

template <typename MatBType, typename MatCType>
void CpuMatrix::mul(
    CpuSparseMatrix* a, MatBType* b, MatCType* c, real scaleAB, real scaleT) {
  CHECK(!c->isTransposed()) << "Not supported";
  CHECK(!b->isTransposed()) << "Not supported";
  // TODO(yuyang18): Maybe bug implementation here.
  CHECK(scaleAB == 1) << "Not supported";
  CHECK(scaleT == 0 || scaleT == 1) << "Not supported";
  CHECK_EQ(a->getFormat(), SPARSE_CSR) << "Not supported";

  real* B = b->getData();
  real* C = c->getData();
  size_t height = c->getHeight();
  size_t width = c->getWidth();
  int* cols = a->getCols();
  real* values = a->getValue();

  if (scaleT == 0) {
    c->zeroMem();
  }

  if (!a->isTransposed()) {
    size_t m = a->getWidth();
    CHECK_EQ(b->getHeight(), m);
    CHECK_EQ(a->getHeight(), height);
    CHECK_EQ(b->getWidth(), width);

    if (a->getValueType() == NO_VALUE) {
      if (width % 32 == 0) {  // use libaddto
        // @TODO(yuyang18) Make input addr can be unaligned.
        // So merge this if and else
        CHECK_EQ((size_t)B % 32, 0UL);
        CHECK_EQ((size_t)C % 32, 0UL);
        auto& colArray = *threadLocalColArray;
        for (size_t i = 0; i < a->getHeight(); ++i) {
          const int start = a->getRowStartIdx(i);
          const int end = a->getRowStartIdx(i + 1);
          size_t colNum = end - start;
          colArray.resize(colNum);
          for (int j = 0; j < end - start; ++j) {
            colArray[j] = b->getRow(cols[j + start]);
          }
          simd::batchAddTo(c->getRow(i), &colArray[0], colNum, width);
        }

      } else {
        for (size_t i = 0; i < a->getHeight(); ++i) {
          const int start = a->getRowStartIdx(i);
          const int end = a->getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            vecAddTo(c->getRow(i), b->getRow(cols[j]), width);
          }
        }
      }
    } else if (a->getValueType() == FLOAT_VALUE) {
      for (size_t i = 0; i < a->getHeight(); ++i) {
        const int start = a->getRowStartIdx(i);
        const int end = a->getRowStartIdx(i + 1);
        for (int j = start; j < end; ++j) {
          vecAddTo(c->getRow(i), b->getRow(cols[j]), values[j], width);
        }
      }
    }
  } else /*if (a->isTransposed())*/ {
    size_t m = a->getHeight();
    CHECK_EQ(b->getHeight(), m);
    CHECK_EQ(a->getWidth(), height);
    CHECK_EQ(b->getWidth(), width);
    if (a->getValueType() == NO_VALUE) {
      if (width % 32 == 0) {  // use libaddto
        // @TODO(yuyang18) Make input addr can be unaligned.
        // So merge this if and else
        CHECK_EQ((size_t)B % 32, 0UL);
        CHECK_EQ((size_t)C % 32, 0UL);
        for (size_t i = 0; i < a->getHeight(); ++i) {
          const int start = a->getRowStartIdx(i);
          const int end = a->getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            simd::addTo(c->getRow(cols[j]), b->getRow(i), width);
          }
        }

      } else {
        for (size_t i = 0; i < a->getHeight(); ++i) {
          const int start = a->getRowStartIdx(i);
          const int end = a->getRowStartIdx(i + 1);
          for (int j = start; j < end; ++j) {
            vecAddTo(c->getRow(cols[j]), b->getRow(i), width);
          }
        }
      }
    } else if (a->getValueType() == FLOAT_VALUE) {
      for (size_t i = 0; i < a->getHeight(); ++i) {
        const int start = a->getRowStartIdx(i);
        const int end = a->getRowStartIdx(i + 1);
        for (int j = start; j < end; ++j) {
          vecAddTo(c->getRow(cols[j]), b->getRow(i), values[j], width);
        }
      }
    }
  }
}

// instantiation mul() called in SparseRowMatrix.cpp
template void CpuMatrix::mul<CpuMatrix, SparseRowCpuMatrix>(
    CpuSparseMatrix* a,
    CpuMatrix* b,
    SparseRowCpuMatrix* c,
    real scaleAB,
    real scaleT);
template void CpuMatrix::mul<CpuMatrix, SparseAutoGrowRowCpuMatrix>(
    CpuSparseMatrix* a,
    CpuMatrix* b,
    SparseAutoGrowRowCpuMatrix* c,
    real scaleAB,
    real scaleT);
template void CpuMatrix::mul<CpuMatrix, CacheRowCpuMatrix>(CpuSparseMatrix* a,
                                                           CpuMatrix* b,
                                                           CacheRowCpuMatrix* c,
                                                           real scaleAB,
                                                           real scaleT);

void SharedCpuMatrix::mul(CpuSparseMatrix* a,
                          CpuMatrix* b,
                          real scaleAB,
                          real scaleT) {
  CHECK(!isTransposed()) << "Not supported";
  CHECK(!b->isTransposed()) << "Not supported";
  CHECK_EQ(scaleAB, 1) << "Not supported";
  CHECK_EQ(scaleT, 1) << "Not supported";
  CHECK_EQ(a->getFormat(), SPARSE_CSR) << "not supported";

  real* B = b->getData();
  real* C = getData();
  size_t height = getHeight();
  size_t width = getWidth();

  // get real trans
  MatrixPtr aTrans;
  if (a->isTransposed()) {
    aTrans = a->getTmpSparseMatrix(a->getWidth(), a->getHeight());
    a->transpose(aTrans, false);
  }
  a = dynamic_cast<CpuSparseMatrix*>(aTrans.get());

  size_t m = a->getWidth();
  CHECK_EQ(b->getHeight(), m);
  CHECK_EQ(a->getHeight(), height);
  CHECK_EQ(b->getWidth(), width);

  size_t blockSize = (height / blockNum_) + 1;
  CpuMatrixPtr localBuf = *localBuf_;
  if (!localBuf) {
    localBuf = std::make_shared<CpuMatrix>(blockSize, width);
  } else {
    localBuf->resize(blockSize, width);
  }
  localBuf->zeroMem();
  real* localC = localBuf->getData();
  std::vector<int>& blockSeq = *blockSeq_;
  if (blockSeq.size() == 0) {
    for (int k = 0; k < blockNum_; ++k) {
      blockSeq.push_back(k);
    }
    std::shuffle(
        blockSeq.begin(), blockSeq.end(), ThreadLocalRandomEngine::get());
  }
  std::vector<int>& localBufRows = *localBufRows_;
  int* cols = a->getCols();
  real* value = a->getValue();

  for (int k = 0; k < blockNum_; ++k) {
    int blockId = blockSeq[k];
    size_t blockBegin = blockId * blockSize;
    size_t blockEnd = (blockId + 1) * blockSize;
    if (blockId == blockNum_ - 1) {
      blockEnd = height;
    }
    if (a->getValueType() == NO_VALUE) {
      for (size_t i = blockBegin; i < blockEnd; ++i) {
        int start = a->getRowStartIdx(i);
        int end = a->getRowStartIdx(i);
        size_t colNum = a->getColNum(i);
        if (colNum == 0) {
          continue;
        }  // skip empty row
        localBufRows.push_back(i);
        size_t bufPos = localBufRows.size() - 1;
        for (int j = start; j < end; ++j) {
          vecAddTo(localC + bufPos * width, B + cols[j] * width, width);
        }
      }
    } else if (a->getValueType() == FLOAT_VALUE) {
      for (size_t i = blockBegin; i < blockEnd; ++i) {
        int start = a->getRowStartIdx(i);
        int end = a->getRowStartIdx(i);
        size_t colNum = a->getColNum(i);
        if (colNum == 0) {
          continue;
        }  // skip empty row
        localBufRows.push_back(i);
        size_t bufPos = localBufRows.size() - 1;
        for (int j = start; j < end; ++j) {
          vecAddTo(
              localC + bufPos * width, B + cols[j] * width, value[j], width);
        }
      }
    }

    {
      std::lock_guard<std::mutex> guard(*blockLocks_[blockId]);
      for (size_t i = 0; i < localBufRows.size(); ++i) {
        vecAddTo(C + localBufRows[i] * width, localC + i * width, width);
      }
    }
    memset(localC, 0, localBufRows.size() * width * sizeof(real));
    localBufRows.clear();
  }

  VLOG(2) << " B[0]=" << B[0] << " B[1]=" << B[1] << " C[0]=" << C[0]
          << " C[1]=" << C[1];
}

void SharedCpuMatrix::add(Matrix& b, real p1, real p2) {
  CHECK_EQ(blockNum_, 1);
  std::lock_guard<std::mutex> guard(*blockLocks_[0]);
  CpuMatrix::add(b, p1, p2);
}

void SharedCpuMatrix::add(real p1, real p2) {
  CHECK_EQ(blockNum_, 1);
  std::lock_guard<std::mutex> guard(*blockLocks_[0]);
  CpuMatrix::add(p1, p2);
}

void SharedCpuMatrix::initShared(int blockNum) {
  CHECK_GT(height_ * width_, 1UL * 1024 * 1024)
      << "should not share small matrix";
  initBlock(blockNum);
}

void SharedCpuMatrix::initBlock(int blockNum) {
  CHECK_LE(blockNum, 200) << "should not use large block number";
  blockNum_ = blockNum;
  blockLocks_.resize(blockNum);
  for (auto& locker : blockLocks_) {
    locker.reset(new std::mutex);
  }
}

/* Add a (column) vector b to matrix a, column by column */
void CpuMatrix::addColumnVector(const Matrix& b) {
  BaseMatrix::addColVector(const_cast<Matrix&>(b));
}

/* this = a*b */
void CpuMatrix::mul(const Matrix& a, const Matrix& b) {
  return mul(a, b, 1.0, 0.0);
}

/* this = scaleAB*(this*b) +  scaleT*this */
void CpuMatrix::rightMul(Matrix& b, real scaleAB, real scaleT) {
  (void)b;
  (void)scaleAB;
  (void)scaleT;
  LOG(FATAL) << "Not implemented";
}

/* this = this* b */
void CpuMatrix::rightMul(Matrix& b) { return rightMul(b, 1.0, 0.0); }

/* this = scaleAB*(a*this) +  scaleT*this */
void CpuMatrix::leftMul(Matrix& a, real scaleAB, real scaleT) {
  (void)a;
  (void)scaleAB;
  (void)scaleT;
  LOG(FATAL) << "Not implemented";
}

/* this = a*this) */
void CpuMatrix::leftMul(Matrix& a) { return leftMul(a, 1.0, 0.0); }

void CpuMatrix::colMerge(Matrix& src) { src.rowSum(*this); }

void CpuMatrix::rowSum(Matrix& sum) {
  CHECK_EQ(sum.getHeight(), getHeight());
  CHECK_EQ(sum.getWidth(), (size_t)1);

  sum.sumRows(*this, /* scaleSum= */ 1, /* scaleDest= */ 0);
}

void CpuMatrix::rowMaxId(IVector& maxIds) {
  CHECK(!maxIds.useGpu()) << "Matrix type are not equal";

  size_t numSamples = getHeight();
  CHECK_EQ(maxIds.getSize(), numSamples);

  real* a = getData();
  int* s = maxIds.getData();
  size_t dim = getWidth();

  for (size_t i = 0; i < numSamples; i++) {
    real sm = a[i * dim];
    int maxId = 0;
    for (size_t j = 1; j < dim; j++) {
      if (a[i * dim + j] > sm) {
        maxId = j;
        sm = a[i * dim + j];
      }
    }
    s[i] = maxId;
  }
}

void CpuMatrix::rowMax(Matrix& max) {
  CHECK_EQ(max.getHeight(), getHeight());
  CHECK_EQ(max.getWidth(), (size_t)1);
  max.maxRows(*this);
}

/* Get the top k elements of each row of this matrix */
void CpuMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
  CHECK(isContiguous());
  CHECK(!maxIds.useGpu() && !maxVal.useGpu()) << "Matrix type are not equal";
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  CHECK_EQ(maxVal.getWidth(), beam);

  real* a = getData();
  int* s = maxIds.getData();
  real* t = maxVal.getData();
  size_t dim = getWidth();
  for (size_t i = 0; i < numSamples; i++) {
    std::vector<std::pair<real, size_t>> vec;
    for (size_t j = 0; j < dim; j++) {
      vec.push_back(std::pair<real, size_t>(a[i * dim + j], j));
    }

    std::partial_sort(
        vec.begin(),
        vec.begin() + beam,
        vec.end(),
        [](const std::pair<real, size_t>& l, const std::pair<real, size_t>& r) {
          return l.first > r.first;
        });
    for (size_t j = 0; j < beam; j++) {
      t[i * beam + j] = vec[j].first;
      s[i * beam + j] = vec[j].second;
    }
  }
}

void CpuMatrix::colMax(Matrix& max) {
  CHECK_EQ(max.getWidth(), getWidth());
  CHECK_EQ(max.getHeight(), (size_t)1);
  max.maxCols(*this);
}

void CpuMatrix::colMax(IVector& maxIds, Matrix& maxVal) {
  CHECK(isContiguous());
  CHECK(!maxIds.useGpu() && !maxVal.useGpu()) << "Matrix type are not equal";
  size_t numSamples = getWidth();
  size_t beam = maxVal.getHeight();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getWidth(), numSamples);

  real* a = getData();
  int* s = maxIds.getData();
  real* t = maxVal.getData();
  size_t dim = getHeight();
  for (size_t i = 0; i < numSamples; i++) {
    std::vector<std::pair<real, size_t>> vec;
    for (size_t j = 0; j < dim; j++) {
      vec.push_back(std::pair<real, size_t>(a[i + j * numSamples], j));
    }

    std::partial_sort(
        vec.begin(),
        vec.begin() + beam,
        vec.end(),
        [](const std::pair<real, size_t>& l, const std::pair<real, size_t>& r) {
          return l.first > r.first;
        });
    for (size_t j = 0; j < beam; j++) {
      t[i + j * numSamples] = vec[j].first;
      s[i + j * numSamples] = vec[j].second;
    }
  }
}

void CpuMatrix::maxoutForward(Matrix& a,
                              IVector& id,
                              size_t channels,
                              size_t groups) {
  CHECK(dynamic_cast<CpuMatrix*>(&a));
  CHECK(dynamic_cast<CpuIVector*>(&id));
  CHECK_EQ(a.getHeight(), getHeight());

  size_t size = getWidth();
  size_t batchSize = getHeight();
  size_t featLen = size / channels;
  const real* input = a.getData();
  int* idForCpu = id.getData();

  MatrixPtr maxInMat, maxOutMat;
  Matrix::resizeOrCreate(maxInMat, groups, size, false, false);
  Matrix::resizeOrCreate(maxOutMat, 1, size, false, false);

  for (size_t batch_idx = 0; batch_idx < batchSize; ++batch_idx) {
    size_t newIndex = batch_idx * size;
    IVectorPtr tmpId = IVector::create(idForCpu + newIndex, size, false);

    for (size_t i = 0; i < channels; ++i) {
      size_t newFeatLen = i * featLen;
      for (size_t j = 0; j < groups; ++j) {
        maxInMat->subMatrix(j, j + 1, newFeatLen, newFeatLen + featLen)
            ->copyFrom(input + (newIndex + newFeatLen) * groups + j * featLen,
                       featLen);
      }
    }
    maxInMat->colMax(*tmpId, *maxOutMat);
    this->subRowMatrix(batch_idx, batch_idx + 1)->copyFrom(*maxOutMat);
  }
}

void CpuMatrix::maxoutBackward(Matrix& a,
                               IVector& id,
                               size_t channels,
                               size_t groups) {
  CHECK(dynamic_cast<CpuMatrix*>(&a));
  CHECK(dynamic_cast<CpuIVector*>(&id));
  CHECK_EQ(a.getHeight(), getHeight());

  size_t size = a.getWidth();
  size_t batchSize = getHeight();
  size_t featLen = size / channels;
  size_t newFeatLen = groups * featLen;
  real* inputG = getData();
  const real* outG = a.getData();
  int* idForCpu = id.getData();

  for (size_t batch_idx = 0; batch_idx < batchSize; ++batch_idx) {
    size_t newIndex = batch_idx * size;
    int* idData = idForCpu + newIndex;

    for (size_t i = 0; i < size; ++i) {
      int gradIdx =
          idData[i] * featLen + (i / featLen) * newFeatLen + i % featLen;
      (inputG + newIndex * groups)[gradIdx] += (outG + newIndex)[i];
    }
  }
}

void CpuMatrix::rowNormalizeL1(Matrix& out) {
  CHECK(!out.useGpu());

  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(out.getHeight(), numSamples);
  CHECK_EQ(out.getWidth(), dim);
  real* a = getData();
  real* b = out.getData();
  for (size_t i = 0; i < numSamples; ++i) {
    real s = 0;
    for (size_t j = 0; j < dim; ++j) {
      s += a[i * dim + j];
    }
    // Right now, we just bet that sum won't be zero. If this really happens,
    // we will figure out what should be done then.
    CHECK_GT(s, 0);
    s = 1 / s;
    for (size_t j = 0; j < dim; ++j) {
      b[i * dim + j] = s * a[i * dim + j];
    }
  }
}

/* calulate classification error */
void CpuMatrix::classificationError(Matrix& output,
                                    IVector& label,
                                    size_t topkSize) {
  size_t numSamples = this->getHeight();
  auto cpuOutput = dynamic_cast<CpuMatrix*>(&output);
  auto cpuLabel = dynamic_cast<CpuIVector*>(&label);
  IVectorPtr cpuTopIds = std::make_shared<CpuIVector>(numSamples * topkSize);
  MatrixPtr cpuTopVal = std::make_shared<CpuMatrix>(numSamples, topkSize);

  CHECK(cpuOutput && cpuLabel) << "Invalid argument pointer";
  CHECK(cpuTopIds && cpuTopVal) << "Allocate cpu memory failed";
  CHECK(cpuLabel->getSize() == numSamples) << "Vector size is not equal";
  CHECK(cpuOutput->getHeight() == numSamples && this->getWidth() == 1)
      << "Matrix dimensions are not equal";

  // top k matrix classification
  cpuOutput->rowMax(*cpuTopIds, *cpuTopVal);

  size_t dim = cpuOutput->getWidth();
  real* result = this->getData();
  int* ids = cpuTopIds->getData();
  int* lbl = cpuLabel->getData();
  for (size_t i = 0; i < numSamples; ++i) {
    CHECK_GE(lbl[i], 0);
    CHECK_LT((size_t)lbl[i], dim);

    for (size_t j = 0; j < topkSize; ++j) {
      if (ids[j + i * topkSize] == lbl[i]) {
        result[i] = 0;
        break;
      }
      result[i] = 1.0f;
    }
  }
}

/* copy -log(output[label]) to this->data[i] */
void CpuMatrix::oneHotCrossEntropy(Matrix& output, IVector& label) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  CHECK(dynamic_cast<CpuIVector*>(&label));

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(label.getSize(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(getWidth(), (size_t)1);

  real* out = output.getData();
  real* cost = getData();
  int* lbl = label.getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim) {
    CHECK_GE(lbl[i], 0);
    CHECK_LT((size_t)lbl[i], dim);
    cost[i] = -std::log(out[lbl[i]]);
  }
}

/* calculate the error of outputV according to label */
void CpuMatrix::oneHotCrossEntropyBp(Matrix& output, IVector& label) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  CHECK(dynamic_cast<CpuIVector*>(&label));
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(output.getWidth(), dim);
  real* out = output.getData();
  real* grad = getData();
  int* lbl = label.getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim, grad += dim) {
    grad[lbl[i]] -= 1 / out[lbl[i]];
  }
}

/*
    We implement the matrix functionality in CostLayer.cpp,
    but we define the scalar function here for sanity check
    deletion of the function does not affect anything neverthelss
*/
void CpuMatrix::oneHotCrossEntropyWithSelfNorm(Matrix& output,
                                               IVector& label,
                                               real alpha) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  CHECK(dynamic_cast<CpuIVector*>(&label));

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(label.getSize(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(getWidth(), (size_t)1);

  real* out = output.getData();
  real* cost = getData();
  int* lbl = label.getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim) {
    CHECK_GE(lbl[i], 0);
    CHECK_LT((size_t)lbl[i], dim);
    real sum = 0;
    for (size_t j = 0; j < dim; ++j) {
      sum += out[j];
    }
    sum = _safelog(sum);
    cost[i] = -_safelog(out[lbl[i]]) + sum + alpha * _square(sum);
  }
}

/*
    We implement the matrix functionality in CostLayer.cpp,
    but we define the scalar function here for sanity check
    deletion of the function does not affect anything neverthelss
*/
void CpuMatrix::oneHotCrossEntropyWithSelfNormBp(Matrix& output,
                                                 IVector& label,
                                                 real alpha) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  CHECK(dynamic_cast<CpuIVector*>(&label));
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(output.getWidth(), dim);
  real* out = output.getData();
  real* grad = getData();
  int* lbl = label.getData();

  for (size_t i = 0; i < numSamples; ++i, out += dim, grad += dim) {
    grad[lbl[i]] -= 1 / out[lbl[i]];
    real sum = 0;
    for (size_t j = 0; j < dim; ++j) {
      sum += out[j];
    }
    for (size_t j = 0; j < dim; ++j) {
      if (j == (size_t)lbl[i]) {
        grad[j] += -1 / out[j];
      }
      grad[j] += 1 / sum + 2 * alpha * _safelog(sum) / sum;
    }
  }
}

#define FORWARD_LOOP()                      \
  size_t numSamples = getHeight();          \
  size_t dim = getWidth();                  \
  CHECK_EQ(output.getHeight(), numSamples); \
  CHECK_EQ(output.getWidth(), dim);         \
  const real* in = getData();               \
  real* out = output.getData();             \
  for (size_t i = 0; i < numSamples; ++i, in += dim, out += dim)

#define BACKWARD_LOOP()                     \
  size_t numSamples = getHeight();          \
  size_t dim = getWidth();                  \
  CHECK_EQ(output.getHeight(), numSamples); \
  CHECK_EQ(output.getWidth(), dim);         \
  real* grad = getData();                   \
  real* out = output.getData();             \
  for (size_t i = 0; i < numSamples; ++i, grad += dim, out += dim)

void CpuMatrix::softmax(Matrix& output) {
  CHECK(!output.useGpu());

  const float THRESHOLD = -64.0;

  FORWARD_LOOP() {
    real max = -1.0e20;
    for (size_t j = 0; j < dim; ++j) {
      if (in[j] > max) {
        max = in[j];
      }
    }
    for (size_t j = 0; j < dim; ++j) {
      real a = in[j] - max;
      if (a < THRESHOLD) {
        a = THRESHOLD;
      }
      out[j] = a;
    }
    vExp(dim, out, out);

    real sum = 0;
    for (size_t j = 0; j < dim; ++j) {
      sum += out[j];
    }
    sum = 1 / sum;
    for (size_t j = 0; j < dim; ++j) {
      out[j] *= sum;
    }
  }
}

void CpuMatrix::sequenceSoftmax(Matrix& output, const IVector& index) {
  CHECK_EQ(getWidth(), 1UL);
  CHECK_EQ(output.getWidth(), 1UL);
  CHECK(isContiguous());

  MatrixPtr inTmp = Matrix::create(nullptr,
                                   /* height= */ 1,
                                   1,
                                   /* trans= */ false,
                                   false);
  MatrixPtr outTmp = Matrix::create(nullptr,
                                    /* height= */ 1,
                                    1,
                                    /* trans= */ false,
                                    false);
  size_t numSequences = index.getSize() - 1;
  auto starts = index.getData();
  for (size_t i = 0; i < numSequences; ++i) {
    size_t offset = starts[i];
    size_t size = starts[i + 1] - starts[i];
    inTmp->setData(getData() + offset, 1UL, size);
    outTmp->setData(output.getData() + offset, 1UL, size);
    inTmp->softmax(*outTmp);
  }
}

void CpuMatrix::softmaxDerivative(Matrix& output, Matrix& sftmaxSum) {
  CHECK(output.useGpu_ == false) << "Matrix type are not equal";
  CHECK_EQ(getHeight(), sftmaxSum.getHeight());

  real* sums = sftmaxSum.getData();

  BACKWARD_LOOP() {
    real sum = sums[i];
    for (size_t j = 0; j < dim; ++j) {
      grad[j] = out[j] * (grad[j] - sum);
    }
  }
}

void CpuMatrix::sumOfSquares(Matrix& output, Matrix& label) {
  CHECK(output.useGpu_ == false && label.useGpu_ == false)
      << "Matrix type are not equal";

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(label.getHeight(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(label.getWidth(), dim);
  CHECK_EQ(getWidth(), (size_t)1);
  real* out = output.getData();
  real* cost = getData();

  auto labelptr = dynamic_cast<CpuSparseMatrix*>(&label);
  if (labelptr) {
    // it is a CpuSparseMatrix
    if (labelptr->getFormat() == SPARSE_CSR) {
      // treat label as a SparseMatrix
      for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          cost[i] += _square(out[i * dim + j]);
        }
      }
      if (labelptr->getValueType() == NO_VALUE) {
        int* cols = labelptr->getCols();
        for (size_t i = 0; i < numSamples; ++i) {
          for (size_t j = labelptr->getRowStartIdx(i);
               j < labelptr->getRowStartIdx(i + 1);
               ++j) {
            cost[i] += 1.0 - 2.0 * out[i * dim + cols[j]];
            /*
             * explanation of above line: original codes are follows:
             * cost[i] -= _square(out[i * dim + feature.col]);
             * cost[i] += _square(1.0 - out[i * dim + feature.col]);
             */
          }
        }
      } else if (labelptr->getValueType() == FLOAT_VALUE) {
        int* cols = labelptr->getCols();
        real* values = labelptr->getValue();
        for (size_t i = 0; i < numSamples; ++i) {
          real sum1 = 0;
          real sum2 = 0;
          for (size_t j = labelptr->getRowStartIdx(i);
               j < labelptr->getRowStartIdx(i + 1);
               ++j) {
            sum1 += values[j] * values[j];
            sum2 += values[j] * out[i * dim + cols[j]];
            /*
             * explanation of above line: original codes are follows:
             * cost[i] -= _square(out[i * dim + feature.col]);
             * cost[i] += _square(value.col - out[i * dim + feature.col]);
             */
          }
          cost[i] += sum1 - 2.0 * sum2;
        }
      } else {
        LOG(FATAL) << "unsupported sparse matrix value type in sumOfSquares";
        return;
      }
      return;
    } else {
      LOG(FATAL) << "unsupported sparse matrix format in sumOfSquares";
      return;
    }
  }

  BaseMatrix::sumOfSquaredDiffs(output,
                                label,
                                /* scaleSum= */ 1,
                                /* scaleDest= */ 1);
}

/* calculate the error of outputV according to label */
void CpuMatrix::sumOfSquaresBp(Matrix& output, Matrix& label) {
  CHECK(output.useGpu_ == false && label.useGpu_ == false)
      << "Matrix type are not equal";

  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(output.getWidth(), dim);
  CHECK_EQ(label.getWidth(), dim);

  real* out = output.getData();
  real* grad = getData();

  auto labelptr = dynamic_cast<CpuSparseMatrix*>(&label);
  if (labelptr) {
    // it is a CpuSparseMatrix
    if (labelptr->getFormat() == SPARSE_CSR) {
      // treat label as a SparseMatrix
      for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < dim; ++j) {
          grad[i * dim + j] += 2.0 * out[i * dim + j];
        }
      }
      if (labelptr->getValueType() == NO_VALUE) {
        int* cols = labelptr->getCols();
        for (size_t i = 0; i < numSamples; ++i) {
          for (size_t j = labelptr->getRowStartIdx(i);
               j < labelptr->getRowStartIdx(i + 1);
               ++j) {
            grad[i * dim + cols[j]] -= 2.0;
            /*
             * explanation of above line: original codes are follows:
             * grad[i * dim + feature.col] -= 2.0 * out[i * dim + feature.col];
             * grad[i * dim + feature.col] += 2.0 * (out[i * dim + feature.col]
             * - 1);
             */
          }
        }
      } else if (labelptr->getValueType() == FLOAT_VALUE) {
        int* cols = labelptr->getCols();
        real* values = labelptr->getValue();
        for (size_t i = 0; i < numSamples; ++i) {
          for (size_t j = labelptr->getRowStartIdx(i);
               j < labelptr->getRowStartIdx(i + 1);
               ++j) {
            grad[i * dim + cols[j]] -= 2.0 * values[j];
            /*
             * explanation of above line: original codes are follows:
             * grad[i * dim + feature.col] -= 2.0 * out[i * dim + feature.col];
             * grad[i * dim + feature.col] += 2.0 * (out[i * dim + feature.col]
             * - value.col);
             */
          }
        }
      } else {
        LOG(FATAL) << "unsupported sparse matrix value type in sumOfSquares";
        return;
      }
      return;
    } else {
      LOG(FATAL) << "unsupported sparse matrix format in sumOfSquares";
      return;
    }
  }

  real* lbl = label.getData();
  size_t ld = getStride();
  size_t outLd = output.getStride();
  size_t lblLd = label.getStride();
  CHECK(lbl);
  for (size_t i = 0; i < numSamples;
       ++i, out += outLd, lbl += lblLd, grad += ld) {
    for (size_t j = 0; j < dim; ++j) {
      grad[j] += 2.0 * (out[j] - lbl[j]);  // positive gradient;
    }
  }
}

void CpuMatrix::smoothL1(Matrix& output, Matrix& label, real destScale) {
  CHECK(output.useGpu_ == false && label.useGpu_ == false)
      << "Matrix type are not equal";

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(label.getHeight(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(label.getWidth(), dim);
  CHECK_EQ(getWidth(), (size_t)1);

  real* cost = getData();
  real* out = output.getData();
  real* lbl = label.getData();

  for (size_t i = 0; i < numSamples; ++i, out += dim, lbl += dim) {
    for (size_t j = 0; j < dim; ++j) {
      real absVal = std::fabs(out[j] - lbl[j]);
      cost[i] *= destScale;
      if (absVal < 1.0)
        cost[i] += 0.5 * absVal * absVal;
      else
        cost[i] += absVal - 0.5;
    }
  }
}

void CpuMatrix::smoothL1Bp(Matrix& output, Matrix& label, real destScale) {
  CHECK(output.useGpu_ == false && label.useGpu_ == false)
      << "Matrix type are not equal";

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(label.getHeight(), numSamples);
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(label.getWidth(), dim);
  CHECK_EQ(getWidth(), dim);

  real* out = output.getData();
  real* lbl = label.getData();
  real* grad = getData();

  for (size_t i = 0; i < numSamples; ++i, out += dim, grad += dim, lbl += dim) {
    for (size_t j = 0; j < dim; ++j) {
      real val = out[j] - lbl[j];
      grad[j] *= destScale;
      if (std::fabs(val) < 1) {
        grad[j] += val;
      } else {
        grad[j] += (real(0) < val) - (val < real(0));
      }
    }
  }
}

void CpuMatrix::tanh(Matrix& output) {
  CHECK(isContiguous());
  CHECK(output.isContiguous());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(output.getWidth(), dim);
  vTanh(numSamples * dim, getData(), output.getData());
}

void CpuMatrix::tanhDerivative(Matrix& output) {
  BaseMatrix::tanhDerivative(output);
}

void CpuMatrix::softrelu(Matrix& output) {
  CHECK(isContiguous());
  CHECK(output.isContiguous());
  const real THRESHOLD = 40.0;
  FORWARD_LOOP() {  // TODO(yuyang18): SIMD it?
    for (size_t j = 0; j < dim; ++j) {
      real x = in[j];
      if (x > THRESHOLD) {
        x = THRESHOLD;
      } else if (x < -THRESHOLD) {
        x = -THRESHOLD;
      }
      out[j] = x;
    }
  }
  vExp(numSamples * dim, output.getData(), output.getData());
  vLog1p(numSamples * dim, output.getData(), output.getData());
}

void CpuMatrix::softreluDerivative(Matrix& output) {
  CHECK(isContiguous());
  CHECK(output.isContiguous());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  size_t size = numSamples * dim;
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(output.getWidth(), dim);
  real* grad = getData();
  MatrixPtr tmpMat = Matrix::create(numSamples, dim);
  real* tmp = tmpMat->getData();

  vExp(size, output.getData(), tmpMat->getData());

  for (size_t i = 0; i < size; ++i) {
    grad[i] *= (1.0 - 1.0 / tmp[i]);
  }
}

void CpuMatrix::scaledTanh(Matrix& output, real p1, real p2) {
  CHECK(isContiguous());
  CHECK(output.isContiguous());
  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(output.getHeight(), numSamples);
  CHECK_EQ(output.getWidth(), dim);

  const real* in = getData();
  real* out = output.getData();

  // out = p2*in
  for (size_t i = 0; i < numSamples * dim; ++i) {
    out[i] = p2 * in[i];
  }

  vTanh(numSamples * dim, out, out);

  // out = p1 * out
  for (size_t i = 0; i < numSamples * dim; ++i) {
    out[i] = p1 * out[i];
  }
}

/* uniform randomization, minimize precision = 1e-5 */
void CpuMatrix::randomizeUniform() {
  CHECK(isContiguous());
  real* data = getData();
  unsigned int* randSeed = ThreadLocalRand::getSeed();
  real recipRandMax = 1.0f / (real)RAND_MAX;
  for (size_t i = 0; i < elementCnt_; ++i) {
    *data++ = rand_r(randSeed) * recipRandMax;
  }
}

void CpuMatrix::print(std::ostream& os) const {
  CHECK(isContiguous());
  for (size_t i = 0; i < height_; ++i) {
    for (size_t j = 0; j < width_; ++j) {
      os << data_[i * width_ + j] << " ";
    }
    os << std::endl;
  }
}

void CpuMatrix::paramReluForward(Matrix& data, Matrix& W) {
  real* input = data.getData();
  real* w = W.getData();
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = W.getHeight() * W.getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  for (size_t n = 0, k = 0; n < numSamples; ++n) {
    for (size_t i = 0; i < numElements; ++i, ++k) {
      data_[k] = input[k] > 0 ? input[k] : input[k] * w[i / partial_sum];
    }
  }
}

void CpuMatrix::paramReluBackwardW(Matrix& oGrad, Matrix& data) {
  real* ograd = oGrad.getData();
  real* input = data.getData();
  real* wgrad = data_;
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = this->getHeight() * this->getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  for (size_t n = 0, k = 0; n < numSamples; ++n) {
    for (size_t i = 0; i < numElements; ++i, ++k) {
      wgrad[i / partial_sum] += ograd[k] * (input[k] > 0 ? 0 : input[k]);
    }
  }
}

void CpuMatrix::paramReluBackwardDiff(Matrix& oGrad, Matrix& data, Matrix& W) {
  real* diff = data_;
  real* input = data.getData();
  real* ograd = oGrad.getData();
  real* w = W.getData();
  size_t numElements = data.getWidth();
  size_t numSamples = data.getHeight();
  size_t paraSize = W.getHeight() * W.getWidth();
  CHECK(!(numElements % paraSize));  // this check from ParameterReluLayer::init
  size_t partial_sum = numElements / paraSize;
  for (size_t n = 0, k = 0; n < numSamples; ++n) {
    for (size_t i = 0; i < numElements; ++i, ++k) {
      diff[k] += ograd[k] * (input[k] > 0 ? 1 : w[i / partial_sum]);
    }
  }
}

void CpuMatrix::print(std::ostream& os, size_t height, size_t width) const {
  CHECK(isContiguous());
  size_t h = height_ < height ? height_ : height;
  size_t w = width_ < width ? width_ : width;
  os.setf(std::ostream::scientific);
  os << "[";
  for (size_t i = 0; i < h; ++i) {
    for (size_t j = 0; j < w; ++j) {
      os << data_[i * width_ + j] << " ";
    }
    if (i == h - 1) {
      os << "]";
    }
    os << std::endl;
  }
}

void CpuMatrix::printOneRow(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, height_);
  size_t offset = idx * stride_;
  os << data_[offset];
  for (size_t i = 1; i < width_; ++i) {
    os << " " << data_[offset + i];
  }
  os << ";";
}

void CpuMatrix::check(std::ostream& os, Matrix& refMat, bool printDiff) {
  CHECK(isContiguous());
  CHECK(height_ == refMat.getHeight());
  CHECK(width_ == refMat.getWidth());
  CpuMatrix cpuRef(height_, width_);
  cpuRef.copyFrom(refMat);
  size_t diffCnt = 0;
  for (size_t i = 0; i < height_; ++i) {
    for (size_t j = 0; j < width_; ++j) {
      real a = getElement(i, j);
      real b = cpuRef.getElement(i, j);
      if (fabs(a - b) > 0.00001) {
        ++diffCnt;
        if (printDiff) {
          os << "ref= " << a << "  check= " << b << std::endl;
        }
      }
    }
  }
  LOG(INFO) << "the  diffCnt is " << diffCnt;
}

real CpuMatrix::getMin() {
  size_t size = getHeight() * getWidth();
  real* data = getData();
  real res = data[0];
  for (size_t i = 1; i < size; ++i) {
    if (res > data[i]) {
      res = data[i];
    }
  }
  return res;
}

real CpuMatrix::getMax() {
  size_t size = getHeight() * getWidth();
  real* data = getData();
  real res = data[0];
  for (size_t i = 1; i < size; ++i) {
    if (res < data[i]) {
      res = data[i];
    }
  }
  return res;
}

void CpuMatrix::circularConv(Matrix& in0, Matrix& in1) {
  size_t height = this->getHeight();
  size_t width0 = this->getWidth();
  size_t width1 = in1.getWidth();

  CHECK_EQ(height, in0.getHeight());
  CHECK_EQ(width0, in0.getWidth());
  CHECK_EQ(height, in1.getHeight());

  CHECK_EQ(width1 % 2, 1U);

  real* outV = this->getData();
  real* inV0 = in0.getData();
  real* inV1 = in1.getData();

  int leftCtxLen = (width1 - 1) / 2;
  for (size_t x = 0; x < height;
       ++x, outV += width0, inV0 += width0, inV1 += width1) {
    for (size_t i = 0; i < width0; ++i) {  // each dimension of output
      for (size_t j = 0; j < width1; ++j) {
        // iterate over all dimentions of inV1
        int index = i + j - leftCtxLen;
        index = (index + width0) % width0;
        outV[i] += inV0[index] * inV1[j];
      }
    }
  }
}

void CpuMatrix::circularConvDerivative(
    Matrix& outG, Matrix& in0, Matrix& in1, Matrix& inG0, Matrix& inG1) {
  size_t height = in0.getHeight();
  size_t width0 = in0.getWidth();
  size_t width1 = in1.getWidth();

  CHECK_EQ(height, in1.getHeight());
  CHECK_EQ(height, inG0.getHeight());
  CHECK_EQ(width0, inG0.getWidth());
  CHECK_EQ(height, inG1.getHeight());
  CHECK_EQ(width1, inG1.getWidth());
  CHECK_EQ(height, outG.getHeight());
  CHECK_EQ(width0, outG.getWidth());

  real* outGV = outG.getData();
  real* inV0 = in0.getData();
  real* inV1 = in1.getData();
  real* inGV0 = inG0.getData();
  real* inGV1 = inG1.getData();

  int leftCtxLen = (width1 - 1) / 2;
  for (size_t x = 0; x < height; ++x,
              outGV += width0,
              inV0 += width0,
              inV1 += width1,
              inGV0 += width0,
              inGV1 += width1) {
    for (size_t j = 0; j < width1; ++j) {  // iterate over width1
      for (size_t i = 0; i < width0; ++i) {
        // such over all dimensions of outG
        int index = i + j - leftCtxLen;
        index = (index + width0) % width0;
        inGV0[index] += outGV[i] * inV1[j];
        inGV1[j] += outGV[i] * inV0[index];
      }
    }
  }
}

void CpuMatrix::multiBinaryLabelCrossEntropy(Matrix& output, Matrix& label) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  auto labelPtr = dynamic_cast<CpuSparseMatrix*>(&label);
  CHECK(labelPtr);

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(numSamples, output.getHeight());
  CHECK_EQ(numSamples, labelPtr->getHeight());
  CHECK_EQ(dim, labelPtr->getWidth());

  real* out = output.getData();
  real* cost = getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim) {
    for (size_t j = 0; j < dim; ++j) {
      CHECK(out[j] > 0 && out[j] < 1.0);
      cost[i] -= std::log(1 - out[j]);
    }

    const int* cols = labelPtr->getRowCols(i);
    for (size_t j = 0; j < labelPtr->getColNum(i); ++j) {
      CHECK_LT(size_t(cols[j]), dim);
      cost[i] -= std::log(out[cols[j]] / (1 - out[cols[j]]));
    }
  }
}

void CpuMatrix::multiBinaryLabelCrossEntropyBp(Matrix& output, Matrix& label) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  auto labelPtr = dynamic_cast<CpuSparseMatrix*>(&label);
  CHECK(labelPtr);

  size_t numSamples = getHeight();
  size_t dim = getWidth();
  CHECK_EQ(numSamples, output.getHeight());
  CHECK_EQ(numSamples, labelPtr->getHeight());
  CHECK_EQ(dim, output.getWidth());
  CHECK_EQ(dim, labelPtr->getWidth());

  real* out = output.getData();
  real* grad = getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim, grad += dim) {
    for (size_t j = 0; j < dim; ++j) {
      CHECK(out[j] > 0 && out[j] < 1.0);
      grad[j] += 1.0 / (1 - out[j]);
    }

    const int* cols = labelPtr->getRowCols(i);
    for (size_t j = 0; j < labelPtr->getColNum(i); ++j) {
      CHECK_LT(size_t(cols[j]), dim);
      grad[cols[j]] -= 1.0 / (out[cols[j]] * (1 - out[cols[j]]));
    }
  }
}

/* calculate the classification error for multi binary label */
void CpuMatrix::classificationErrorMulti(Matrix& output,
                                         Matrix& label,
                                         real threshold) {
  CHECK(dynamic_cast<CpuMatrix*>(&output));
  auto labelPtr = dynamic_cast<CpuSparseMatrix*>(&label);
  CHECK(labelPtr);

  size_t numSamples = getHeight();
  size_t dim = output.getWidth();
  CHECK_EQ(numSamples, output.getHeight());
  CHECK_EQ(numSamples, labelPtr->getHeight());
  CHECK_EQ(dim, labelPtr->getWidth());

  real* out = output.getData();
  real* result = getData();
  for (size_t i = 0; i < numSamples; ++i, out += dim) {
    real sum = 0.0;
    for (size_t j = 0; j < dim; ++j) {
      if (out[j] >= threshold) {
        sum += 1.0;
      }
    }

    const int* cols = labelPtr->getRowCols(i);
    for (size_t j = 0; j < labelPtr->getColNum(i); ++j) {
      CHECK_LT(size_t(cols[j]), dim);
      if (out[cols[j]] < threshold) {
        sum += 1.0;
      } else {
        sum -= 1.0;
      }
    }
    result[i] = sum / dim;
  }
}

void CpuMatrix::bilinearForward(const Matrix& in,
                                const size_t inImgH,
                                const size_t inImgW,
                                const size_t outImgH,
                                const size_t outImgW,
                                const size_t numChannels,
                                const real ratioH,
                                const real ratioW) {
  CHECK(dynamic_cast<const CpuMatrix*>(&in));

  size_t outputW = getWidth();
  size_t batchSize = getHeight();
  size_t inputW = in.getWidth();
  size_t inputH = in.getHeight();
  size_t inPosOffset = inImgH * inImgW;
  size_t outPosOffset = outImgH * outImgW;
  (void)(inputH);

  real* outData = getData();
  const real* inData = in.getData();

  if (inImgH == outImgH && inImgW == outImgW) {
    this->copyFrom(in);
  } else {
    for (size_t k = 0; k < batchSize; ++k) {  // loop for batches
      for (size_t i = 0; i < outImgH; ++i) {  // loop for images
        size_t h = ratioH * i;
        size_t hid = (h < inImgH - 1) ? 1 : 0;
        real h1lambda = ratioH * i - h;
        real h2lambda = 1 - h1lambda;

        for (size_t j = 0; j < outImgW; ++j) {
          size_t w = ratioW * j;
          size_t wid = (w < inImgW - 1) ? 1 : 0;
          real w1lambda = ratioW * j - w;
          real w2lambda = 1 - w1lambda;
          // calculate four position for bilinear interpolation
          const real* inPos = &inData[k * inputW + h * inImgW + w];
          real* outPos = &outData[k * outputW + i * outImgW + j];
          for (size_t c = 0; c < numChannels; ++c) {  // loop for channels
            // bilinear interpolation
            outPos[0] =
                h2lambda * (w2lambda * inPos[0] + w1lambda * inPos[wid]) +
                h1lambda * (w2lambda * inPos[hid * inImgW] +
                            w1lambda * inPos[hid * inImgW + wid]);
            inPos += inPosOffset;
            outPos += outPosOffset;
          }
        }
      }
    }
  }
}

void CpuMatrix::bilinearBackward(const Matrix& out,
                                 const size_t outImgH,
                                 const size_t outImgW,
                                 const size_t inImgH,
                                 const size_t inImgW,
                                 const size_t numChannels,
                                 const real ratioH,
                                 const real ratioW) {
  CHECK(dynamic_cast<const CpuMatrix*>(&out));

  size_t inputW = getWidth();
  size_t inputH = getHeight();
  size_t outputW = out.getWidth();
  size_t batchSize = out.getHeight();
  size_t inPosOffset = inImgH * inImgW;
  size_t outPosOffset = outImgH * outImgW;
  (void)(inputH);

  real* inGrad = getData();
  const real* outGrad = out.getData();

  if (inImgH == outImgH && inImgW == outImgW) {
    this->add(const_cast<Matrix&>(out));
  } else {
    for (size_t k = 0; k < batchSize; ++k) {  // loop for batches
      for (size_t i = 0; i < outImgH; ++i) {  // loop for images
        size_t h = ratioH * i;
        size_t hid = (h < inImgH - 1) ? 1 : 0;
        real h1lambda = ratioH * i - h;
        real h2lambda = 1 - h1lambda;
        for (size_t j = 0; j < outImgW; ++j) {
          size_t w = ratioW * j;
          size_t wid = (w < inImgW - 1) ? 1 : 0;
          real w1lambda = ratioW * j - w;
          real w2lambda = 1 - w1lambda;

          real* inPos = &inGrad[k * inputW + h * inImgW + w];
          const real* outPos = &outGrad[k * outputW + i * outImgW + j];
          for (size_t c = 0; c < numChannels; ++c) {  // loop for channels
            inPos[0] += h2lambda * w2lambda * outPos[0];
            inPos[wid] += h2lambda * w1lambda * outPos[0];
            inPos[hid * inImgW] += h1lambda * w2lambda * outPos[0];
            inPos[hid * inImgW + wid] += h1lambda * w1lambda * outPos[0];
            inPos += inPosOffset;
            outPos += outPosOffset;
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////
//               functions executed via cpu                   //
////////////////////////////////////////////////////////////////

void GpuMatrix::selectElements(Matrix& table, IVector& ids) {
  execViaCpu2(&CpuMatrix::selectElements, *this, table, ids);
}
}  // namespace paddle
