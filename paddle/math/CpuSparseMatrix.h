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
#include <cstddef>
#include "Matrix.h"

namespace paddle {

class CpuSparseMatrix : public Matrix {
public:
  CpuSparseMatrix(size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format = SPARSE_CSR,
                  bool trans = false);

  CpuSparseMatrix(CpuMemHandlePtr memHandle,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  CpuSparseMatrix(real* data,
                  int* rows,
                  int* cols,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  ~CpuSparseMatrix() {}

  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format);
  void resize(size_t newHeight, size_t newWidth);

  MatrixPtr getTranspose();

  SparseValueType getValueType();

  real* getRowValues(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return value_ + rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  int* getRowCols(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return cols_ + rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  /// fill row indices of each value in CSR matrix
  void fillRowIndices(IVectorPtr& outVec) const;

  size_t getColNum(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return rows_[i + 1] - rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  real* getColumn(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return value_ + cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  size_t getColStartIdx(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  size_t getRowStartIdx(size_t i) const {
    if (format_ == SPARSE_CSR) {
      return rows_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSC not supported";
      return 0;
    }
  }

  size_t getRowNum(size_t i) const {
    if (format_ == SPARSE_CSC) {
      return cols_[i + 1] - cols_[i];
    } else {
      LOG(FATAL) << "SPARSE_CSR not supported";
      return 0;
    }
  }

  virtual real getSum() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return elementCnt_;
    }
    double sum = 0;
    for (size_t i = 0; i < elementCnt_; ++i) {
      sum += value_[i];
    }
    return sum;
  }

  virtual void square2() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return;
    }
    for (size_t i = 0; i < elementCnt_; ++i) {
      value_[i] = value_[i] * value_[i];
    }
  }

  /**
   * only consider nonzero values.
   * the actual min value should compare with 0.0.
   */
  virtual real getMin() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return (elementCnt_ > 0 ? 1.0 : 0.0);
    }
    real min = value_[0];
    for (size_t i = 1; i < elementCnt_; ++i) {
      min = value_[i] < min ? value_[i] : min;
    }
    return min;
  }

  /**
   * only consider nonzero values.
   * the actual max value should compare with 0.0.
   */
  virtual real getMax() {
    CHECK(isContiguous());
    if (valueType_ == NO_VALUE) {
      return (elementCnt_ > 0 ? 1.0 : 0.0);
    }
    real max = value_[0];
    for (size_t i = 1; i < elementCnt_; ++i) {
      max = value_[i] > max ? value_[i] : max;
    }
    return max;
  }

  void rowMax(IVector& maxIds, Matrix& maxVal);
  int* getRows() const { return rows_; }
  int* getCols() const { return cols_; }
  real* getValue() const { return value_; }
  SparseFormat getFormat() const { return format_; }
  SparseValueType getValueType() const { return valueType_; }

  /**
   * @brief return value_ of sparse matrix
   *
   * Some times CpuSparseMatrix maybe Matrix,
   * if getValue, must dynamic_cast to CpuSparseMatrix,
   * getData is convenient to get value
   */
  real* getData() { return getValue(); }
  const real* getData() const { return getValue(); }

  /**
   * @brief only set value_ of FLOAT_VALUE sparse matrix to zero
   */
  void zeroMem();

  /// mem MUST be alloced outside (memAlloc=false)
  void transpose(MatrixPtr& matTrans, bool memAlloc);

  void mul(const Matrix& A, const Matrix& B, real alpha, real beta);

  /**
   * @brief sparseMatrix += denseMatrix
   *
   *  Named add3 just because add/add2 has been used in BaseMatrix.cu
   *  and they are not virtual function.
   *
   *  Only add value of same (row, col) index in dense matrix
   *  and do not use others values whoes postions are not in sparse matirx.
   *
   * @param[in]  b   dense matrix
   */
  void add3(CpuMatrix* b);
  void add3(MatrixPtr b);

  /**
   * @brief sparseMatrix[i,j] += bias[j], (j is the col index of sparse matrix)
   *
   * @param[in]  b      bias, dense matrix and height = 1
   * @param[in]  scale  scale of b
   */
  void addBias(Matrix& b, real scale);

  void print(std::ostream& os) const;

  void printOneRow(std::ostream& os, size_t idx) const;

  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values);

  void randomizeUniform();

  void copyFrom(const GpuSparseMatrix& src, hl_stream_t stream);

  void copyFrom(const Matrix& src, hl_stream_t stream = HPPL_STREAM_DEFAULT);

  void copyFrom(const Matrix& src);

  /**
   * Get a temporary matrix. This is threadsafe. It should be only used
   * temporarily, i.e. do not store it or use it as return value.
   *
   * @note  Do NOT use large amount of tmp matrix.
   */
  CpuSparseMatrixPtr getTmpSparseMatrix(size_t height, size_t width);

  virtual MatrixPtr subMatrix(size_t startRow, size_t numRows);

  void copyFrom(std::vector<int>& rows,
                std::vector<int>& cols,
                std::vector<real>& values);

  void copyFrom(const CpuMatrix& src);

  void copyFrom(const CpuSparseMatrix& src);

  // trim the large size
  void trimFrom(const CpuSparseMatrix& src);

  void copyRow(int offsets, size_t colNum, const sparse_non_value_t* row);

  void copyRow(int offsets, size_t colNum, const sparse_float_value_t* row);

  template <class T>
  void copyFrom(int64_t* ids, int64_t* indices, T* data);

  template <class T>
  void copyFrom(int64_t* indices, T* data);

  void copyFrom(const real* data, size_t len) {
    LOG(FATAL) << "not supported!";
  }

private:
  MatrixPtr clone(size_t height = 0, size_t width = 0, bool useGpu = false);

protected:
  void sparseResize();
  /*for csr , record row start position, for csc, record row index for every no
   * zero value*/
  int* rows_;
  /*for csc , record col start position, for csr, record col index for every no
   * zero value*/
  int* cols_;
  real* value_;               /*nonzero value*/
  SparseFormat format_;       /* matrix format */
  SparseValueType valueType_; /*with value or not  */
  static const size_t DEFAULT_AVG_WIDTH = 20;

  static ThreadLocal<std::vector<CpuSparseMatrixPtr>> cpuLocalMats_;

  // BaseMatrixT interface
public:
  bool isSparse() const { return true; }

private:
  using Matrix::copyFrom;
};
}  // namespace paddle
