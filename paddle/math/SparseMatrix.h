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
#include "CpuSparseMatrix.h"
#include "Matrix.h"

namespace paddle {

typedef std::shared_ptr<_hl_sparse_matrix_s> hl_sparse_matrix_s_ptr;

class GpuSparseMatrix : public Matrix {
public:
  MemoryHandlePtr sMemoryHandle_;
  int* rows_;
  int* cols_;
  real* value_;
  const char* end_; /* point to the end of sMemoryHandle_ */

  hl_sparse_matrix_s_ptr sMatrix_;
  SparseValueType valueType_;
  SparseFormat format_;

public:
  GpuSparseMatrix(size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format_ = SPARSE_CSR,
                  bool trans = false);

  GpuSparseMatrix(GpuMemHandlePtr dataHandle,
                  hl_sparse_matrix_s_ptr sMatrix,
                  size_t height,
                  size_t width,
                  size_t nnz, /* used to allocate space */
                  SparseValueType valueType = FLOAT_VALUE,
                  SparseFormat format_ = SPARSE_CSR,
                  bool trans = false,
                  MemoryHandlePtr sMemoryHandle = NULL);

  GpuSparseMatrix(real* value,
                  int* rows,
                  int* cols,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans);

  GpuSparseMatrix(hl_sparse_matrix_s_ptr sMatrix,
                  size_t height,
                  size_t width,
                  size_t nnz,
                  SparseValueType valueType,
                  SparseFormat format,
                  bool trans,
                  MemoryHandlePtr sMemoryHandle);

protected:
  struct Element {
    int row;
    int col;
    real val;
    Element(int rowIn, int colIn, real valIn)
        : row(rowIn), col(colIn), val(valIn) {}
  };

public:
  ~GpuSparseMatrix() {}

  void resize(size_t newHeight,
              size_t newWidth,
              size_t newNnz, /* used to allocate space */
              SparseValueType valueType,
              SparseFormat format);

  void resize(size_t newHeight, size_t newWidth);

  void sparseResizeCSR();

  void sparseResizeCSC();

  void resizeCSR(size_t newHeight,
                 size_t newWidth,
                 size_t newNnz,
                 SparseValueType valueType);

  void resizeCSC(size_t newHeight,
                 size_t newWidth,
                 size_t newNnz,
                 SparseValueType valueType);

  void mul(const GpuMatrix& a, const GpuMatrix& b, real scaleAB, real scaleT);
  /// B = A , B.trans = !A.trans
  MatrixPtr getTranspose();

  /// B = A'
  void transpose(MatrixPtr& matTrans, bool memAlloc);

  void copyFrom(const Matrix& src);
  void copyFrom(const Matrix& src, hl_stream_t stream);
  void copyFromCSR(CpuSparseMatrix& src, hl_stream_t stream);
  void copyFromCSC(CpuSparseMatrix& src, hl_stream_t stream);

  void copyFrom(const IVector& src) { LOG(FATAL) << "not implemented"; }
  void copyFrom(const IVector& src, hl_stream_t stream) {
    LOG(FATAL) << "not implemented";
  }

  template <class T>
  void copyFrom(int64_t* ids, int64_t* indices, T* data, hl_stream_t stream);

  void setRow(size_t row,
              size_t colNum,
              const unsigned int* cols,
              const real* values);
  SparseValueType getValueType() const;
  SparseFormat getFormat() const { return format_; }

  const int* getRowCols(size_t x) const { return cols_ + rows_[x]; }
  const real* getRowValues(size_t x) const { return value_ + rows_[x]; }
  size_t getColNum(size_t x) const { return rows_[x + 1] - rows_[x]; }
  void print(std::ostream& os) const;

  /**
   * @brief only set value_ of FLOAT_VALUE sparse matrix to zero
   */
  void zeroMem();

  /**
   * @brief sparseMatrix += denseMatrix
   *
   * Named add3 just because add/add2 has been used in BaseMatrix.cu
   * and they are not virtual function.
   *
   * Only add value of same (row, col) index in dense matrix
   * and do not use others values.
   *
   * @param[in]  b   dense matrix
   */
  void add3(GpuMatrix* b);
  void add3(MatrixPtr b);

  /**
   * @brief sparseMatrix[i,j] += bias[j], (j is the col index of sparse matrix)
   *
   * @param[in]  b      bias, dense matrix and height = 1
   * @param[in]  scale  scale of b
   */
  void addBias(Matrix& b, real scale);

  /**
   * @brief return rows, which is gpu address
   */
  int* getRows() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_rows(sMatrix_.get());
  }

  /**
   * @brief return cols, which is gpu address
   */
  int* getCols() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_cols(sMatrix_.get());
  }

  /**
   * @brief return value, which is gpu address
   */
  real* getValue() const {
    CHECK(sMatrix_.get()) << "sMatrix_ is NULL";
    return hl_sparse_matrix_get_value(sMatrix_.get());
  }

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
   * @brief  Get top k value of each row in sparse matrix.
   *
   * Store the value in maxVal and theirs index in maxIds.
   * k = maxVal.width
   *
   * @param[out]  maxIds    index of top k
   * @param[out]  maxVal    value of top k
   */
  void rowMax(IVector& maxIds, Matrix& maxVal);

protected:
  void sparseResize();

  void copyRow(int offsets, size_t colNum, const sparse_non_value_t* row);
  void copyRow(int offsets, size_t colNum, const sparse_float_value_t* row);

public:
  void mul(const Matrix& a, const Matrix& b, real scaleAB, real scaleT);

  void copyFrom(CpuSparseMatrix& src, hl_stream_t stream);
  void copyFrom(GpuSparseMatrix& src, hl_stream_t stream);

  void trimFrom(const CpuSparseMatrix& src);
  void trimFromCSR(const CpuSparseMatrix& src);
  void trimFromCSC(const CpuSparseMatrix& src);

  // BaseMatrixT interface
public:
  bool isSparse() const { return true; }

private:
  using Matrix::mul;
  using Matrix::copyFrom;
};

}  // namespace paddle
