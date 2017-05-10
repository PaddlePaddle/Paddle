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

#include "SparseMatrix.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include "hl_gpu.h"
#include "hl_top_k.h"
#include "paddle/utils/Util.h"

namespace paddle {

GpuSparseMatrix::GpuSparseMatrix(size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, true) {
  resize(height, width, nnz, valueType, format);
}

GpuSparseMatrix::GpuSparseMatrix(GpuMemHandlePtr dataHandle,
                                 hl_sparse_matrix_s_ptr sMatrix,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans,
                                 MemoryHandlePtr sMemoryHandle)
    : Matrix(dataHandle, height, width, trans, true) {
  CHECK(dataHandle && sMatrix) << "Invalid argument pointer";

  size_t size = 0;
  if (format == SPARSE_CSR) {
    size = (height + 1) * sizeof(int) + nnz * sizeof(int);
  } else {
    size = (width + 1) * sizeof(int) + nnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    size += nnz * sizeof(real);
  }
  CHECK_LE(size, dataHandle->getSize());

  sMatrix_ = sMatrix;

  if (sMemoryHandle == NULL) {
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(dataHandle->getSize());
  } else {
    CHECK_EQ(sMemoryHandle->getSize(), dataHandle->getSize());
    sMemoryHandle_ = sMemoryHandle;
  }

  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;
  if (format_ == SPARSE_CSR)
    sparseResizeCSR();
  else
    sparseResizeCSC();
}

GpuSparseMatrix::GpuSparseMatrix(hl_sparse_matrix_s_ptr sMatrix,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans,
                                 MemoryHandlePtr sMemoryHandle)
    : Matrix(NULL, height, width, trans, true) {
  CHECK(sMatrix) << "Invalid argument pointer";
  sMatrix_ = sMatrix;
  sMemoryHandle_ = sMemoryHandle;
  elementCnt_ = nnz;
  format_ = format;
  valueType_ = valueType;
}

GpuSparseMatrix::GpuSparseMatrix(real* value,
                                 int* rows,
                                 int* cols,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, true) {
  size_t size = 0;
  if (format == SPARSE_CSR) {
    size = (height + 1) * sizeof(int) + nnz * sizeof(int);
  } else {
    size = (width + 1) * sizeof(int) + nnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    size += nnz * sizeof(real);
  }
  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;

  sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(size);
  if (format_ == SPARSE_CSR) {
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
          (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }

    if (sMatrix_ == NULL) {
      /* construct hl_sparse_matrix_s */
      hl_sparse_matrix_s tmp;
      hl_construct_sparse_matrix(
          &tmp,
          value,
          rows,
          cols,
          HL_SPARSE_CSR,
          valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
          height_,
          width_,
          elementCnt_);
      hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
      sMatrix_ = tmp2;
    }

  } else {
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
          (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }

    if (sMatrix_ == NULL) {
      /* construct hl_sparse_matrix_s */
      hl_sparse_matrix_s tmp;
      hl_construct_sparse_matrix(
          &tmp,
          value,
          rows,
          cols,
          HL_SPARSE_CSC,
          valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
          height_,
          width_,
          elementCnt_);
      hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
      sMatrix_ = tmp2;
    }
  }
}

void GpuSparseMatrix::sparseResizeCSR() {
  rows_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
  cols_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
                             (height_ + 1) * sizeof(int));
  if (NO_VALUE != valueType_) {
    value_ = reinterpret_cast<real*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
  } else {
    value_ = NULL;
  }

  if (sMatrix_ == NULL) {
    /* construct hl_sparse_matrix_s */
    hl_sparse_matrix_s tmp;
    hl_construct_sparse_matrix(
        &tmp,
        data_,
        memoryHandle_->getSize(),
        HL_SPARSE_CSR,
        valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
        height_,
        width_,
        elementCnt_);
    hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
    sMatrix_ = tmp2;
  }
}

void GpuSparseMatrix::sparseResizeCSC() {
  cols_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()));
  rows_ =
      reinterpret_cast<int*>(reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
                             (width_ + 1) * sizeof(int));
  if (NO_VALUE != valueType_) {
    value_ = reinterpret_cast<real*>(
        reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
  } else {
    value_ = NULL;
  }

  if (sMatrix_ == NULL) {
    /* construct hl_sparse_matrix_s */
    hl_sparse_matrix_s tmp;
    hl_construct_sparse_matrix(
        &tmp,
        memoryHandle_->getBuf(),
        memoryHandle_->getSize(),
        HL_SPARSE_CSC,
        valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE,
        height_,
        width_,
        elementCnt_);
    hl_sparse_matrix_s_ptr tmp2(tmp, hl_destruct_sparse_matrix);
    sMatrix_ = tmp2;
  }
}

void GpuSparseMatrix::resize(size_t newHeight,
                             size_t newWidth,
                             size_t newNnz,
                             SparseValueType valueType,
                             SparseFormat format) {
  if (format == SPARSE_CSR) {
    resizeCSR(newHeight, newWidth, newNnz, valueType);
  } else {
    resizeCSC(newHeight, newWidth, newNnz, valueType);
  }
}

void GpuSparseMatrix::resizeCSR(size_t newHeight,
                                size_t newWidth,
                                size_t newNnz,
                                SparseValueType valueType) {
  size_t newSize = (newHeight + 1) * sizeof(int) + newNnz * sizeof(int);
  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<GpuMemoryHandle>(newSize);
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
    end_ = reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
           sMemoryHandle_->getSize();
    sMatrix_ = NULL;
  } else if (valueType != valueType_) {
    sMatrix_ = NULL;
  } else {
    /*
     * newNnz > elementCnt_ is necessary for the following condition:
     * Firstly, height_ is 9 elementCnt_ is 56
     * Secondly, height_ is 11 elementCnt_ is 44
     *   ==> height_ is bigger, sMatrix_ will resize, and total item is 44 now
     * Then, height_ is 10 elementCnt_ is 52
     *   ==> Without newNnz > elementCnt_ condition, sMatrix_ will fail
     */
    if ((ssize_t)((newHeight + 1) * sizeof(int)) >
            ((char*)cols_ - (char*)rows_) ||
        newNnz > static_cast<size_t>(sMatrix_->nnz)) {
      sMatrix_ = NULL;
    } else if (NO_VALUE == valueType) {
      if ((ssize_t)(newNnz * sizeof(int)) > (end_ - (char*)cols_)) {
        sMatrix_ = NULL;
      }
    } else {
      if ((ssize_t)(newNnz * sizeof(int)) > ((char*)value_ - (char*)cols_) ||
          (ssize_t)(newNnz * sizeof(real)) > (end_ - (char*)value_)) {
        sMatrix_ = NULL;
      }
    }
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = SPARSE_CSR;

  if (sMatrix_ == NULL) {
    sparseResizeCSR();
  }
}

void GpuSparseMatrix::resizeCSC(size_t newHeight,
                                size_t newWidth,
                                size_t newNnz,
                                SparseValueType valueType) {
  size_t newSize = (newWidth + 1) * sizeof(int) + newNnz * sizeof(int);
  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<GpuMemoryHandle>(newSize);
    data_ = reinterpret_cast<real*>(memoryHandle_->getBuf());
    sMemoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
    end_ = reinterpret_cast<char*>(sMemoryHandle_->getBuf()) +
           sMemoryHandle_->getSize();
    sMatrix_ = NULL;
  } else if (valueType != valueType_) {
    sMatrix_ = NULL;
  } else {
    /*
     * newNnz > elementCnt_ is necessary for the following condition:
     * Firstly, height_ is 9 elementCnt_ is 56
     * Secondly, height_ is 11 elementCnt_ is 44
     *   ==> height_ is bigger, sMatrix_ will resize,
     *       and total item is 44 now
     * Then, height_ is 10 elementCnt_ is 52
     *   ==> Without newNnz > elementCnt_ condition, sMatrix_ will fail
     */
    if ((ssize_t)((newWidth + 1) * sizeof(int)) >
            ((char*)rows_ - (char*)cols_) ||
        newNnz > static_cast<size_t>(sMatrix_->nnz)) {
      sMatrix_ = NULL;
    } else if (NO_VALUE == valueType) {
      if ((ssize_t)(newNnz * sizeof(int)) > (end_ - (char*)rows_)) {
        sMatrix_ = NULL;
      }
    } else {
      if ((ssize_t)(newNnz * sizeof(int)) > ((char*)value_ - (char*)rows_) ||
          (ssize_t)(newNnz * sizeof(real)) > (end_ - (char*)value_)) {
        sMatrix_ = NULL;
      }
    }
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = SPARSE_CSC;

  if (sMatrix_ == NULL) {
    sparseResizeCSC();
  }
}

void GpuSparseMatrix::resize(size_t newHeight, size_t newWidth) {
  resize(newHeight, newWidth, elementCnt_, valueType_, format_);
}

MatrixPtr GpuSparseMatrix::getTranspose() {
  CHECK(memoryHandle_.get() || sMatrix_) << "not supported";
  if (memoryHandle_.get()) {
    MatrixPtr copy_T(new GpuSparseMatrix(
        std::dynamic_pointer_cast<GpuMemoryHandle>(memoryHandle_),
        sMatrix_,
        height_,
        width_,
        elementCnt_,
        valueType_,
        format_,
        true,
        sMemoryHandle_));
    return copy_T;
  } else {
    MatrixPtr copy_T(new GpuSparseMatrix(sMatrix_,
                                         height_,
                                         width_,
                                         elementCnt_,
                                         valueType_,
                                         format_,
                                         true,
                                         sMemoryHandle_));
    return copy_T;
  }
}

void GpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_non_value_t* row) {
  memcpy(cols_ + offsets, row, sizeof(int) * colNum);
}

void GpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_float_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
    value_[offsets + j] = row[j].value;
  }
}

void GpuSparseMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  if (auto mat = dynamic_cast<const CpuSparseMatrix*>(&src)) {
    copyFrom(*(const_cast<CpuSparseMatrix*>(mat)), stream);
  } else if (auto mat = dynamic_cast<const GpuSparseMatrix*>(&src)) {
    copyFrom(*(const_cast<GpuSparseMatrix*>(mat)), stream);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

void GpuSparseMatrix::copyFrom(const Matrix& src) {
  copyFrom(src, HPPL_STREAM_1);
  hl_stream_synchronize(HPPL_STREAM_1);
}

template <class T>
void GpuSparseMatrix::copyFrom(int64_t* ids,
                               int64_t* indices,
                               T* data,
                               hl_stream_t stream) {
  CHECK_EQ(format_, SPARSE_CSR);
  size_t nnz = 0;
  for (size_t i = 0; i < height_; i++) {
    int64_t id = ids[i];
    nnz += indices[id + 1] - indices[id];
  }

  resize(height_,
         width_,
         nnz,
         sizeof(T) == sizeof(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE,
         format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; i++) {
    int64_t id = ids[i];
    size_t colNum = indices[id + 1] - indices[id];
    rows_[i + 1] = rows_[i] + colNum;

    T* row = data + indices[id];
    copyRow(rows_[i], colNum, row);
  }

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;
  hl_memcpy_csr_matrix(sMatrix_.get(), value_, rows_, cols_, stream);
}

void GpuSparseMatrix::setRow(size_t row,
                             size_t colNum,
                             const unsigned int* cols,
                             const real* values) {
  CHECK_EQ(format_, SPARSE_CSR);
  if (NO_VALUE == valueType_) {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    CHECK(NULL == values);
  } else {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    CHECK(NULL != values);
  }
  if (0 == row) {
    rows_[row] = 0;
  }
  rows_[row + 1] = rows_[row] + colNum;

  memcpy(cols_ + rows_[row], cols, sizeof(*cols) * colNum);
  if (FLOAT_VALUE == valueType_) {
    memcpy(value_ + rows_[row], values, sizeof(*values) * colNum);
  }

  if (height_ - 1 == row) {
    sMatrix_->format = HL_SPARSE_CSR;
    sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
    sMatrix_->rows = height_;
    sMatrix_->cols = width_;
    sMatrix_->nnz = elementCnt_;
    hl_memcpy_csr_matrix(
        sMatrix_.get(), value_, rows_, cols_, HPPL_STREAM_DEFAULT);
  }
}

SparseValueType GpuSparseMatrix::getValueType() const { return valueType_; }

void GpuSparseMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  CHECK_EQ(format_, SPARSE_CSC);
  int nnz = sMatrix_->nnz;
  if (memAlloc) {
    matTrans = std::make_shared<GpuSparseMatrix>(
        width_, height_, nnz, valueType_, format_, false);
  } else {
    CHECK(matTrans != nullptr);
  }

  CpuIVector rows(nnz);
  CpuIVector cols(width_ + 1);
  CpuIVector cols_full(nnz);
  CpuVector value(nnz);
  hl_stream_t stream = HPPL_STREAM_1;
  hl_memcpy_from_csc_matrix(value.getData(),
                            nnz,
                            rows.getData(),
                            nnz,
                            cols.getData(),
                            width_ + 1,
                            sMatrix_.get(),
                            stream);

  hl_stream_synchronize(stream);

  /*for every non zero number, get its column index*/
  std::vector<Element> dataVec;
  for (size_t i = 0; i < width_; i++) {
    for (int j = cols.getData()[i]; j < cols.getData()[i + 1]; j++) {
      cols_full.getData()[j] = i;
    }
  }

  /*sort row index and column index by the ascending order*/
  for (int i = 0; i < nnz; i++) {
    dataVec.emplace_back(
        rows.getData()[i], cols_full.getData()[i], value.getData()[i]);
  }
  std::sort(dataVec.begin(), dataVec.end(), [](Element a, Element b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });

  /*get sorted data, row index, and col index, put them in the right place*/
  cols.resize(height_ + 1);
  rows.resize(nnz);
  value.resize(nnz);

  cols.getData()[0] = 0;
  rows.getData()[0] = dataVec[0].col;
  value.getData()[0] = dataVec[0].val;
  for (int i = 1; i < nnz; i++) {
    if (dataVec[i].row != dataVec[i - 1].row) {
      for (int j = dataVec[i - 1].row + 1; j <= dataVec[i].row; j++) {
        cols.getData()[j] = i;
      }
    }
    rows.getData()[i] = dataVec[i].col;
    value.getData()[i] = dataVec[i].val;
  }
  cols.getData()[height_] = nnz;

  /*copy back from cpu*/
  GpuSparseMatrixPtr dest =
      std::dynamic_pointer_cast<GpuSparseMatrix>(matTrans);
  hl_memcpy_csc_matrix((dest->sMatrix_).get(),
                       value.getData(),
                       rows.getData(),
                       cols.getData(),
                       stream);
  hl_stream_synchronize(stream);
}

void GpuSparseMatrix::mul(const GpuMatrix& a,
                          const GpuMatrix& b,
                          real scaleAB,
                          real scaleT) {
  CHECK(a.useGpu_ && b.useGpu_) << "type not match";
  CHECK(!trans_) << "trans not supported";
  real* A_d = (real*)a.getData();
  real* B_d = (real*)b.getData();
  hl_sparse_matrix_s C_d = sMatrix_.get();
  hl_trans_op_t a_trans = a.trans_ ? HPPL_OP_T : HPPL_OP_N;
  hl_trans_op_t b_trans = b.trans_ ? HPPL_OP_T : HPPL_OP_N;

  if (!a.trans_ && !b.trans_) {
    CHECK(height_ == a.getHeight());
    CHECK(width_ == b.getWidth());
    CHECK(a.getWidth() == b.getHeight());
  } else if (a.trans_ && !b.trans_) {
    CHECK(height_ == a.getWidth());
    CHECK(width_ == b.getWidth());
    CHECK(a.getHeight() == b.getHeight());
  } else if (!a.trans_ && b.trans_) {
    CHECK(height_ == a.getHeight());
    CHECK(width_ == b.getHeight());
    CHECK(a.getWidth() == b.getWidth());
  } else {
    LOG(INFO) << "Not support";
  }
  int dimM = height_;
  int dimN = width_;
  int dimK = !b.trans_ ? b.getHeight() : b.getWidth();
  hl_sparse_matrix_mul(
      A_d, a_trans, B_d, b_trans, C_d, dimM, dimN, dimK, scaleAB, scaleT);
}

void GpuSparseMatrix::mul(const Matrix& a,
                          const Matrix& b,
                          real scaleAB,
                          real scaleT) {
  const auto a_ptr = dynamic_cast<const GpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const GpuMatrix*>(&b);
  if (a_ptr && b_ptr) {
    mul(*a_ptr, *b_ptr, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "not supported";
  }
}

template <class T>
void printBuf(std::ostream& os, T* a, size_t len, const char* name) {
  os << "\n: " << name << " [";
  for (size_t i = 0; i < len; i++) {
    os << a[i] << " ";
  }
  os << "]\n";
}

void GpuSparseMatrix::print(std::ostream& os) const {
  if (format_ == SPARSE_CSC) {
    int nnz = sMatrix_->nnz;
    IVectorPtr rows = IVector::create(nnz, false);
    IVectorPtr cols = IVector::create(width_ + 1, false);
    VectorPtr value = Vector::create(nnz, false);
    hl_stream_t stream = HPPL_STREAM_DEFAULT;
    hl_memcpy_from_csc_matrix(value->getData(),
                              value->getSize(),
                              rows->getData(),
                              rows->getSize(),
                              cols->getData(),
                              cols->getSize(),
                              sMatrix_.get(),
                              stream);
    hl_stream_synchronize(stream);

    printBuf(os, cols->getData(), width_ + 1, "col idx");
    printBuf(os, rows->getData(), elementCnt_, "row idx");
    printBuf(os, value->getData(), elementCnt_, "value");
  }
}

void GpuSparseMatrix::copyFromCSR(CpuSparseMatrix& src, hl_stream_t stream) {
  trans_ = src.trans_;
  size_t nnz = src.getElementCnt();

  resize(src.getHeight(), src.getWidth(), nnz, valueType_, src.getFormat());
  // if have different value type, only copy rows and cols
  SparseValueType vType =
      valueType_ != src.getValueType() ? NO_VALUE : valueType_;

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = vType == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csr_matrix(sMatrix_.get(),
                       vType == NO_VALUE ? NULL : src.getValue(),
                       src.getRows(),
                       src.getCols(),
                       stream);

  // restore type of sMatrix_
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
}

void GpuSparseMatrix::copyFromCSC(CpuSparseMatrix& src, hl_stream_t stream) {
  trans_ = src.trans_;
  size_t nnz = src.getElementCnt();

  resize(src.getHeight(), src.getWidth(), nnz, valueType_, src.getFormat());

  // if have different value type, only copy rows and cols
  SparseValueType vType =
      valueType_ != src.getValueType() ? NO_VALUE : valueType_;

  sMatrix_->format = HL_SPARSE_CSC;
  sMatrix_->type = vType == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csc_matrix(sMatrix_.get(),
                       vType == NO_VALUE ? NULL : src.getValue(),
                       src.getRows(),
                       src.getCols(),
                       stream);

  // restore type of sMatrix_
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
}

void GpuSparseMatrix::copyFrom(GpuSparseMatrix& src, hl_stream_t stream) {
  CHECK(trans_ == src.trans_);
  CHECK(format_ == src.getFormat());
  resize(src.getHeight(),
         src.getWidth(),
         elementCnt_,
         valueType_,
         src.getFormat());

  size_t rowSize = format_ == SPARSE_CSC ? elementCnt_ : height_ + 1;
  size_t colSize = format_ == SPARSE_CSC ? width_ + 1 : elementCnt_;

  if (valueType_ == FLOAT_VALUE && src.getValueType() == FLOAT_VALUE) {
    hl_memcpy_async(
        getValue(), src.getValue(), sizeof(real) * elementCnt_, stream);
  }
  CHECK(getRows());
  CHECK(src.getRows());

  hl_memcpy_async(getRows(), src.getRows(), sizeof(int) * rowSize, stream);
  hl_memcpy_async(getCols(), src.getCols(), sizeof(int) * colSize, stream);
}

void GpuSparseMatrix::copyFrom(CpuSparseMatrix& src, hl_stream_t stream) {
  if (format_ == SPARSE_CSR) {
    copyFromCSR(src, stream);
  } else {
    copyFromCSC(src, stream);
  }
}

void GpuSparseMatrix::trimFromCSR(const CpuSparseMatrix& src) {
  trans_ = src.trans_;
  int* srcCols = src.getCols();
  size_t nnz = std::count_if(srcCols,
                             srcCols + src.getElementCnt(),
                             [this](size_t n) { return n < this->width_; });
  resize(height_, width_, nnz, valueType_, format_);

  rows_[0] = 0;
  size_t index = 0;
  for (size_t r = 0; r < height_; ++r) {
    for (int i = src.getRows()[r]; i < src.getRows()[r + 1]; ++i) {
      if (srcCols[i] < (int)width_) {
        cols_[index] = srcCols[i];
        if (valueType_ == FLOAT_VALUE) {
          value_[index] = src.getValue()[i];
        }
        ++index;
      }
    }
    rows_[r + 1] = index;
  }
  CHECK_EQ(index, nnz);

  sMatrix_->format = HL_SPARSE_CSR;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csr_matrix(sMatrix_.get(),
                       valueType_ == NO_VALUE ? NULL : value_,
                       rows_,
                       cols_,
                       /*default stream = */ HPPL_STREAM_DEFAULT);
}

void GpuSparseMatrix::trimFromCSC(const CpuSparseMatrix& src) {
  trans_ = src.trans_;
  size_t nnz = src.getCols()[width_] - src.getCols()[0];
  resize(height_, width_, nnz, valueType_, format_);

  cols_[0] = 0;
  for (size_t i = 0; i < width_; i++) {
    cols_[i + 1] = cols_[i] + (int)(src.getRowNum(i));
  }
  memcpy(rows_, src.getRows() + src.getCols()[0], sizeof(int) * nnz);
  if (valueType_ == FLOAT_VALUE) {
    memcpy(value_, src.getValue() + src.getCols()[0], sizeof(real) * nnz);
  }

  sMatrix_->format = HL_SPARSE_CSC;
  sMatrix_->type = valueType_ == NO_VALUE ? HL_NO_VALUE : HL_FLOAT_VALUE;
  sMatrix_->rows = height_;
  sMatrix_->cols = width_;
  sMatrix_->nnz = nnz;

  hl_memcpy_csc_matrix(sMatrix_.get(),
                       valueType_ == NO_VALUE ? NULL : value_,
                       rows_,
                       cols_,
                       /*default stream = */ HPPL_STREAM_DEFAULT);
}

void GpuSparseMatrix::trimFrom(const CpuSparseMatrix& src) {
  if (format_ == SPARSE_CSR) {
    trimFromCSR(src);
  } else {
    trimFromCSC(src);
  }
}

void GpuSparseMatrix::addBias(Matrix& b, real scale) {
  CHECK(b.getHeight() == 1) << "the Bias should be a vector";
  hl_sparse_matrix_s A_d = sMatrix_.get();
  hl_sparse_matrix_add_bias(A_d, b.getData(), scale);
}

void GpuSparseMatrix::add3(GpuMatrix* b) {
  CHECK(getFormat() != SPARSE_CSC) << "Not supported";
  CHECK(height_ == b->getHeight());
  CHECK(width_ == b->getWidth());
  real* B_d = b->getData();
  hl_sparse_matrix_s A_d = sMatrix_.get();
  hl_sparse_matrix_add_dense(A_d, B_d, height_, width_, 1, 0);
}

void GpuSparseMatrix::add3(MatrixPtr b) {
  if (dynamic_cast<GpuMatrix*>(b.get())) {
    add3(dynamic_cast<GpuMatrix*>(b.get()));
  } else {
    LOG(FATAL) << "not supported";
  }
}

void GpuSparseMatrix::zeroMem() {
  CHECK(valueType_ == FLOAT_VALUE);
  real* value = getValue();
  if (value == NULL) {
    LOG(FATAL) << "value is nullptr";
  }
  hl_matrix_zero_mem(value, elementCnt_);
}

void GpuSparseMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
#ifndef PADDLE_ONLY_CPU
  CHECK(maxIds.useGpu() && maxVal.useGpu()) << "Matrix type are not equal";
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  CHECK_EQ(format_, SPARSE_CSR) << "Only support SPARSE_CSR";

  hl_sparse_matrix_top_k(maxVal.getData(),
                         maxVal.getStride(),
                         maxIds.getData(),
                         sMatrix_.get(),
                         beam,
                         numSamples);
#endif
}

template void GpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_non_value_t* data,
                                        hl_stream_t stream);
template void GpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_float_value_t* data,
                                        hl_stream_t stream);
}  // namespace paddle
