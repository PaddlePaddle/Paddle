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

#include "CpuSparseMatrix.h"
#include "SparseMatrix.h"
#include "float.h"
#include "hl_gpu.h"
#include "paddle/math/MathUtils.h"
#include "paddle/utils/Util.h"

namespace paddle {

const size_t CpuSparseMatrix::DEFAULT_AVG_WIDTH;

CpuSparseMatrix::CpuSparseMatrix(size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, false) {
  resize(height, width, nnz, valueType, format);
}

CpuSparseMatrix::CpuSparseMatrix(CpuMemHandlePtr dataHandle,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(dataHandle, height, width, trans, false) {
  resize(height, width, nnz, valueType, format);
}

CpuSparseMatrix::CpuSparseMatrix(real* data,
                                 int* rows,
                                 int* cols,
                                 size_t height,
                                 size_t width,
                                 size_t nnz,
                                 SparseValueType valueType,
                                 SparseFormat format,
                                 bool trans)
    : Matrix(NULL, height, width, trans, false) {
  cols_ = cols;
  rows_ = rows;
  value_ = data;
  height_ = height;
  width_ = width;
  elementCnt_ = nnz;
  valueType_ = valueType;
  format_ = format;
}

void CpuSparseMatrix::resize(size_t newHeight,
                             size_t newWidth,
                             size_t newNnz,
                             SparseValueType valueType,
                             SparseFormat format) {
  CHECK_LE(newNnz, newHeight * newWidth);
  size_t newSize = 0;
  if (format == SPARSE_CSR) {
    newSize = (newHeight + 1) * sizeof(int) + newNnz * sizeof(int);
  } else {
    newSize = (newWidth + 1) * sizeof(int) + newNnz * sizeof(int);
  }

  if (NO_VALUE != valueType) {
    newSize += newNnz * sizeof(real);
  }

  if (NULL == memoryHandle_.get() || newSize > memoryHandle_->getSize()) {
    memoryHandle_ = std::make_shared<CpuMemoryHandle>(newSize);
  }

  height_ = newHeight;
  width_ = newWidth;
  elementCnt_ = newNnz;
  valueType_ = valueType;
  format_ = format;
  sparseResize();
}
void CpuSparseMatrix::sparseResize() {
  if (format_ == SPARSE_CSR) {
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()));
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()) +
        (height_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(memoryHandle_->getBuf()) +
          (height_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }
  } else {
    cols_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()));
    rows_ = reinterpret_cast<int*>(
        reinterpret_cast<char*>(memoryHandle_->getBuf()) +
        (width_ + 1) * sizeof(int));
    if (NO_VALUE != valueType_) {
      value_ = reinterpret_cast<real*>(
          reinterpret_cast<char*>(memoryHandle_->getBuf()) +
          (width_ + 1) * sizeof(int) + elementCnt_ * sizeof(int));
    } else {
      value_ = NULL;
    }
  }
}

void CpuSparseMatrix::resize(size_t newHeight, size_t newWidth) {
  resize(newHeight,
         newWidth,
         newHeight * std::min(DEFAULT_AVG_WIDTH, newWidth),
         valueType_,
         format_);
}

MatrixPtr CpuSparseMatrix::getTranspose() {
  if (!memoryHandle_ && !value_) {
    MatrixPtr dest(new CpuSparseMatrix(
        height_, width_, elementCnt_, valueType_, format_, true));
    return dest;
  } else if (memoryHandle_) {
    MatrixPtr dest(new CpuSparseMatrix(
        std::dynamic_pointer_cast<CpuMemoryHandle>(memoryHandle_),
        height_,
        width_,
        elementCnt_,
        valueType_,
        format_,
        true));
    return dest;
  } else if (value_) {
    MatrixPtr dest(new CpuSparseMatrix(value_,
                                       rows_,
                                       cols_,
                                       height_,
                                       width_,
                                       elementCnt_,
                                       valueType_,
                                       format_,
                                       true));
    return dest;
  } else {
    return NULL;
  }
}

SparseValueType CpuSparseMatrix::getValueType() { return valueType_; }

void CpuSparseMatrix::mul(const Matrix& a,
                          const Matrix& b,
                          real scaleAB,
                          real scaleT) {
  CHECK(!isTransposed()) << "Not supported";
  const auto a_ptr = dynamic_cast<const CpuMatrix*>(&a);
  const auto b_ptr = dynamic_cast<const CpuMatrix*>(&b);

  if (a_ptr && b_ptr) {
    CpuMatrix::mul((CpuMatrix*)a_ptr, (CpuMatrix*)b_ptr, this, scaleAB, scaleT);
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::add3(CpuMatrix* b) {
  CHECK(getFormat() != SPARSE_CSC) << "Not supported";
  CHECK(height_ == b->getHeight());
  CHECK(width_ == b->getWidth());
  real* A = getValue();
  real* B = b->getData();
  int* cols = getCols();
  for (size_t i = 0; i < height_; i++) {
    size_t start = getRowStartIdx(i);
    size_t end = getRowStartIdx(i + 1);
    for (size_t j = start; j < end; j++) {
      A[j] = B[i * width_ + cols[j]];
    }
  }
}

void CpuSparseMatrix::add3(MatrixPtr b) {
  if (dynamic_cast<CpuMatrix*>(b.get())) {
    add3(dynamic_cast<CpuMatrix*>(b.get()));
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::addBias(Matrix& b, real scale) {
  CHECK_EQ(b.getHeight(), (size_t)1);
  CHECK_EQ(width_, b.getWidth());
  real* A = getValue();
  real* B = b.getData();
  int* cols = getCols();
  size_t nnz = getElementCnt();
  for (size_t i = 0; i < nnz; i++) {
    A[i] += scale * B[cols[i]];
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

void CpuSparseMatrix::print(std::ostream& os) const {
  size_t rowSize = format_ == SPARSE_CSC ? elementCnt_ : height_ + 1;
  size_t colSize = format_ == SPARSE_CSC ? width_ + 1 : elementCnt_;
  printBuf(os, rows_, rowSize, "row");
  printBuf(os, cols_, colSize, "col");
  if (valueType_ == FLOAT_VALUE) {
    printBuf(os, value_, elementCnt_, "value");
  }
  return;
}

void CpuSparseMatrix::printOneRow(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, height_);
  if (format_ == SPARSE_CSC) {
    LOG(FATAL) << "SPARSE_CSC not supported";
    return;
  }

  const int* col = getRowCols(idx);
  size_t num = getColNum(idx);
  if (num > 0) {
    if (valueType_ == FLOAT_VALUE) {
      const real* data = getRowValues(idx);
      os << col[0] << ":" << data[0];
      for (size_t i = 1; i < num; ++i) {
        os << " " << col[i] << ":" << data[i];
      }
    } else {
      os << col[0];
      for (size_t i = 1; i < num; ++i) {
        os << " " << col[i];
      }
    }
  }
  os << ";";
}

void CpuSparseMatrix::randomizeUniform() {
  CHECK_LE(elementCnt_, height_ * width_);
  if (valueType_ == FLOAT_VALUE) {
    real* data = getValue();
    for (size_t i = 0; i < elementCnt_; ++i) {
      *data++ = rand() / static_cast<real>(RAND_MAX);  // NOLINT
    }
  }
  if (format_ == SPARSE_CSR) {
    sparseRand(rows_, cols_, elementCnt_, height_ + 1, width_, false);
  } else {
    sparseRand(cols_, rows_, elementCnt_, width_ + 1, height_, false);
  }
}

void CpuSparseMatrix::copyFrom(std::vector<int>& rows,
                               std::vector<int>& cols,
                               std::vector<real>& values) {
  size_t size = format_ == SPARSE_CSR ? cols.size() : rows.size();
  resize(height_, width_, size, valueType_, format_);
  if (valueType_ == FLOAT_VALUE) {
    memcpy(&value_[0], &values[0], sizeof(real) * values.size());
  }
  memcpy(&cols_[0], &cols[0], sizeof(int) * cols.size());
  memcpy(&rows_[0], &rows[0], sizeof(int) * rows.size());
}

// Copy from a CpuMatrix, only supported in sparse_float_value_t
// SparseMatrix.
void CpuSparseMatrix::copyFrom(const CpuMatrix& src) {
  CHECK_EQ(getHeight(), src.getHeight());
  CHECK_EQ(getWidth(), src.getWidth());
  CHECK(!src.trans_ && !trans_);
  if (format_ == SPARSE_CSR) {
    std::vector<int> rows(getHeight() + 1);
    std::vector<int> cols;
    std::vector<real> values;
    rows[0] = 0;
    for (size_t r = 0; r < getHeight(); ++r) {
      for (size_t c = 0; c < getWidth(); ++c) {
        real v = src.getElement(r, c);
        if (fabs(v) > FLT_EPSILON) {
          cols.push_back(c);
          values.push_back(v);
        }
      }
      rows[r + 1] = values.size();
    }
    copyFrom(rows, cols, values);
  } else {
    std::vector<int> cols(getWidth() + 1);
    std::vector<int> rows;
    std::vector<real> values;
    cols[0] = 0;
    for (size_t r = 0; r < getWidth(); ++r) {
      for (size_t c = 0; c < getHeight(); ++c) {
        real v = src.getElement(c, r);
        if (fabs(v) > FLT_EPSILON) {
          rows.push_back(c);
          values.push_back(v);
        }
      }
      cols[r + 1] = values.size();
    }
    copyFrom(rows, cols, values);
  }
}

MatrixPtr CpuSparseMatrix::clone(size_t height, size_t width, bool useGpu) {
  if (height == 0 && width == 0) {
    height = height_;
    width = width_;
  }
  CHECK(width && height);
  if (!useGpu) {
    return std::make_shared<CpuSparseMatrix>(
        height, width, 0, valueType_, format_);
  } else {
    return std::make_shared<GpuSparseMatrix>(
        height, width, elementCnt_, valueType_, format_);
  }
}

MatrixPtr CpuSparseMatrix::subMatrix(size_t startRow, size_t numRows) {
  CHECK_LE(startRow + numRows, height_);
  CHECK_EQ(format_, SPARSE_CSR);
  if (valueType_ == NO_VALUE) {
    return std::make_shared<CpuSparseMatrix>(
        nullptr,
        rows_ + startRow,
        cols_,
        numRows,
        width_,
        rows_[startRow + numRows] - rows_[startRow],
        valueType_,
        format_,
        trans_);
  } else {
    return std::make_shared<CpuSparseMatrix>(
        value_,
        rows_ + startRow,
        cols_,
        numRows,
        width_,
        rows_[startRow + numRows] - rows_[startRow],
        valueType_,
        format_,
        trans_);
  }
}

/* mem MUST be alloced outside (memAlloc=false) */
void CpuSparseMatrix::transpose(MatrixPtr& matTrans, bool memAlloc) {
  CHECK(!memAlloc);
  CpuSparseMatrix* mat = dynamic_cast<CpuSparseMatrix*>(matTrans.get());
  if (format_ == SPARSE_CSR) {
    /*statistic element number in each col*/
    int* colCounters = mat->getRows() + 1;
    memset(colCounters, 0, sizeof(int) * width_);
    for (size_t i = 0; i < elementCnt_; ++i) {
      int col = cols_[i];
      colCounters[col]++;
    }
    /*fill mat rows */
    mat->getRows()[0] = 0;
    for (size_t i = 1; i < width_ + 1; i++) {
      mat->getRows()[i] = mat->getRows()[i - 1] + mat->getRows()[i];
    }
    /*fill mat values and cols*/
    std::vector<int> colNumVec(width_, 0);
    if (valueType_ == FLOAT_VALUE) {
      for (size_t i = 0; i < height_; i++) {
        for (int j = rows_[i]; j < rows_[i + 1]; j++) {
          int colIdx = cols_[j];
          int index = mat->getRows()[colIdx] + colNumVec[colIdx];
          mat->getCols()[index] = i;
          mat->getValue()[index] = value_[j];
          colNumVec[colIdx]++;
        }
      }
    } else {
      for (size_t i = 0; i < height_; i++) {
        for (int j = rows_[i]; j < rows_[i + 1]; j++) {
          int colIdx = cols_[j];
          int index = mat->getRows()[colIdx] + colNumVec[colIdx];
          mat->getCols()[index] = i;
          colNumVec[colIdx]++;
        }
      }
    }
  } else {
    /*statistic element number in each row*/
    int* rowCounters = mat->getCols() + 1;
    memset(rowCounters, 0, sizeof(int) * height_);
    for (size_t i = 0; i < elementCnt_; ++i) {
      int row = rows_[i];
      rowCounters[row]++;
    }

    /*fill mat cols */
    mat->getCols()[0] = 0;
    for (size_t i = 1; i < height_ + 1; i++) {
      mat->getCols()[i] = mat->getCols()[i - 1] + mat->getCols()[i];
    }
    /*fill mat values and rows*/
    std::vector<int> rowNumVec(height_, 0);
    if (valueType_ == FLOAT_VALUE) {
      for (size_t i = 0; i < width_; i++) {
        for (int j = cols_[i]; j < cols_[i + 1]; j++) {
          int rowIdx = rows_[j];
          int index = mat->getCols()[rowIdx] + rowNumVec[rowIdx];
          mat->getRows()[index] = i;
          mat->getValue()[index] = value_[j];
          rowNumVec[rowIdx]++;
        }
      }
    } else {
      for (size_t i = 0; i < width_; i++) {
        for (int j = cols_[i]; j < cols_[i + 1]; j++) {
          int rowIdx = rows_[j];
          int index = mat->getCols()[rowIdx] + rowNumVec[rowIdx];
          mat->getRows()[index] = i;
          rowNumVec[rowIdx]++;
        }
      }
    }
  }
}

void CpuSparseMatrix::setRow(size_t row,
                             size_t colNum,
                             const unsigned int* cols,
                             const real* values) {
  if (format_ == SPARSE_CSR) {
    CHECK_LT(row, height_);
    CHECK(NULL != cols);
    if (0 == row) {
      rows_[row] = 0;
    }
    rows_[row + 1] = rows_[row] + colNum;
    for (size_t i = 0; i < colNum; ++i) {
      cols_[rows_[row] + i] = cols[i];
    }
    if (valueType_ == NO_VALUE) {
      CHECK(!values);
    } else {
      for (size_t i = 0; i < colNum; ++i) {
        value_[rows_[row] + i] = values[i];
      }
    }
  } else {
    LOG(FATAL) << "not supported";
  }
}

void CpuSparseMatrix::fillRowIndices(IVectorPtr& outVec) const {
  if (format_ == SPARSE_CSR) {
    auto nnz = getElementCnt();
    IVector::resizeOrCreate(outVec, nnz, false);
    auto out = outVec->getData();
    int* rows = getRows();
    for (size_t i = 0; i < height_; i++) {
      for (int j = rows[i]; j < rows[i + 1]; j++) {
        out[j] = i;
      }
    }
  } else {
    LOG(FATAL) << "SPARSE_CSC not supported";
  }
}

ThreadLocal<std::vector<CpuSparseMatrixPtr>> CpuSparseMatrix::cpuLocalMats_;

CpuSparseMatrixPtr CpuSparseMatrix::getTmpSparseMatrix(size_t height,
                                                       size_t width) {
  std::vector<CpuSparseMatrixPtr>* localMats = cpuLocalMats_.get();
  auto it = localMats->begin();
  while (it != localMats->end()) {
    if (it->unique()) {
      (*it)->resize(height, width, elementCnt_, valueType_, format_);
      return *it;
    }
  }
  localMats->emplace_back(std::make_shared<CpuSparseMatrix>(
      height, width, elementCnt_, valueType_, format_, false));
  return localMats->back();
}

void CpuSparseMatrix::copyFrom(const Matrix& src, hl_stream_t stream) {
  if (dynamic_cast<const GpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const GpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc, stream);
  } else if (dynamic_cast<const CpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else if (dynamic_cast<const CpuMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else {
    LOG(FATAL) << "not implemented";
  }
}

void CpuSparseMatrix::copyFrom(const Matrix& src) {
  if (dynamic_cast<const CpuSparseMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuSparseMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else if (dynamic_cast<const CpuMatrix*>(&src)) {
    auto tmpSrc = dynamic_cast<const CpuMatrix*>(&src);
    copyFrom(*tmpSrc);
  } else {
    LOG(FATAL) << "not implemented";
  }
}

void CpuSparseMatrix::copyFrom(const GpuSparseMatrix& src, hl_stream_t stream) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_EQ(width_, src.getWidth());
  CHECK_EQ(size_t(elementCnt_), src.getElementCnt());
  size_t valSize = valueType_ == NO_VALUE ? 0 : elementCnt_;
  if (format_ == SPARSE_CSC)
    hl_memcpy_from_csc_matrix(value_,
                              valSize,
                              rows_,
                              elementCnt_,
                              cols_,
                              width_ + 1,
                              src.sMatrix_.get(),
                              stream);
  else
    hl_memcpy_from_csr_matrix(value_,
                              valSize,
                              rows_,
                              height_ + 1,
                              cols_,
                              elementCnt_,
                              src.sMatrix_.get(),
                              stream);
}

void CpuSparseMatrix::copyFrom(const CpuSparseMatrix& src) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_EQ(width_, src.getWidth());
  CHECK_EQ(format_, src.getFormat());
  int start = format_ == SPARSE_CSR ? src.getRows()[0] : src.getCols()[0];
  if (format_ == SPARSE_CSR) {
    size_t totalColNum = 0;
    for (size_t i = 0; i < height_; ++i) {
      totalColNum += src.getColNum(i);
    }
    resize(height_, width_, totalColNum, valueType_, format_);
    rows_[0] = 0;
    for (size_t i = 0; i < height_; ++i) {
      rows_[i + 1] = rows_[i] + src.getColNum(i);
    }
    memcpy(cols_, src.getCols() + start, totalColNum * sizeof(int));
  } else {
    size_t totalColNum = 0;
    for (size_t i = 0; i < width_; ++i) {
      totalColNum += src.getRowNum(i);
    }
    resize(height_, width_, totalColNum, valueType_, format_);
    cols_[0] = 0;
    for (size_t i = 0; i < width_; ++i) {
      cols_[i + 1] = cols_[i] + src.getRowNum(i);
    }
    memcpy(rows_, src.getRows() + start, totalColNum * sizeof(int));
  }

  // if have different value type, only copy rows and cols
  if (valueType_ == FLOAT_VALUE && src.getValueType() == FLOAT_VALUE) {
    memcpy(value_, src.getValue() + start, elementCnt_ * sizeof(real));
  }
}

void CpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_non_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
  }
}

void CpuSparseMatrix::copyRow(int offsets,
                              size_t colNum,
                              const sparse_float_value_t* row) {
  for (size_t j = 0; j < colNum; j++) {
    cols_[offsets + j] = row[j].col;
    value_[offsets + j] = row[j].value;
  }
}

template <class T>
void CpuSparseMatrix::copyFrom(int64_t* ids, int64_t* indices, T* data) {
  size_t totalColNum = 0;
  for (size_t i = 0; i < height_; ++i) {
    int64_t id = ids[i];
    totalColNum += indices[id + 1] - indices[id];
  }
  valueType_ = typeid(T) == typeid(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE;

  resize(height_, width_, totalColNum, valueType_, format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; ++i) {
    int64_t id = ids[i];
    T* row = data + indices[id];
    size_t colNum = indices[id + 1] - indices[id];
    rows_[i + 1] = rows_[i] + colNum;
    copyRow(rows_[i], colNum, row);
  }
}

template <class T>
void CpuSparseMatrix::copyFrom(int64_t* indices, T* data) {
  CHECK(format_ == SPARSE_CSR);
  size_t totalColNum = indices[height_] - indices[0];
  valueType_ = typeid(T) == typeid(sparse_non_value_t) ? NO_VALUE : FLOAT_VALUE;
  resize(height_, width_, totalColNum, valueType_, format_);

  rows_[0] = 0;
  for (size_t i = 0; i < height_; ++i) {
    T* row = data + indices[i];
    size_t colNum = indices[i + 1] - indices[i];
    rows_[i + 1] = rows_[i] + colNum;
    copyRow(rows_[i], colNum, row);
  }
}

void CpuSparseMatrix::trimFrom(const CpuSparseMatrix& src) {
  CHECK_EQ(height_, src.getHeight());
  CHECK_LE(width_, src.getWidth());
  CHECK_EQ(format_, src.getFormat());
  CHECK_EQ(valueType_, src.getValueType());
  if (format_ == SPARSE_CSR) {
    int* srcCols = src.getCols();
    size_t numLessWidth =
        std::count_if(srcCols, srcCols + src.getElementCnt(), [this](size_t n) {
          return n < this->width_;
        });
    resize(height_, width_, numLessWidth, valueType_, format_);
    rows_[0] = 0;
    size_t index = 0;
    for (size_t r = 0; r < height_; ++r) {
      for (int i = src.getRows()[r]; i < src.getRows()[r + 1]; ++i) {
        if (srcCols[i] < static_cast<int>(width_)) {
          cols_[index] = srcCols[i];
          if (valueType_ == FLOAT_VALUE) {
            value_[index] = src.getValue()[i];
          }
          ++index;
        }
      }
      rows_[r + 1] = index;
    }
    CHECK_EQ(index, numLessWidth);
  } else {
    size_t numLessWidth = src.getCols()[width_] - src.getCols()[0];
    resize(height_, width_, numLessWidth, valueType_, format_);
    cols_[0] = 0;
    size_t index = 0;
    // note: c < width_, not src.getWidth();
    for (size_t c = 0; c < width_; ++c) {
      for (int i = src.getCols()[c]; i < src.getCols()[c + 1]; ++i) {
        rows_[index] = src.getRows()[i];
        if (valueType_ == FLOAT_VALUE) {
          value_[index] = src.getValue()[i];
        }
        ++index;
      }
      cols_[c + 1] = index;
    }
    CHECK_EQ(index, numLessWidth);
  }
}

void CpuSparseMatrix::zeroMem() {
  CHECK(valueType_ == FLOAT_VALUE);
  memset(value_, 0, elementCnt_ * sizeof(real));
}

template void CpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_non_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* ids,
                                        int64_t* indices,
                                        sparse_float_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* indices,
                                        sparse_non_value_t* data);

template void CpuSparseMatrix::copyFrom(int64_t* indices,
                                        sparse_float_value_t* data);

void CpuSparseMatrix::rowMax(IVector& maxIds, Matrix& maxVal) {
  size_t numSamples = getHeight();
  size_t beam = maxVal.getWidth();
  CHECK_EQ(maxIds.getSize(), numSamples * beam);
  CHECK_EQ(maxVal.getHeight(), numSamples);
  maxVal.zeroMem();
  int* outids = maxIds.getData();
  real* outvalues = maxVal.getData();

  typedef std::pair<real, size_t> valuepair;
  std::vector<valuepair> vec;
  for (size_t i = 0; i < numSamples; i++) {
    vec.clear();

    auto num = getColNum(i);
    auto ids = getRowCols(i);
    auto values = getRowValues(i);
    for (size_t j = 0; j < num; j++) {
      vec.push_back(std::make_pair(values[j], ids[j]));
    }

    size_t outsize = std::min(num, beam);
    std::partial_sort(vec.begin(),
                      vec.begin() + outsize,
                      vec.end(),
                      [](const valuepair& a, const valuepair& b) {
                        return a.first > b.first;
                      });
    for (size_t j = 0; j < outsize; j++) {
      outids[i * beam + j] = vec[j].second;
      outvalues[i * beam + j] = vec[j].first;
    }
    if (outsize < beam) {
      // if the number of values to sort are less than the output size,
      // use -1 to indicate the end of valid sorted values.
      outids[i * beam + outsize] = -1;
    }
  }
}

}  // namespace paddle
