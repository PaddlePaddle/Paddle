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

#include "paddle/math/Matrix.h"
#include <cstring>
#include <iostream>
#include "PaddleAPI.h"
#include "paddle/math/CpuSparseMatrix.h"
#include "paddle/math/SparseMatrix.h"

struct MatrixPrivate {
  std::shared_ptr<paddle::Matrix> mat;
};

Matrix::Matrix() : m(new MatrixPrivate()) {}

Matrix* Matrix::createByPaddleMatrixPtr(void* sharedPtr) {
  auto* mat = reinterpret_cast<paddle::MatrixPtr*>(sharedPtr);
  if ((*mat) != nullptr) {
    auto m = new Matrix();
    m->m->mat = *mat;
    return m;
  } else {
    return nullptr;
  }
}

Matrix* Matrix::createZero(size_t height, size_t width, bool useGpu) {
  auto m = new Matrix();
  m->m->mat = paddle::Matrix::create(height, width, useGpu);
  m->m->mat->zero();
  return m;
}

Matrix* Matrix::createDense(const std::vector<float>& data,
                            size_t height,
                            size_t width,
                            bool useGpu) {
  auto m = new Matrix();
  m->m->mat = paddle::Matrix::create(height, width, useGpu);
  m->m->mat->copyFrom(data.data(), data.size());
  return m;
}

Matrix* Matrix::createDenseFromNumpy(float* data,
                                     int dim1,
                                     int dim2,
                                     bool copy,
                                     bool useGpu) throw(UnsupportError) {
  if (useGpu) {
    /// Gpu mode only supports copy=True
    if (!copy) {
      throw UnsupportError("Gpu mode only supports copy=True");
    }
    return Matrix::createGpuDenseFromNumpy(data, dim1, dim2);
  } else {
    return Matrix::createCpuDenseFromNumpy(data, dim1, dim2, copy);
  }
}

Matrix* Matrix::createCpuDenseFromNumpy(float* data,
                                        int dim1,
                                        int dim2,
                                        bool copy) {
  auto m = new Matrix();
  if (copy) {
    m->m->mat = paddle::Matrix::create(dim1, dim2);
    m->m->mat->copyFrom(data, dim1 * dim2);
  } else {
    m->m->mat = paddle::Matrix::create(data, dim1, dim2, false);
  }
  return m;
}

Matrix* Matrix::createGpuDenseFromNumpy(float* data, int dim1, int dim2) {
  auto m = new Matrix();
  m->m->mat = paddle::Matrix::create(dim1, dim2, false, true);
  m->m->mat->copyFrom(data, dim1 * dim2);
  return m;
}

Matrix* Matrix::createSparse(size_t height,
                             size_t width,
                             size_t nnz,
                             bool isNonVal,
                             bool isTrans,
                             bool useGpu) {
  auto m = new Matrix();
  m->m->mat = paddle::Matrix::createSparseMatrix(
      height,
      width,
      nnz,
      isNonVal ? paddle::NO_VALUE : paddle::FLOAT_VALUE,
      isTrans,
      useGpu);
  return m;
}

Matrix::~Matrix() { delete m; }

size_t Matrix::getHeight() const { return m->mat->getHeight(); }

size_t Matrix::getWidth() const { return m->mat->getWidth(); }

float Matrix::get(size_t x, size_t y) const throw(RangeError) {
  if (x > this->getWidth() || y > this->getHeight()) {
    RangeError e;
    throw e;
  }
  return m->mat->getElement(x, y);
}

void Matrix::set(size_t x, size_t y, float val) throw(RangeError,
                                                      UnsupportError) {
  if (x > this->getWidth() || y > this->getHeight()) {
    RangeError e;
    throw e;
  }
  auto rawMat = m->mat.get();
  if (auto cDenseMat = dynamic_cast<paddle::CpuMatrix*>(rawMat)) {
    *(cDenseMat->getData() + x + y * cDenseMat->getWidth()) = val;
  } else {
    UnsupportError e;
    throw e;
  }
}

bool Matrix::isSparse() const {
  auto raw_mat = m->mat.get();
  return dynamic_cast<paddle::CpuSparseMatrix*>(raw_mat) != nullptr ||
         dynamic_cast<paddle::GpuSparseMatrix*>(raw_mat) != nullptr;
}

SparseValueType Matrix::getSparseValueType() const throw(UnsupportError) {
  auto cpuSparseMat =
      std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(m->mat);
  if (cpuSparseMat != nullptr) {
    return (SparseValueType)cpuSparseMat->getValueType();
  } else {
    auto gpuSparseMat =
        std::dynamic_pointer_cast<paddle::GpuSparseMatrix>(m->mat);
    if (gpuSparseMat != nullptr) {
      return (SparseValueType)gpuSparseMat->getValueType();
    } else {
      UnsupportError e;
      throw e;
    }
  }
}

SparseFormatType Matrix::getSparseFormat() const throw(UnsupportError) {
  auto cpuSparseMat =
      std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(m->mat);
  if (cpuSparseMat != nullptr) {
    return (SparseFormatType)cpuSparseMat->getFormat();
  } else {
    auto gpuSparseMat =
        std::dynamic_pointer_cast<paddle::GpuSparseMatrix>(m->mat);
    if (gpuSparseMat != nullptr) {
      return SPARSE_CSR;
    } else {
      UnsupportError e;
      throw e;
    }
  }
}

IntArray Matrix::getSparseRowCols(size_t i) const
    throw(UnsupportError, RangeError) {
  auto cpuSparseMat =
      std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(m->mat);
  if (cpuSparseMat != nullptr &&
      cpuSparseMat->getFormat() == paddle::SPARSE_CSR) {
    if (i < cpuSparseMat->getHeight()) {
      // cpuSparseMat->print(std::cout);
      size_t len = cpuSparseMat->getColNum(i);
      return IntArray(cpuSparseMat->getRowCols(i), len);
    } else {
      RangeError e;
      throw e;
    }
  } else {
    UnsupportError e;
    throw e;
  }
}

IntWithFloatArray Matrix::getSparseRowColsVal(size_t i) const
    throw(UnsupportError, RangeError) {
  auto cpuSparseMat =
      std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(m->mat);
  if (cpuSparseMat != nullptr &&
      cpuSparseMat->getValueType() == paddle::FLOAT_VALUE) {
    if (i < cpuSparseMat->getHeight()) {
      return IntWithFloatArray(cpuSparseMat->getRowValues(i),
                               cpuSparseMat->getRowCols(i),
                               cpuSparseMat->getColNum(i));
    } else {
      RangeError e;
      throw e;
    }
  } else {
    UnsupportError e;
    throw e;
  }
}

FloatArray Matrix::getData() const {
  auto rawMat = m->mat.get();
  if (dynamic_cast<paddle::GpuMemoryHandle*>(rawMat->getMemoryHandle().get())) {
    // is gpu. then copy data
    float* data = rawMat->getData();
    size_t len = rawMat->getElementCnt();
    float* cpuData = new float[len];
    hl_memcpy_device2host(cpuData, data, len * sizeof(float));
    FloatArray ret_val(cpuData, len);
    ret_val.needFree = true;
    return ret_val;
  } else {
    FloatArray ret_val(rawMat->getData(), rawMat->getElementCnt());
    return ret_val;
  }
}

void Matrix::sparseCopyFrom(
    const std::vector<int>& rows,
    const std::vector<int>& cols,
    const std::vector<float>& vals) throw(UnsupportError) {
  auto cpuSparseMat =
      std::dynamic_pointer_cast<paddle::CpuSparseMatrix>(m->mat);
  if (cpuSparseMat != nullptr) {
    // LOG(INFO) <<"RowSize = "<<rows.size()
    //  <<" ColSize = "<<cols.size()
    //  <<" ValSize = "<<vals.size();
    cpuSparseMat->copyFrom(const_cast<std::vector<int>&>(rows),
                           const_cast<std::vector<int>&>(cols),
                           const_cast<std::vector<float>&>(vals));
  } else {
    UnsupportError e;
    throw e;
  }
}

void* Matrix::getSharedPtr() const { return &m->mat; }

void Matrix::toNumpyMatInplace(float** view_data,
                               int* dim1,
                               int* dim2) throw(UnsupportError) {
  auto cpuMat = std::dynamic_pointer_cast<paddle::CpuMatrix>(m->mat);
  if (cpuMat) {
    *dim1 = cpuMat->getHeight();
    *dim2 = cpuMat->getWidth();
    *view_data = cpuMat->getData();
  } else {
    throw UnsupportError();
  }
}
void Matrix::copyToNumpyMat(float** view_m_data,
                            int* dim1,
                            int* dim2) throw(UnsupportError) {
  static_assert(sizeof(paddle::real) == sizeof(float),
                "Currently PaddleAPI only support for single "
                "precision version of paddle.");
  if (this->isSparse()) {
    throw UnsupportError();
  } else {
    *dim1 = m->mat->getHeight();
    *dim2 = m->mat->getWidth();
    *view_m_data = new float[(*dim1) * (*dim2)];
    if (auto cpuMat = dynamic_cast<paddle::CpuMatrix*>(m->mat.get())) {
      auto src = cpuMat->getData();
      auto dest = *view_m_data;
      std::memcpy(dest, src, sizeof(paddle::real) * (*dim1) * (*dim2));
    } else if (auto gpuMat = dynamic_cast<paddle::GpuMatrix*>(m->mat.get())) {
      auto src = gpuMat->getData();
      auto dest = *view_m_data;
      hl_memcpy_device2host(
          dest, src, sizeof(paddle::real) * (*dim1) * (*dim2));
    } else {
      LOG(WARNING) << "Unexpected Situation";
      throw UnsupportError();
    }
  }
}

void Matrix::copyFromNumpyMat(float* data,
                              int dim1,
                              int dim2) throw(UnsupportError, RangeError) {
  if (isSparse()) {
    throw UnsupportError();
  } else {
    if (this->getHeight() == (size_t)dim1 && this->getWidth() == (size_t)dim2) {
      if (m->mat->getData() != data) {
        m->mat->copyFrom(data, dim1 * dim2);
      }
    } else {
      throw RangeError();
    }
  }
}

bool Matrix::isGpu() const {
  auto rawPtr = m->mat.get();
  return dynamic_cast<paddle::GpuMatrix*>(rawPtr) != nullptr ||
         dynamic_cast<paddle::GpuSparseMatrix*>(rawPtr) != nullptr;
}
