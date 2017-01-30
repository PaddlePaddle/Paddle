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
#include <gtest/gtest.h>
#include <paddle/utils/Util.h>
#include "paddle/math/SparseMatrix.h"

namespace paddle {

void checkMatrixEqual(const MatrixPtr& a, const MatrixPtr& b) {
  ASSERT_EQ(a->getWidth(), b->getWidth());
  ASSERT_EQ(a->getHeight(), b->getHeight());
  ASSERT_EQ(a->isTransposed(), b->isTransposed());
  for (size_t r = 0; r < a->getHeight(); ++r) {
    for (size_t c = 0; c < a->getWidth(); ++c) {
      ASSERT_FLOAT_EQ(a->getElement(r, c), b->getElement(r, c));
    }
  }
}

void checkSMatrixEqual(const CpuSparseMatrix& a, const CpuSparseMatrix& b) {
  ASSERT_EQ(a.getWidth(), b.getWidth());
  ASSERT_EQ(a.getHeight(), b.getHeight());
  ASSERT_EQ(a.isTransposed(), b.isTransposed());
  ASSERT_EQ(a.getFormat(), b.getFormat());
  ASSERT_EQ(a.getElementCnt(), b.getElementCnt());
  for (size_t r = 0; r < a.getElementCnt(); ++r) {
    ASSERT_FLOAT_EQ(a.getValue()[r], b.getValue()[r]);
  }
}

void checkSMatrixEqual(const CpuSparseMatrixPtr& a,
                       const CpuSparseMatrixPtr& b) {
  ASSERT_EQ(a->getWidth(), b->getWidth());
  ASSERT_EQ(a->getHeight(), b->getHeight());
  ASSERT_EQ(a->isTransposed(), b->isTransposed());
  ASSERT_EQ(a->getFormat(), b->getFormat());
  ASSERT_EQ(a->getElementCnt(), b->getElementCnt());
  for (size_t r = 0; r < a->getElementCnt(); ++r) {
    ASSERT_FLOAT_EQ(a->getValue()[r], b->getValue()[r]);
  }
}

void checkSMatrixEqual2(const CpuSparseMatrixPtr& a,
                        const CpuSparseMatrixPtr& b) {
  ASSERT_EQ(a->getWidth(), b->getWidth());
  ASSERT_EQ(a->getHeight(), b->getHeight());
  ASSERT_EQ(a->isTransposed(), b->isTransposed());
  ASSERT_EQ(a->getFormat(), b->getFormat());
  ASSERT_EQ(a->getValueType(), b->getValueType());
  ASSERT_EQ(a->getElementCnt(), b->getElementCnt());
  if (a->getFormat() == SPARSE_CSR) {
    for (size_t r = 0; r < a->getElementCnt(); ++r) {
      ASSERT_EQ(a->getCols()[r], b->getCols()[r]);
      if (a->getValueType() == FLOAT_VALUE) {
        ASSERT_FLOAT_EQ(a->getValue()[r], b->getValue()[r]);
      }
    }
    for (size_t r = 0; r <= a->getHeight(); r++) {
      ASSERT_EQ(a->getRows()[r], b->getRows()[r]);
    }
  } else {
    for (size_t r = 0; r < a->getElementCnt(); ++r) {
      ASSERT_EQ(a->getRows()[r], b->getRows()[r]);
      if (a->getValueType() == FLOAT_VALUE) {
        ASSERT_FLOAT_EQ(a->getValue()[r], b->getValue()[r]);
      }
    }
    for (size_t r = 0; r <= a->getWidth(); r++) {
      ASSERT_EQ(a->getCols()[r], b->getCols()[r]);
    }
  }
}

void checkSMatrixEqual2Dense(const CpuSparseMatrix& a, const CpuMatrix& b) {
  ASSERT_EQ(a.getWidth(), b.getWidth());
  ASSERT_EQ(a.getHeight(), b.getHeight());
  ASSERT_EQ(a.isTransposed(), b.isTransposed());

  if (a.getFormat() == SPARSE_CSC) {
    int* rows = a.getRows();
    for (size_t i = 0; i < a.getWidth(); i++) {
      for (size_t j = a.getColStartIdx(i); j < a.getColStartIdx(i + 1); j++) {
        if (a.getValueType() == FLOAT_VALUE) {
          ASSERT_FLOAT_EQ(a.getValue()[j], b.getElement(rows[j], i));
        } else {
          ASSERT_FLOAT_EQ(1.0, b.getElement(rows[j], i));
        }
      }
    }
  } else {
    int* cols = a.getCols();
    for (size_t i = 0; i < a.getHeight(); i++) {
      for (size_t j = a.getRowStartIdx(i); j < a.getRowStartIdx(i + 1); j++) {
        if (a.getValueType() == FLOAT_VALUE) {
          ASSERT_FLOAT_EQ(a.getValue()[j], b.getElement(i, cols[j]));
        } else {
          ASSERT_FLOAT_EQ(1.0, b.getElement(i, cols[j]));
        }
      }
    }
  }
}

void checkSMatrixEqual2Dense(const CpuSparseMatrixPtr& a,
                             const CpuMatrixPtr& b) {
  ASSERT_EQ(a->getWidth(), b->getWidth());
  ASSERT_EQ(a->getHeight(), b->getHeight());
  ASSERT_EQ(a->isTransposed(), b->isTransposed());

  if (a->getFormat() == SPARSE_CSC) {
    int* rows = a->getRows();
    for (size_t i = 0; i < a->getWidth(); i++) {
      for (size_t j = a->getColStartIdx(i); j < a->getColStartIdx(i + 1); j++) {
        if (a->getValueType() == FLOAT_VALUE) {
          ASSERT_FLOAT_EQ(a->getValue()[j], b->getElement(rows[j], i));
        } else {
          ASSERT_FLOAT_EQ(1.0, b->getElement(rows[j], i));
        }
      }
    }
  } else {
    int* cols = a->getCols();
    for (size_t i = 0; i < a->getHeight(); i++) {
      for (size_t j = a->getRowStartIdx(i); j < a->getRowStartIdx(i + 1); j++) {
        if (a->getValueType() == FLOAT_VALUE) {
          ASSERT_FLOAT_EQ(a->getValue()[j], b->getElement(i, cols[j]));
        } else {
          ASSERT_FLOAT_EQ(1.0, b->getElement(i, cols[j]));
        }
      }
    }
  }
}

void checkSMatrixErr(const CpuSparseMatrixPtr& a, const CpuSparseMatrixPtr& b) {
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif
  ASSERT_EQ(a->getWidth(), b->getWidth());
  ASSERT_EQ(a->getHeight(), b->getHeight());
  ASSERT_EQ(a->isTransposed(), b->isTransposed());
  ASSERT_EQ(a->getFormat(), b->getFormat());
  ASSERT_EQ(a->getValueType(), b->getValueType());
  ASSERT_EQ(a->getElementCnt(), b->getElementCnt());
  int count = 0;
  if (a->getFormat() == SPARSE_CSR) {
    for (size_t r = 0; r < a->getElementCnt(); ++r) {
      ASSERT_EQ(a->getCols()[r], b->getCols()[r]);
      if (a->getValueType() == FLOAT_VALUE) {
        real aVal = a->getValue()[r];
        real bVal = b->getValue()[r];
        if (std::abs(aVal - bVal) > err) {
          if ((std::abs(aVal - bVal) / std::abs(aVal)) > (err / 10.0f)) {
            LOG(INFO) << "a=" << aVal << "\t"
                      << "b=" << bVal;
            count++;
          }
        }
      }
    }
    for (size_t r = 0; r <= a->getHeight(); r++) {
      ASSERT_EQ(a->getRows()[r], b->getRows()[r]);
    }
  } else {
    for (size_t r = 0; r < a->getElementCnt(); ++r) {
      ASSERT_EQ(a->getRows()[r], b->getRows()[r]);
      if (a->getValueType() == FLOAT_VALUE) {
        real aVal = a->getValue()[r];
        real bVal = b->getValue()[r];
        if (std::abs(aVal - bVal) > err) {
          if ((std::abs(aVal - bVal) / std::abs(aVal)) > (err / 10.0f)) {
            count++;
          }
        }
      }
    }
    for (size_t r = 0; r <= a->getWidth(); r++) {
      ASSERT_EQ(a->getCols()[r], b->getCols()[r]);
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

void checkMatrixErr(const Matrix& matrix1, const Matrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();
  const real* data1 = matrix1.getData();
  const real* data2 = matrix2.getData();
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      real a = data1[i * width + j];
      real b = data2[i * width + j];
      if (std::abs(a - b) > err) {
        if ((std::abs(a - b) / std::abs(a)) > (err / 10.0f)) {
          count++;
        }
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

void checkDataEqual(const real* a, const real* b, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    ASSERT_FLOAT_EQ(a[i], b[i]);
  }
}

}  //  namespace paddle
