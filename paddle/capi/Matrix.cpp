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

#include "capi_private.h"
#include "hl_cuda.h"
#include "matrix.h"

#define cast(v) paddle::capi::cast<paddle::capi::CMatrix>(v)
extern "C" {
paddle_matrix paddle_matrix_create(uint64_t height,
                                   uint64_t width,
                                   bool useGpu) {
  auto ptr = new paddle::capi::CMatrix();
  ptr->mat = paddle::Matrix::create(height, width, false, useGpu);
  return ptr;
}

paddle_matrix paddle_matrix_create_none() {
  return new paddle::capi::CMatrix();
}

paddle_error paddle_matrix_destroy(paddle_matrix mat) {
  if (mat == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  delete ptr;
  return kPD_NO_ERROR;
}

paddle_error paddle_matrix_set_row(paddle_matrix mat,
                                   uint64_t rowID,
                                   paddle_real* rowArray) {
  if (mat == nullptr || rowArray == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  if (ptr->mat == nullptr) return kPD_NULLPTR;
  if (rowID >= ptr->mat->getHeight()) return kPD_OUT_OF_RANGE;
  paddle::real* buf = ptr->mat->getRowBuf(rowID);
  size_t width = ptr->mat->getWidth();
#ifdef PADDLE_WITH_CUDA
  hl_memcpy(buf, rowArray, sizeof(paddle::real) * width);
#else
  std::copy(rowArray, rowArray + width, buf);
#endif
  return kPD_NO_ERROR;
}

PD_API paddle_error paddle_matrix_set_value(paddle_matrix mat,
                                            paddle_real* value) {
  if (mat == nullptr || value == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  if (ptr->mat == nullptr) return kPD_NULLPTR;
  paddle::real* buf = ptr->mat->getRowBuf(0);
  size_t width = ptr->mat->getWidth();
  size_t height = ptr->mat->getHeight();
  if (ptr->mat->useGpu()) {
#ifdef PADDLE_WITH_CUDA
    hl_memcpy(buf, value, sizeof(paddle::real) * width * height);
#else
    return kPD_NOT_SUPPORTED;
#endif
  } else {
    std::copy(value, value + width * height, buf);
  }
  return kPD_NO_ERROR;
}

PD_API paddle_error paddle_matrix_get_value(paddle_matrix mat,
                                            paddle_real* result) {
  if (mat == nullptr || result == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  if (ptr->mat == nullptr) return kPD_NULLPTR;
  paddle::real* buf = ptr->mat->getRowBuf(0);
  size_t width = ptr->mat->getWidth();
  size_t height = ptr->mat->getHeight();
  if (ptr->mat->useGpu()) {
#ifdef PADDLE_WITH_CUDA
    hl_memcpy(result, buf, width * height * sizeof(paddle::real));
#else
    return kPD_NOT_SUPPORTED;
#endif
  } else {
    std::copy(buf, buf + width * height, result);
  }
  return kPD_NO_ERROR;
}

paddle_error paddle_matrix_get_row(paddle_matrix mat,
                                   uint64_t rowID,
                                   paddle_real** rawRowBuffer) {
  if (mat == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  if (ptr->mat == nullptr) return kPD_NULLPTR;
  if (rowID >= ptr->mat->getHeight()) return kPD_OUT_OF_RANGE;
  *rawRowBuffer = ptr->mat->getRowBuf(rowID);
  return kPD_NO_ERROR;
}

paddle_error paddle_matrix_get_shape(paddle_matrix mat,
                                     uint64_t* height,
                                     uint64_t* width) {
  if (mat == nullptr || cast(mat)->mat == nullptr) return kPD_NULLPTR;
  if (height != nullptr) {
    *height = cast(mat)->mat->getHeight();
  }
  if (width != nullptr) {
    *width = cast(mat)->mat->getWidth();
  }
  return kPD_NO_ERROR;
}
}

paddle_matrix paddle_matrix_create_sparse(
    uint64_t height, uint64_t width, uint64_t nnz, bool isBinary, bool useGpu) {
#ifndef PADDLE_MOBILE_INFERENCE
  auto ptr = new paddle::capi::CMatrix();
  ptr->mat = paddle::Matrix::createSparseMatrix(
      height,
      width,
      nnz,
      isBinary ? paddle::NO_VALUE : paddle::FLOAT_VALUE,
      paddle::SPARSE_CSR,
      false,
      useGpu);
  return ptr;
#else
  return nullptr;
#endif
}

paddle_error paddle_matrix_sparse_copy_from(paddle_matrix mat,
                                            int* rowArray,
                                            uint64_t rowSize,
                                            int* colArray,
                                            uint64_t colSize,
                                            float* valueArray,
                                            uint64_t valueSize) {
#ifndef PADDLE_MOBILE_INFERENCE
  if (mat == nullptr) return kPD_NULLPTR;
  auto ptr = cast(mat);
  if (rowArray == nullptr || colArray == nullptr ||
      (valueSize != 0 && valueArray == nullptr) || ptr->mat == nullptr) {
    return kPD_NULLPTR;
  }
  if (auto sparseMat = dynamic_cast<paddle::CpuSparseMatrix*>(ptr->mat.get())) {
    std::vector<int> row(rowSize);
    row.assign(rowArray, rowArray + rowSize);
    std::vector<int> col(colSize);
    col.assign(colArray, colArray + colSize);
    std::vector<paddle_real> val(valueSize);
    if (valueSize) {
      val.assign(valueArray, valueArray + valueSize);
    }
    sparseMat->copyFrom(row, col, val);
    return kPD_NO_ERROR;
  } else {
    return kPD_NOT_SUPPORTED;
  }
#else
  return kPD_NOT_SUPPORTED;
#endif
}
