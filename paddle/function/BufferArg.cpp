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

#include <glog/logging.h>

#include "BufferArg.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {

const SequenceArg& BufferArg::sequence() const {
  CHECK_EQ(bufferType_, TENSOR_SEQUENCE_DATA);
  return dynamic_cast<const SequenceArg&>(*this);
}

const SparseMatrixArg& BufferArg::sparse() const {
  CHECK_EQ(bufferType_, TENSOR_SPARSE);
  return dynamic_cast<const SparseMatrixArg&>(*this);
}

SparseMatrixArg::SparseMatrixArg(const CpuSparseMatrix& sparse, ArgType argType)
    : BufferArg(sparse, argType),
      row_(reinterpret_cast<void*>(sparse.getRows()), VALUE_TYPE_INT32),
      col_(reinterpret_cast<void*>(sparse.getCols()), VALUE_TYPE_INT32),
      nnz_(sparse.getElementCnt()),
      format_(static_cast<SparseDataFormat>(sparse.getFormat())),
      type_(static_cast<SparseDataType>(sparse.getValueType())) {
  bufferType_ = TENSOR_SPARSE;
}

SparseMatrixArg::SparseMatrixArg(const GpuSparseMatrix& sparse, ArgType argType)
    : BufferArg(sparse, argType),
      row_(reinterpret_cast<void*>(sparse.getRows()), VALUE_TYPE_INT32),
      col_(reinterpret_cast<void*>(sparse.getCols()), VALUE_TYPE_INT32),
      nnz_(sparse.getElementCnt()),
      format_(static_cast<SparseDataFormat>(sparse.getFormat())),
      type_(static_cast<SparseDataType>(sparse.getValueType())) {
  bufferType_ = TENSOR_SPARSE;
}

}  // namespace paddle
