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

#include <glog/logging.h>

#include "BufferArg.h"

namespace paddle {

const SequenceArg& BufferArg::sequence() const {
  // CHECK_EQ(bufferType_, TENSOR_SEQUENCE_DATA);
  return dynamic_cast<const SequenceArg&>(*this);
}

const SparseMatrixArg& BufferArg::sparse() const {
  // CHECK_EQ(bufferType_, TENSOR_SPARSE);
  return dynamic_cast<const SparseMatrixArg&>(*this);
}

void BufferArgs::addArg(const Matrix& arg, const TensorShape& shape) {
  args_.push_back(std::make_shared<BufferArg>(arg, shape));
}

void BufferArgs::addArg(const CpuSparseMatrix& arg) {
  args_.push_back(std::make_shared<SparseMatrixArg>(arg));
}

void BufferArgs::addArg(const GpuSparseMatrix& arg) {
  args_.push_back(std::make_shared<SparseMatrixArg>(arg));
}

}  // namespace paddle
