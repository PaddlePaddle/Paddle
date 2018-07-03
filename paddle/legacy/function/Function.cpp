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

#include "Function.h"

namespace paddle {

void BufferArgs::addArg(const Matrix& arg,
                        const TensorShape& shape,
                        ArgType argType) {
  _args_.push_back(new BufferArg(arg, shape, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const CpuSparseMatrix& arg, ArgType argType) {
  _args_.push_back(new SparseMatrixArg(arg, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const GpuSparseMatrix& arg, ArgType argType) {
  _args_.push_back(new SparseMatrixArg(arg, argType));
  addArg(*_args_.back());
}

void BufferArgs::addArg(const Matrix& matrix,
                        const IVector& vector,
                        ArgType argType) {
  _args_.push_back(new SequenceArg(matrix, vector, argType));
  addArg(*_args_.back());
}

ClassRegistrar<FunctionBase> FunctionBase::funcRegistrar_;

}  // namespace paddle
