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
#include "BufferArg.h"

namespace paddle {
/**
 * Argument type for Function::calc().
 * A BufferArgs contains a set of BufferArg,
 * because Function can have multiple inputs and outputs.
 *
 * addArg() with Matix object used to adapt Layer Argument.
 * Will create a BufferArg object in addArg(),
 * and free in destructor of BufferArgs.
 *
 * addArg() with BufferArg object, just save BufferArg object address,
 * and the caller needs to guarantee the validity of the BufferArg object
 * in the BufferArgs life time.
 */
class BufferArgs {
public:
  BufferArgs() {}

  ~BufferArgs() {
    for (auto arg : _args_) {
      delete arg;
    }
  }

  size_t size() const { return args_.size(); }

  // add argument into BufferArgs
  // Tensor can be Matrix, Vector, IVector.
  // For inputs, do not need argType.
  // For outputs, the argType needs to be specified as ASSIGN_TO or ADD_TO.
  void addArg(const Matrix& arg, ArgType argType = UNSPECIFIED) {
    _args_.push_back(new BufferArg(arg, argType));
    addArg(*_args_.back());
  }

  void addArg(const Vector& arg, ArgType argType = UNSPECIFIED) {
    _args_.push_back(new BufferArg(arg, argType));
    addArg(*_args_.back());
  }

  void addArg(const IVector& arg, ArgType argType = UNSPECIFIED) {
    _args_.push_back(new BufferArg(arg, argType));
    addArg(*_args_.back());
  }

  // Add arg into BufferArgs and reshape the arg.
  //
  // For example, arg represents an image buffer,
  // but Matrix can only represent a two-dimensional Tensor.
  // So need an extra argument to describe the shape of the image buffer.
  void addArg(const Matrix& arg,
              const TensorShape& shape,
              ArgType argType = UNSPECIFIED);

  void addArg(const CpuSparseMatrix& arg, ArgType argType = UNSPECIFIED);
  void addArg(const GpuSparseMatrix& arg, ArgType argType = UNSPECIFIED);

  void addArg(const Matrix& matrix,
              const IVector& vector,
              ArgType argType = UNSPECIFIED);

  // get argument
  const BufferArg& operator[](size_t num) const {
    CHECK_LT(num, args_.size());
    return *args_[num];
  }

  void addArg(BufferArg& arg) { args_.push_back(&arg); }

  void addArg(SequenceIdArg& arg) { args_.push_back(&arg); }

  void addArg(SequenceArg& arg) { args_.push_back(&arg); }

  void addArg(SparseMatrixArg& arg) { args_.push_back(&arg); }

private:
  std::vector<BufferArg*> args_;
  // The BufferArg object is constructed and freed by BufferArgs.
  std::vector<BufferArg*> _args_;
};

}  // namespace paddle
