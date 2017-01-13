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

#include <map>
#include <vector>
#include "BufferArg.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {

/**
 * Function Configuration.
 * The argument type of Function::init.
 * Follow-up will consider moving this data structure to Proto inside.
 */
class FuncConfig {
public:
  union value {
    size_t s;
    real r;
    int i;
    bool b;
  };

  template <typename T>
  T get(const std::string& key) const;

  template <typename T>
  FuncConfig& set(const std::string& key, T v);

protected:
  std::map<std::string, value> valueMap_;
};

/**
 * Argument type for Function::calc().
 * A BufferArgs contains a set of BufferArg,
 * because Function can have multiple inputs and outputs.
 */
class BufferArgs {
public:
  BufferArgs() {}
  size_t size() const { return args_.size(); }

  // add argument into BufferArgs
  // Tensor can be Matrix, Vector, IVector.
  // For inputs, do not need argType.
  // For outputs, the argType needs to be specified as ASSIGN_TO or ADD_TO.
  template <typename Tensor>
  void addArg(const Tensor& arg, ArgType argType = UNSPECIFIED) {
    args_.push_back(std::make_shared<BufferArg>(arg, argType));
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

  // get argument
  const BufferArg& operator[](size_t num) const {
    CHECK_LT(num, args_.size());
    return *args_[num];
  }

private:
  std::vector<BufferArgPtr> args_;
};

/**
 * \brief Base class for Function.
 * The basic Function implementation requires override init and calc interfaces.
 *
 * Function inputs are readonly, Function outputs have two modes: ASSIGN_TO
 * and ADD_TO.
 * If output.getArgType() == ASSIGN_TO, this is assign mode, and the calculation
 * result of Function assigned to the output BufferArg.
 * If output.getArgType() == ADD_TO, this is add mode, and the calculation
 * result of Function need added to the output BufferArg.
 *
 * For example:
 * ASSIGN_TO: output = Function(inputs)
 * ADD_TO: output += Function(inputs)
 * If Function has more than one output, each output can have different modes.
 */
class FunctionBase {
public:
  virtual ~FunctionBase() {}

  virtual void init(const FuncConfig& config) {}

  virtual void calc(const BufferArgs& inputs, const BufferArgs& outputs) {}

  static ClassRegistrar<FunctionBase> funcRegistrar_;
};

#define FUNC_NAME(typeName, deviceName) #typeName "-" #deviceName

#define REGISTER_TYPED_FUNC(typeName, deviceName, className)   \
  static InitFunction __reg_type_##typeName##deviceName([]() { \
    FunctionBase::funcRegistrar_                               \
        .registerClass<className<DEVICE_TYPE_##deviceName>>(   \
            FUNC_NAME(typeName, deviceName));                  \
  })

}  // namespace paddle
