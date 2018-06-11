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

#pragma once

#include <map>
#include <vector>
#include "BufferArg.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Any.h"
#include "paddle/utils/ClassRegistrar.h"
#include "paddle/utils/Error.h"

namespace paddle {

/**
 * Function Configuration.
 * The argument type of Function::init.
 */
class FuncConfig {
 public:
  template <typename T>
  T get(const std::string& key, Error* err = nullptr) const {
    try {
      return any_cast<T>(valueMap_.at(key));
    } catch (std::exception& e) {  // could be cast or out of range exception.
      if (err) {
        *err = Error(e.what());
      } else {
        LOG(FATAL) << "Cannot get key " << key << " with error " << e.what();
      }
      return T();
    }
  }

  template <typename T>
  FuncConfig& set(const std::string& key, T v, Error* err = nullptr) {
    auto it = valueMap_.find(key);
    if (it != valueMap_.end()) {  // already contains key.
      if (err) {
        *err = Error("Key %s is already set in FuncConfig", key.c_str());
      } else {
        LOG(FATAL) << "Key " << key << " is already set in FuncConfig.";
      }
      return *this;
    }
    valueMap_[key] = any(v);
    return *this;
  }

 protected:
  mutable std::unordered_map<std::string, any> valueMap_;
};

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

/**
 * \brief Base class for Function.
 * The basic Function implementation requires override init and calc interfaces.
 *
 * The caller needs to ensure the validity of the arguments
 * during Function execution.
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

  // This member function is used to check whether the BufferType and shape of
  // the inputs and outputs arguments of the Function are correct.
  // General calc function which will call this check to do arguments check.
  // And before the calc called, the caller can also check their own arguments.
  virtual void check(const BufferArgs& inputs, const BufferArgs& outputs) {}

  // Calculate the number of floating-point operations of this Function.
  // The inputs and outputs arguments do not need to contain the actual data,
  // only the shape.
  // And some Functions have the same input and output shapes,
  // so you may not need to enter the complete number of arguments.
  // But entering the full arguments is always correct for this interface.
  virtual size_t ops(const BufferArgs& inputs, const BufferArgs& outputs) {
    return 0;
  }

  int getNumInputs() const { return numInputs_; }

  int getNumOutputs() const { return numOutputs_; }

  static ClassRegistrar<FunctionBase> funcRegistrar_;

 protected:
  // numInputs_ and numOutputs_ represents the maximum
  // input and output supported by Function.
  // Some functions are optimized for input and output,
  // so when comparing the number of arguments, for these functions
  // inputs.size() <= numInputs_ or outputs.size() <= numOutputs_
  size_t numInputs_;
  size_t numOutputs_;
};

#define FUNC_NAME(typeName, deviceName) #typeName "-" #deviceName

#define REGISTER_TYPED_FUNC(typeName, deviceName, className)   \
  static InitFunction __reg_type_##typeName##deviceName([]() { \
    FunctionBase::funcRegistrar_                               \
        .registerClass<className<DEVICE_TYPE_##deviceName>>(   \
            FUNC_NAME(typeName, deviceName));                  \
  })

}  // namespace paddle
