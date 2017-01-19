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
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {
class FuncConfig;
class BufferArgs;

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

  virtual void init(const FuncConfig& config) { (void)(config); }

  virtual void calc(const BufferArgs& inputs, const BufferArgs& outputs) {
    (void)(inputs);
    (void)(outputs);
  }

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
