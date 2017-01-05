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

class FunctionBase {
public:
  virtual ~FunctionBase() {}

  virtual void init(const FuncConfig& config) {}

  virtual void calc(const BufferArgs& inputs,
                    const BufferArgs& outputs,
                    const BufferArgs& inouts) {}

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
