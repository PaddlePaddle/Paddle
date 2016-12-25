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
#include "paddle/math/Matrix.h"
#include "paddle/utils/ClassRegistrar.h"

namespace paddle {

enum DeviceType {
  DEVICE_TYPE_UNSPECIFIED = 0,
  DEVICE_TYPE_CPU = 1,
  DEVICE_TYPE_GPU = 2,
};

template <DeviceType Device>
struct MatrixT;

template <>
struct MatrixT<DEVICE_TYPE_CPU> {
  using type = CpuMatrix;
};

template <>
struct MatrixT<DEVICE_TYPE_GPU> {
  using type = GpuMatrix;
};

typedef std::vector<size_t> Dims;

class Tensor {
public:
  Tensor(real* data, const Dims& dim) : buf_(data), dims_(dim) {}

  real* getData() const { return buf_; }

  real* buf_;
  Dims dims_;
};

typedef std::vector<Tensor> Arguments;

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

  virtual void calc(const Arguments& inputs,
                    const Arguments& outputs,
                    const Arguments& inouts) {}

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
