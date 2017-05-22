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

#include "FunctionList.h"
#include "Function.h"
#include "Register.h"
#include "paddle/topology/Function.h"

namespace paddle {
namespace function {

class FunctionAttributeSetterPrivate {
public:
  FunctionList *self;
  topology::Function func;
};

void FunctionList::add(const std::string &name,
                       const Config &config,
                       bool useGPU) {
  std::shared_ptr<FunctionBase> func;
  if (useGPU) {
    func.reset(FunctionBase::funcRegistrar_.createByType(name + "-GPU"));
  } else {
    func.reset(FunctionBase::funcRegistrar_.createByType(name + "-CPU"));
  }
  func->init(config);
  this->push_back([func](const BufferArgs &inputs, const BufferArgs &outputs) {
    func->calc(inputs, outputs);
    //! TODO(yuyang18): Make FunctionBase::calc return paddle::Error.
    return paddle::Error();
  });
}

FunctionAttributeSetter FunctionList::add(const std::string &name,
                                          bool useGPU) {
  FunctionAttributeSetter setter;
  setter.m->func.type = name;
  setter.m->func.setUseGPU(useGPU);
  setter.m->self = this;
  return setter;
}

FunctionAttributeSetter::FunctionAttributeSetter()
    : m(new FunctionAttributeSetterPrivate()) {}

paddle::topology::meta::AttributeMap &FunctionAttributeSetter::attrs() {
  return m->func.attributes;
}

FunctionAttributeSetter::~FunctionAttributeSetter() {
  if (m != nullptr) {
    m->self->push_back(createFunction(m->func));
    delete m;
  }
}

FunctionAttributeSetter::FunctionAttributeSetter(FunctionAttributeSetter &o) {
  m = o.m;
  o.m = nullptr;
}

}  // namespace function

}  // namespace paddle
