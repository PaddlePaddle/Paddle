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
#include <functional>
#include "../FunctionList.h"
#include "paddle/topology/Function.h"
#include "paddle/topology/meta/FunctionMeta.h"
#include "paddle/utils/Util.h"

namespace paddle {
namespace function {
namespace details {
class FunctionRegister {
public:
  FunctionRegister(topology::meta::FunctionMetaPtr& meta) : meta_(meta) {}

  paddle::Error addCPUFunction(Function kernel) {
    return this->addFunction("CPUKernel", kernel);
  }

  paddle::Error addGPUFunction(Function kernel) {
    return this->addFunction("GPUKernel", kernel);
  }

  paddle::Error addCPUFunction(FunctionWithAttrs kernel) {
    return this->addFunction("CPUKernel", kernel);
  }

  paddle::Error addGPUFunction(FunctionWithAttrs kernel) {
    return this->addFunction("GPUKernel", kernel);
  }

private:
  template <typename T>
  paddle::Error addFunction(const std::string& name, T kernel) {
    return meta_->addMeta(name, kernel);
  }
  topology::meta::FunctionMetaPtr& meta_;
};

}  // namespace details

}  // namespace function
}  // namespace paddle
