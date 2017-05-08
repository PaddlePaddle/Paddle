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
#include "BufferArg.h"
#include "BufferArgs.h"
#include "paddle/topology/meta/FunctionMeta.h"
#include "paddle/utils/Any.h"

namespace paddle {
namespace function {

typedef std::function<paddle::Error(
    const BufferArgs& inputs,
    const BufferArgs& outputs,
    const std::unordered_map<std::string, any>& attrs)>
    KernelType;

class FunctionMetaRegister {
public:
  FunctionMetaRegister(topology::meta::FunctionMetaPtr& meta) : meta_(meta) {}

  paddle::Error addKernel(const std::string& name, KernelType func) {
    return meta_->addMeta(name, func);
  }

  paddle::Error addCPUKernel(KernelType func) {
    return this->addKernel("CPUKernel", func);
  }
  paddle::Error addGPUKernel(KernelType func) {
    return this->addKernel("GPUKernel", func);
  }

private:
  topology::meta::FunctionMetaPtr& meta_;
};

}  // namespace function
}  // namespace paddle
