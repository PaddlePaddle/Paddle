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

#include "paddle/framework/executor.h"
#include <memory>
#include <vector>
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

Executor::Executor(const std::vector<platform::Place>& places) {
  devices_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    devices_[i] = platform::GetDevice(places[i]);
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope,
                   std::vector<Tensor>* outputs) {
  // operators running
  Scope& local_scope = scope->NewScope();
  local_scope.NewVar();
  for (auto device : devices_) {
    device->cpu_device_context->Wait();
#ifndef PADDLE_ONLY_CPU
    if (device->cuda_device_context) {
      device->cuda_device_context->Wait();
    }
#endif
  }
}

}  // namespace framework
}  // namespace paddle
