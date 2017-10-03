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
  device_contexts_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    if (platform::is_cpu_place(places[i])) {
      device_contexts_[i].reset(new platform::CPUDeviceContext(
          boost::get<platform::CPUPlace>(places[i])));
    } else {
#ifndef PADDLE_ONLY_CPU
      device_contexts_[i].reset(new platform::CUDADeviceContext(
          boost::get<platform::CPUPlace>(places[i])));
#else
      PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
    }
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope,
                   std::vector<Tensor>* outputs) {
  // operators running
  // TODO(tonyyang-svail):
  //    - only runs the first block
  //    - only runs on the first device
  Scope& local_scope = scope->NewScope();

  auto& block = pdesc.blocks(0);
  auto& device_context = device_contexts_[0];

  for (auto& var : block.vars()) {
    local_scope.NewVar(var.name());
  }

  // std::vector<op_ptr> ops;
  for (auto& op_desc : block.ops()) {
    auto op = framework::OpRegistry::CreateOp(op_desc);
    // InferShape is now doing inside Run method.
    op->Run(local_scope, *device_context);
  }

  // TODO(tonyyang-svail): need to test gpu device
  for (auto& device_context : device_contexts_) {
    device_context->Wait();
  }
}

}  // namespace framework
}  // namespace paddle
