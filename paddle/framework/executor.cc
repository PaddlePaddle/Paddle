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
#include <iostream>
#include <memory>
#include <vector>
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

Executor::Executor(const std::vector<platform::Place>& places) {
  device_contexts_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    if (platform::is_cpu_place(places[i])) {
      device_contexts_[i] = new platform::CPUDeviceContext(
          boost::get<platform::CPUPlace>(places[i]));
    } else if (platform::is_gpu_place(places[i])) {
#ifndef PADDLE_ONLY_CPU
      device_contexts_[i] = new platform::CUDADeviceContext(
          boost::get<platform::GPUPlace>(places[i]));
#else
      PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
    }
  }
}

Executor::~Executor() {
  for (auto& device_context : device_contexts_) {
    if (device_context) {
      delete device_context;
    }
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope,
                   std::vector<Tensor>* outputs) {
  // TODO(tonyyang-svail):
  //    - only runs the first block
  //    - only runs on the first device
  //    - test on gpu
  auto& block = pdesc.blocks(0);
  auto& device = device_contexts_[0];

  // TODO(tonyyang-svail):
  //    - runs on a new local scope
  // Scope& local_scope = scope->NewScope();

  for (auto& var : block.vars()) {
    scope->NewVar(var.name());
  }

  for (auto& op_desc : block.ops()) {
    auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
    std::cout << op->DebugString() << std::endl;
    op->Run(*scope, *device);
  }

  // TODO(tonyyang-svail): need to test gpu device
  for (auto& device_context : device_contexts_) {
    device_context->Wait();
  }
  // // print tensor value
  // for (auto& var : block.vars()) {
  //   std::cout << var.name() << std::endl;
  //   auto v = scope->FindVar(var.name());
  //   const LoDTensor& t = v->Get<LoDTensor>();
  //   for (int i = 0; i < t.numel(); ++i)
  //     std::cout << t.data<float>()[i] << " ";
  //   std::cout << std::endl;
  // }
}

}  // namespace framework
}  // namespace paddle
