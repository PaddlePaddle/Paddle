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
  devices_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    devices_[i] = platform::GetDevice(places[i]);
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope,
                   std::vector<Tensor>* outputs) {
  // TODO(tonyyang-svail):
  //    - only runs the first block
  //    - only runs on the first device
  //    - test on gpu
  auto& block = pdesc.blocks(0);
  auto& device = devices_[0];

  // TODO(tonyyang-svail):
  //    - runs on a new local scope
  // Scope& local_scope = scope->NewScope();

  for (auto& var : block.vars()) {
    scope->NewVar(var.name());
  }

  for (auto& op_desc : block.ops()) {
    auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
    op->Run(*scope, *device->cpu_device_context);
  }

  // print tensor value
  for (auto& var : block.vars()) {
    std::cout << var.name() << std::endl;
    auto v = scope->FindVar(var.name());
    const LoDTensor& t = v->Get<LoDTensor>();
    for (int i = 0; i < t.numel(); ++i) std::cout << t.data<float>()[i] << " ";
    std::cout << std::endl;
  }
}

}  // namespace framework
}  // namespace paddle
