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

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "paddle/framework/feed_fetch_type.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";

Executor::Executor(const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(), 0);
  device_contexts_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    if (platform::is_cpu_place(places[i])) {
      device_contexts_[i] = new platform::CPUDeviceContext(
          boost::get<platform::CPUPlace>(places[i]));
    } else if (platform::is_gpu_place(places[i])) {
#ifdef PADDLE_WITH_CUDA
      device_contexts_[i] = new platform::CUDADeviceContext(
          boost::get<platform::GPUPlace>(places[i]));
#else
      PADDLE_THROW(
          "'GPUPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    }
  }
}

Executor::~Executor() {
  for (auto& device_context : device_contexts_) {
    delete device_context;
  }
}

void CreateTensor(Variable* var, VarDesc::VarType var_type) {
  LOG(INFO) << var_type;
  if (var_type == VarDesc::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == VarDesc::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == VarDesc::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == VarDesc::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else {
    PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id) {
  // TODO(tonyyang-svail):
  //    - only runs on the first device (i.e. no interdevice communication)
  //    - will change to use multiple blocks for RNN op and Cond Op
  PADDLE_ENFORCE_GT(pdesc.blocks_size(), block_id);
  auto& block = pdesc.blocks(block_id);
  auto& device = device_contexts_[0];

  Scope& local_scope = scope->NewScope();

  for (auto& var : block.vars()) {
    if (var.persistable()) {
      auto* ptr = scope->Var(var.name());
      CreateTensor(ptr, var.type());
      VLOG(3) << "Create Variable " << var.name()
              << " global, which pointer is " << ptr;
    } else {
      auto* ptr = local_scope.Var(var.name());
      CreateTensor(ptr, var.type());
      VLOG(3) << "Create Variable " << var.name()
              << " locally, which pointer is " << ptr;
    }
  }

  for (auto& op_desc : block.ops()) {
    auto op = paddle::framework::OpRegistry::CreateOp(
        op_desc, const_cast<ProgramDesc*>(&pdesc));
    op->Run(local_scope, *device);
  }

  scope->DeleteScope(&local_scope);
}

}  // namespace framework
}  // namespace paddle
