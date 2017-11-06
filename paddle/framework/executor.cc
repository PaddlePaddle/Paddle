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
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";

Executor::Executor(const std::vector<platform::Place>& places) : own_(true) {
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
  if (own_) {
    for (auto& device_context : device_contexts_) {
      delete device_context;
    }
  }
}

static void CreateTensor(Variable* var, VarDesc::VarType var_type) {
  if (var_type == VarDesc::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == VarDesc::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == VarDesc::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == VarDesc::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == VarDesc::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope>>();
  } else if (var_type == VarDesc::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == VarDesc::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LoDTensor, SelectedRows, FEED_MINIBATCH, FETCH_LIST, LOD_RANK_TABLE]",
        var_type);
  }
}

void Executor::Run(const ProgramDescBind& pdesc, Scope* scope, int block_id,
                   bool create_local_scope) {
  // TODO(tonyyang-svail):
  //    - only runs on the first device (i.e. no interdevice communication)
  //    - will change to use multiple blocks for RNN op and Cond Op
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), pdesc.Size());
  auto& block = pdesc.Block(block_id);
  auto& device = device_contexts_[0];

  Scope* local_scope = scope;
  if (create_local_scope) {
    local_scope = &scope->NewScope();
    for (auto& var : block.AllVars()) {
      if (var->Persistable()) {
        auto* ptr = scope->Var(var->Name());
        CreateTensor(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " global, which pointer is " << ptr;
      } else {
        auto* ptr = local_scope->Var(var->Name());
        CreateTensor(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " locally, which pointer is " << ptr;
      }
    }
  } else {
    for (auto& var : block.AllVars()) {
      auto* ptr = local_scope->Var(var->Name());
      CreateTensor(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
              << ptr;
    }
  }

  for (auto& op_desc : block.AllOps()) {
    auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
    op->Run(*local_scope, *device);
  }
  if (create_local_scope) {
    scope->DeleteScope(local_scope);
  }
}

Executor::Executor(const platform::DeviceContext& device)
    : device_contexts_({&device}), own_(false) {}

}  // namespace framework
}  // namespace paddle
