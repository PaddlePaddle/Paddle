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

#include "paddle/operators/recurrent_op.h"

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

int getBlockIdx(const OpDesc& op_desc) {
  for (auto& attr : op_desc.attrs()) {
    if (attr.has_block_idx()) {
      return attr.block_idx();
    }
  }

  PADDLE_THROW("Missing block_idx in recurrent opDesc");
  return -1;
}

std::unique_ptr<OperatorBase> create_op(const ProgramDesc& pdesc,
                                        const OpDesc& op_desc) {
  auto op = paddle::framework::OpRegistry::CreateOp(
      op_desc, const_cast<ProgramDesc*>(&pdesc));

  if (op_desc.type() == "recurrent") {
    int block_idx = getBlockIdx(op_desc);
    std::unique_ptr<std::vector<std::unique_ptr<OperatorBase>>> step_net{
        new std::vector<std::unique_ptr<OperatorBase>>};
    for (auto& my_op_desc : pdesc.blocks(block_idx).ops()) {
      step_net->push_back(create_op(pdesc, my_op_desc));
    }
    if (auto* rnn_op = dynamic_cast<operators::RecurrentOp*>(op.get())) {
      std::vector<std::string> vars;
      for (auto& var : pdesc.blocks(block_idx).vars()) {
        vars.push_back(var.name());
      }
      rnn_op->set_stepnet(step_net, vars);
    } else {
      PADDLE_THROW("dynamic_cast<RecurrentOp*> fail");
    }
    VLOG(3) << "GO";
  }

  return op;
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
    var->GetMutable<std::vector<Scope*>>();
  } else {
    PADDLE_THROW(
        "Variable type must be "
        "LoDTensor/SelectedRows/FEED_MINIBATCH/FETCH_LIST/STEP_SCOPES.");
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
      VLOG(3) << "Create Variable " << var.name();
      auto* ptr = scope->Var(var.name());
      CreateTensor(ptr, var.type());
    } else {
      VLOG(3) << "Create Variable " << var.name();
      auto* ptr = local_scope.Var(var.name());
      CreateTensor(ptr, var.type());
    }
  }

  for (auto& op_desc : block.ops()) {
    VLOG(2) << op_desc.type();
    auto op = create_op(pdesc, op_desc);
    op->Run(local_scope, *device);
  }

  for (auto& var : block.vars()) {
    std::set<std::string> name_to_print{"a", "b", "h_boot"};
    if (!var.persistable() && name_to_print.count(var.name())) {
      VLOG(2) << var.name();
      auto* v = local_scope.Var(var.name());
      const float* f = v->GetMutable<LoDTensor>()->data<float>();
      const int64_t s = v->GetMutable<LoDTensor>()->numel();
      for (int i = 0; i < s; ++i) {
        VLOG(10) << f[i];
      }
    }
  }

  scope->DeleteScope(&local_scope);
}

}  // namespace framework
}  // namespace paddle
