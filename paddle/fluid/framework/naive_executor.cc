// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
void NaiveExecutor::Prepare(Scope *scope, const ProgramDesc &program_desc,
                            int block_id, bool with_feed_fetch_ops) {
  if (!scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = scope;
  }

  VLOG(3) << "NaiveExecutor init with scope " << scope;
  CreateOps(program_desc, block_id, with_feed_fetch_ops);
}

void NaiveExecutor::Run() {
#ifndef PADDLE_ON_INFERENCE
  LOG_FIRST_N(WARNING, 5) << "The NaiveExecutor can not work properly if the "
                             "cmake flag ON_INFER is not set.";
  LOG_FIRST_N(WARNING, 5) << "Unlike the training phase, all the scopes and "
                             "variables will be reused to save the allocation "
                             "overhead.";
  LOG_FIRST_N(WARNING, 5) << "Please re-compile the inference library by "
                             "setting the cmake flag ON_INFER=ON if you are "
                             "running Paddle Inference";
#endif  // PADDLE_ON_INFERENCE

  if (op_prepared_) {
    LOG(INFO) << "prepared run...";
    for (auto &op : prepared_ops_) {
      op.Run(scope_, place_);
    }
  } else {
    LOG(INFO) << "parepared run...";
    for (auto &op : ops_) {
      op->SetIsCalledByExecutor(false);
      op->Run(*scope_, place_);

      if (dynamic_cast<OperatorWithKernel *>(op.get())) {
        LOG(INFO) << "get op kernel";
        // prepare ops
        framework::OperatorWithKernel *op_kernel =
            dynamic_cast<framework::OperatorWithKernel *>(op.get());
        op->SetIsCalledByExecutor(false);
        framework::RuntimeContext ctx(op->Inputs(), op->Outputs(), *scope_);
        PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");
        prepared_ops_.emplace_back();
        auto &latest = prepared_ops_.back();
        latest.is_kernel = true;
        latest.kernel = PreparedKernel::Prepare(ctx, *op_kernel, place_);
      } else {
        LOG(INFO) << "get op";
        prepared_ops_.emplace_back();
        auto &latest = prepared_ops_.back();
        latest.is_kernel = false;
        framework::RuntimeContext ctx(op->Inputs(), op->Outputs(), *scope_);

        platform::DeviceContextPool &pool =
            platform::DeviceContextPool::Instance();
        auto *dev_ctx = pool.Get(place_);
        latest.op.reset(new PreparedOp({*op, ctx, dev_ctx}));
      }
    }

    op_prepared_ = true;
  }
}

void NaiveExecutor::CreateVariables(const ProgramDesc &desc, int block_id,
                                    bool persistable, Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope);

  auto &global_block = desc.Block(block_id);

  const auto *anc = scope;
  PADDLE_ENFORCE(anc->parent() != anc);
  while (anc->parent()) {
    anc = anc->parent();
  }

  for (auto &var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }

    if (persistable == var->Persistable()) {
      if (persistable) {
        if (!anc->FindVar(var->Name())) {
          auto *ptr = const_cast<Scope *>(anc)->Var(var->Name());
          VLOG(3) << scope << " Create persistable variable " << var->Name()
                  << ", which pointer is " << ptr;
          InitializeVariable(ptr, var->GetType());
        }
      } else {
        auto *ptr = const_cast<Scope *>(scope)->Var(var->Name());
        VLOG(3) << scope << " Create variable " << var->Name()
                << ", which pointer is " << ptr;
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
}

void NaiveExecutor::CreateOps(const ProgramDesc &desc, int block_id,
                              bool with_feed_fetch_ops) {
  for (const auto &op_desc : desc.Block(block_id).AllOps()) {
    if (!with_feed_fetch_ops &&
        (op_desc->Type() == "feed" || op_desc->Type() == "fetch")) {
      string::PrettyLogEndl(string::Style::detail(), "---  skip [%s], %s -> %s",
                            op_desc->Input("X")[0], op_desc->Type(),
                            op_desc->Output("Out")[0]);
      continue;
    }
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
  }
}

LoDTensor *NaiveExecutor::FindTensor(const std::string &name) {
  PADDLE_ENFORCE(scope_, "Need to init scope first");
  auto *var = scope_->FindVar(name);
  PADDLE_ENFORCE(var, "No variable [%s] in the scope");
  auto *tensor = const_cast<LoDTensor *>(&var->Get<LoDTensor>());
  return tensor;
}

void NaiveExecutor::CleanFeedFetchOps() {
  std::vector<std::unique_ptr<OperatorBase>> ops;
  for (auto &op : ops_) {
    if (op->Type() != "feed" && op->Type() != "fetch") {
      ops.emplace_back(std::move(op));
    }
  }
  ops_.swap(ops);
}

}  // namespace framework
}  // namespace paddle
