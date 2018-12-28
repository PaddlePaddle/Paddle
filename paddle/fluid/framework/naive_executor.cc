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
#include "paddle/fluid/platform/engine_impl.h"
#include "paddle/fluid/string/pretty_log.h"

DEFINE_int32(engine_size, 1, "");

namespace paddle {
namespace framework {

std::unordered_set<std::string> NaiveExecutor::GetOpInputs(const OpDesc &op) {
  std::unordered_set<std::string> res;
  for (auto &x : op.Inputs()) {
    for (auto &item : x.second) {
      res.insert(item);
    }
  }
  return res;
}
std::unordered_set<std::string> NaiveExecutor::GetOpOutputs(const OpDesc &op) {
  std::unordered_set<std::string> res;
  for (auto &x : op.Outputs()) {
    for (auto &item : x.second) {
      res.insert(item);
    }
  }
  return res;
}

void NaiveExecutor::Prepare(Scope *scope, const ProgramDesc &program_desc,
                            int block_id, bool with_feed_fetch_ops) {
  if (!scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = scope;
  }

  VLOG(3) << "NaiveExecutor init with scope " << scope;
  CreateOps(program_desc, block_id, with_feed_fetch_ops);

  // Create engine resources
  for (size_t i = 0; i < program_desc.Block(0).OpSize(); i++) {
    const auto &op = program_desc.Block(0).Op(i);
    // inputs is enough ?
    auto inputs = GetOpInputs(*op);
    auto outputs = GetOpOutputs(*op);
    inputs.insert(outputs.begin(), outputs.end());

    for (const auto &x : inputs) {
      if (!engine_resources_.count(x)) {
        engine_resources_[x] = engine_->NewResource(x);
      }
    }
  }
}

void NaiveExecutor::Run() {
#ifndef PADDLE_ON_INFERENCE
  LOG_FIRST_N(WARNING, 1) << "The NaiveExecutor can not work properly if the "
                             "cmake flag ON_INFER is not set.";
  LOG_FIRST_N(WARNING, 1) << "Unlike the training phase, all the scopes and "
                             "variables will be reused to save the allocation "
                             "overhead.";
  LOG_FIRST_N(WARNING, 1) << "Please re-compile the inference library by "
                             "setting the cmake flag ON_INFER=ON if you are "
                             "running Paddle Inference";
#endif  // PADDLE_ON_INFERENCE

  for (auto &op : ops_) {
    VLOG(3) << std::this_thread::get_id() << " run " << op->Type()
            << " on scope " << scope_;

    std::unordered_set<std::string> reads, writes;
    for (auto &item : op->Inputs()) {
      for (auto &x : item.second) {
        reads.insert(x);
      }
    }
    for (auto &item : op->Outputs()) {
      for (auto &x : item.second) {
        writes.insert(x);
      }
    }

    // check if reads with writes
    for (auto &x : writes) {
      if (reads.count(x)) {
        reads.erase(x);
      }
    }

    engine::RunContext ctx;

    std::vector<engine::ResourceHandle> inputs, outputs;
    for (auto &x : reads) {
      inputs.push_back(engine_resources_[x]);
    }
    for (auto &x : writes) {
      outputs.push_back(engine_resources_[x]);
    }

    // Push tasks
    engine::Engine::AsyncFn fn = [this, &op](engine::RunContext ctx,
                                             engine::CallbackOnComplete cb) {
      // LOG(INFO) << "real running " << op->DebugStringEx(scope_);
      op->Run(*scope_, place_);
      cb();
    };

    op->SetIsCalledByExecutor(false);

    /*
    for (auto& x : engine_resources_) {
      auto debug =
    static_cast<engine::ThreadedResource*>(x.second.get())->debug_string();
      if (!debug.empty()) {
        LOG(INFO) << debug;
      } else {
      }
    }
    LOG(INFO) << "---------------- depend ends ----------------------";

    LOG(INFO) << ">> push task " << op->DebugStringEx(scope_);
     */
    engine_->PushAsync(fn, ctx, inputs, outputs);
  }

  engine_->WaitForAllFinished();
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
