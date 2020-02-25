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

#include <memory>
#include <string>
#include <utility>
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
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    op->SetIsCalledByExecutor(false);
    op->Run(*scope_, place_);
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

  int num_vars = 0;
  for (auto &var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    num_vars++;

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
  VLOG(4) << "naive executor create " << num_vars << " vars";
}

void NaiveExecutor::CreateOps(const ProgramDesc &desc, int block_id,
                              bool with_feed_fetch_ops) {
  for (const auto &op_desc : desc.Block(block_id).AllOps()) {
    if (!with_feed_fetch_ops &&
        (op_desc->Type() == "feed" || op_desc->Type() == "fetch")) {
      LOG(INFO) << "---  skip [" << op_desc->Input("X")[0] << "], "
                << op_desc->Type() << " -> " << op_desc->Output("Out")[0];
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
