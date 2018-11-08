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
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {

// These code can be shared with Executor.
static void InitializeVariable(Variable *var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope *>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, CHANNEL, RAW]",
        var_type);
  }
}

void NaiveExecutor::Prepare(Scope *parent_scope,
                            const ProgramDesc &program_desc, int block_id,
                            bool with_feed_fetch_ops) {
  if (!parent_scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = &parent_scope->NewScope();
  }
  CreateVariables(program_desc, scope_, block_id);
  CreateOps(program_desc, block_id, with_feed_fetch_ops);
}

void NaiveExecutor::Run() {
  for (auto &op : ops_) {
    VLOG(40) << "run " << op->Type();
    op->Run(*scope_, place_);
  }
}

void NaiveExecutor::CreateVariables(const ProgramDesc &desc, Scope *scope,
                                    int block_id) {
  PADDLE_ENFORCE(scope);
  auto &global_block = desc.Block(block_id);

  const Scope *ancestor_scope = scope;
  while (ancestor_scope->parent()) {
    ancestor_scope = ancestor_scope->parent();
  }

  if (ancestor_scope != scope) {
    for (auto &var : global_block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) {
        continue;
      }
      // Create persistable vars in ancestor scope.
      if (var->Persistable()) {
        auto *ptr = const_cast<Scope *>(ancestor_scope)->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(30) << "Create Variable " << var->Name()
                 << " global, which pointer is " << ptr;
      } else {  // Create temporary variables in local scope.
        auto *ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(30) << "Create Variable " << var->Name()
                 << " locally, which pointer is " << ptr;
      }
    }
  } else {
    for (auto &var : global_block.AllVars()) {
      auto *ptr = scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(30) << "Create variable " << var->Name() << ", which pointer is "
               << ptr;
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
