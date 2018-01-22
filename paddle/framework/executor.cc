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

#include <set>

#include "gflags/gflags.h"
#include "paddle/framework/feed_fetch_type.h"
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/place.h"

DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";

Executor::Executor(const platform::Place& place) : place_(place) {}

static void CreateTensor(Variable* var, proto::VarDesc::VarType var_type) {
  if (var_type == proto::VarDesc::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarDesc::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarDesc::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarDesc::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarDesc::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope>>();
  } else if (var_type == proto::VarDesc::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarDesc::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarDesc::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LoDTensor, SelectedRows, FEED_MINIBATCH, FETCH_LIST, LOD_RANK_TABLE,"
        " PLACE_LIST]",
        var_type);
  }
}

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type().hash_code() != typeid(float).hash_code() &&
      tensor.type().hash_code() != typeid(double).hash_code()) {
    return;
  }
  PADDLE_ENFORCE(!framework::HasInf(tensor), "Tensor %s has Inf", name);
  PADDLE_ENFORCE(!framework::HasNAN(tensor), "Tensor %s has NAN", name);
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id,
                   bool create_local_scope, bool create_vars) {
  // TODO(tonyyang-svail):
  //    - only runs on the first device (i.e. no interdevice communication)
  //    - will change to use multiple blocks for RNN op and Cond Op
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), pdesc.Size());
  auto& block = pdesc.Block(block_id);

  Scope* local_scope = scope;
  if (create_vars) {
    if (create_local_scope) {
      local_scope = &scope->NewScope();
      for (auto& var : block.AllVars()) {
        if (var->Name() == framework::kEmptyVarName) {
          continue;
        }

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
    }  // if (create_local_scope)
  }    // if (create_vars)

  for (auto& op_desc : block.AllOps()) {
    auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
    VLOG(3) << op->DebugStringEx(local_scope);
    op->Run(*local_scope, place_);
    if (FLAGS_check_nan_inf) {
      for (auto& vname : op->OutputVars(true)) {
        auto* var = local_scope->FindVar(vname);
        if (var == nullptr) continue;
        if (var->IsType<framework::LoDTensor>()) {
          CheckTensorNANOrInf(vname, var->Get<framework::LoDTensor>());
        }
      }
    }
  }
  if (create_vars && create_local_scope) {
    scope->DeleteScope(local_scope);
  }
}

void Executor::Run(const ProgramDesc& program, Scope* scope,
                   std::map<std::string, LoDTensor>& feeds,
                   std::map<std::string, LoDTensor>& fetchs,
                   const std::string& feed_var_name,
                   const std::string& fetch_var_name) {
  auto* copy_program = new ProgramDesc(program);
  auto* global_block = copy_program->MutableBlock(0);

  VarDesc* feed_var = nullptr;
  VarDesc* fetch_var = nullptr;
  for (auto& var_desc : global_block->AllVars()) {
    if (var_desc->GetType() == proto::VarDesc::FEED_MINIBATCH) {
      PADDLE_ENFORCE(var_desc->Name() == feed_var_name,
                     "The feed variable name should match the program desc");
      feed_var = var_desc;
    } else if (var_desc->GetType() == proto::VarDesc::FETCH_LIST) {
      PADDLE_ENFORCE(var_desc->Name() == fetch_var_name,
                     "The fetch variable name should match the program desc");
      fetch_var = var_desc;
    }
  }
  if (feed_var == nullptr) {
    feed_var = global_block->Var(feed_var_name);
    feed_var->SetType(proto::VarDesc::FEED_MINIBATCH);
    feed_var->SetPersistable(true);
  }
  if (fetch_var == nullptr) {
    fetch_var = global_block->Var(fetch_var_name);
    fetch_var->SetType(proto::VarDesc::FETCH_LIST);
    feed_var->SetPersistable(true);
  }

  // if there are feed and fetch ops, check that they match the feeds info
  size_t feed_count = 0;
  size_t fetch_count = 0;
  for (auto* op : global_block->AllOps()) {
    if (op->Type() == "feed") {
      feed_count++;
      PADDLE_ENFORCE(op->Input("X")[0] == feed_var_name,
                     "Input to feed op should be '%s'", feed_var_name);
      auto it = feeds.find(op->Output("Out")[0]);
      PADDLE_ENFORCE(
          it != feeds.end(),
          "Feed operator output name '%s' should match the info in 'feeds'",
          op->Output("Out")[0]);
      Attribute attr = op->GetAttr("col");
      PADDLE_ENFORCE(attr.type() == typeid(int),
                     "Attribute type of 'col' should be int");
      //      SetFeedVariable(scope, it->second, feed_var_name,
      //      boost::get<int>(attr));
    } else if (op->Type() == "fetch") {
      fetch_count++;
      PADDLE_ENFORCE(op->Output("Out")[0] == fetch_var_name,
                     "Output of fetch op should be '%s'", fetch_var_name);
      std::string fetch_input_name = op->Input("X")[0];
      auto it = fetchs.find(fetch_input_name);
      PADDLE_ENFORCE(
          it != fetchs.end(),
          "Fetch operator input name '%s' should match the info in 'fetchs'",
          fetch_input_name);
      int index = 0;
      for (auto& fetch_item : fetchs) {
        if (fetch_item.first == fetch_input_name) {
          break;
        }
        index++;
      }
      Attribute attr = op->GetAttr("col");
      PADDLE_ENFORCE(attr.type() == typeid(int),
                     "Attribute type of 'col' should be int");
      PADDLE_ENFORCE(boost::get<int>(attr) == index);
    }
  }

  if (feed_count > 0) {
    PADDLE_ENFORCE(feed_count == feeds.size(),
                   "The number of fetch operators should match 'feeds'");
  }
  if (fetch_count > 0) {
    PADDLE_ENFORCE(fetch_count == fetchs.size(),
                   "The number of fetch operators should match 'feeds'");
  }

  // append fetch_op and related variable
  if (feed_count == 0) {
    int i = 0;
    for (auto& feed_item : feeds) {
      std::string var_name = feed_item.first;
      LOG(INFO) << "feed var's name: " << var_name;

      // prepend_op
      auto* op = global_block->PrependOp();
      op->SetType("feed");
      op->SetInput("X", {feed_var_name});
      op->SetOutput("Out", {var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      //      SetFeedVariable(scope, feed_item.second, feed_var_name, i);
      i++;
    }
  }

  if (fetch_count == 0) {
    int i = 0;
    for (auto& fetch_item : fetchs) {
      std::string var_name = fetch_item.first;
      LOG(INFO) << "fetch var's name: " << var_name;

      // append_op
      auto* op = global_block->AppendOp();
      op->SetType("fetch");
      op->SetInput("X", {var_name});
      op->SetOutput("Out", {fetch_var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  Run(*copy_program, scope, 0, true, true);

  // get fetch variables
  int i = 0;
  for (auto& fetch_item : fetchs) {
    //    fetch_item.second = GetFetchVariable(*scope, fetch_var_name, i);
    LOG(INFO) << fetch_item.first;
    i++;
  }

  delete copy_program;
}

}  // namespace framework
}  // namespace paddle
