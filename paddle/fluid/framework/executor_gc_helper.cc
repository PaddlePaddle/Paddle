// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/executor_gc_helper.h"
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

struct OpInOutInfo {
 public:
  void Build(const OperatorBase *op) {
    is_built_ = true;
    auto &inferer = op->Info().NoNeedBufferVarsInferer();
    if (inferer) {
      no_need_buffer_ins_ = inferer(op->Inputs(), op->Outputs(), op->Attrs());

      if (no_need_buffer_ins_.empty()) return;

      for (auto &in_name_pair : op->Inputs()) {
        if (no_need_buffer_ins_.count(in_name_pair.first) != 0) {
          continue;
        }

        for (auto &in_arg_name : in_name_pair.second) {
          other_args_set_.insert(in_arg_name);
        }
      }

      for (auto &out_name_pair : op->Outputs()) {
        for (auto &out_arg_name : out_name_pair.second) {
          other_args_set_.insert(out_arg_name);
        }
      }
    }
  }

  bool IsBuilt() const { return is_built_; }

  bool IsInArgBufferNeeded(const std::string &in_arg_name) const {
    return no_need_buffer_ins_.empty() ||
           other_args_set_.count(in_arg_name) != 0;
  }

 private:
  // A set to record unused buffer input vars of op
  std::unordered_set<std::string> no_need_buffer_ins_;
  // A set to record other args of op (including in, out)
  std::unordered_set<std::string> other_args_set_;
  bool is_built_{false};
};

static bool VarCanBeDeleted(const std::string &name, const BlockDesc &block,
                            const std::unordered_set<std::string> &skip_vars) {
  if (skip_vars.count(name) != 0) {
    return false;
  }

  auto *var_desc = block.FindVar(name);
  if (var_desc == nullptr || var_desc->Persistable()) {
    return false;
  }

  auto type = var_desc->Proto()->type().type();

  return type == proto::VarType::LOD_TENSOR ||
         type == proto::VarType::SELECTED_ROWS ||
         type == proto::VarType::LOD_TENSOR_ARRAY;
}

std::unordered_map<const OperatorBase *, std::vector<std::string>>
GetUnusedVars(const BlockDesc &block,
              const std::vector<std::unique_ptr<OperatorBase>> &ops,
              const std::vector<std::string> &skip_var_list) {
  std::unordered_set<std::string> skip_vars(skip_var_list.begin(),
                                            skip_var_list.end());

  std::unordered_map<std::string, size_t> var_op_idx_map;

  for (size_t i = 0; i < ops.size(); ++i) {
    auto *op = ops[i].get();

    OpInOutInfo info;
    for (auto &name_pair : op->Inputs()) {
      for (auto &name : name_pair.second) {
        if (!VarCanBeDeleted(name, block, skip_vars)) {
          continue;
        }

        // var can be gc-ed
        if (!info.IsBuilt()) {
          info.Build(op);
        }

        if (info.IsInArgBufferNeeded(name)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        } else {
          VLOG(10) << "Skip reference count computing of variable "
                   << name_pair.first << "(" << name << ") in Operator "
                   << op->Type();
        }
      }
    }

    for (auto &name_pair : op->Outputs()) {
      for (auto &name : name_pair.second) {
        if (VarCanBeDeleted(name, block, skip_vars)) {
          // Update the last living op of variable to current op
          var_op_idx_map[name] = i;
        }
      }
    }
  }

  std::unordered_map<const OperatorBase *, std::vector<std::string>> result;
  for (auto &name_op_idx_pair : var_op_idx_map) {
    auto &name = name_op_idx_pair.first;
    size_t op_idx = name_op_idx_pair.second;
    result[ops[op_idx].get()].emplace_back(name);
  }
  return result;
}

void DeleteUnusedTensors(
    const Scope &scope, const OperatorBase *op,
    const std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &delete_vars_map,
    GarbageCollector *gc) {
  auto iter = delete_vars_map.find(op);
  if (iter == delete_vars_map.end()) {
    return;
  }

  auto &delete_vars = iter->second;

  std::deque<std::shared_ptr<memory::Allocation>> garbages;

  for (auto &var_name : delete_vars) {
    auto *var = scope.FindVar(var_name);
    if (var == nullptr) {
      continue;
    }

    VLOG(2) << "Erase variable " << var_name;
    if (var->IsType<LoDTensor>()) {
      garbages.emplace_back(var->GetMutable<LoDTensor>()->MoveMemoryHolder());
    } else if (var->IsType<SelectedRows>()) {
      garbages.emplace_back(
          var->GetMutable<SelectedRows>()->mutable_value()->MoveMemoryHolder());
    } else if (var->IsType<LoDTensorArray>()) {
      auto *lod_tensor_arr = var->GetMutable<LoDTensorArray>();
      for (auto &t : *lod_tensor_arr) {
        garbages.emplace_back(t.MoveMemoryHolder());
      }
    } else {
      PADDLE_THROW("Type %s of %s is not supported eager deletion",
                   framework::ToTypeName(var->Type()), var_name);
    }
  }

  if (!garbages.empty()) {
    gc->Add(std::move(garbages));
  }
}

}  // namespace framework
}  // namespace paddle
