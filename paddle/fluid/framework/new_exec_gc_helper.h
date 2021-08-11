// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

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

bool VarCanBeDeleted(const std::string &name, const BlockDesc &block) {
  auto *var_desc = block.FindVar(name);
  if (var_desc == nullptr || var_desc->Persistable()) {
    return false;
  }

  auto type = var_desc->Proto()->type().type();

  return type == proto::VarType::LOD_TENSOR ||
         type == proto::VarType::SELECTED_ROWS ||
         type == proto::VarType::LOD_TENSOR_ARRAY;
}

std::unordered_map<const paddle::framework::OperatorBase *,
                   std::vector<std::string>>
GetUnusedVars(const BlockDesc &block, const std::vector<OperatorBase *> &ops) {
  std::unordered_map<std::string, size_t> var_op_idx_map;

  for (size_t i = 0; i < ops.size(); ++i) {
    auto *op = ops[i];

    OpInOutInfo info;
    for (auto &name_pair : op->Inputs()) {
      for (auto &name : name_pair.second) {
        if (!VarCanBeDeleted(name, block)) {
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
        if (VarCanBeDeleted(name, block)) {
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
    result[ops[op_idx]].emplace_back(name);
  }
  return result;
}

}  // namespace framework
}  // namespace paddle
