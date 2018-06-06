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
#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

#include "paddle/contrib/tape/variable.h"

namespace paddle {
namespace tape {

using VariableHandleMap = std::map<std::string, std::vector<VariableHandle>>;

// borrowed from
// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
inline bool ends_with(std::string const& value, std::string const& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

struct OpHandle {
  OpHandle(const std::string& type,
           const VariableHandleMap& in_vars,
           const VariableHandleMap& out_vars,
           const framework::AttributeMap& attrs)
      : type_(type), in_vars_(in_vars), out_vars_(out_vars), attrs_(attrs) {}

  std::string type_;
  VariableHandleMap in_vars_;
  VariableHandleMap out_vars_;
  framework::AttributeMap attrs_;
};

std::ostream& operator<<(std::ostream& os, const framework::VarDesc& var_desc) {
  os << var_desc.Name();
  os << "[" << var_desc.GetType() << "]";
  os << "[" << var_desc.GetDataType() << "]";
  os << "{";
  for (auto& i : var_desc.GetShape()) {
    os << i << ",";
  }
  os << "}";
  return os;
}

std::string to_string(const std::string& type,
                      const VariableHandleMap& in_vars,
                      const VariableHandleMap& out_vars,
                      const framework::AttributeMap& attrs) {
  std::stringstream ss;
  ss << type << " ";
  for (auto& param_name : in_vars) {
    for (auto& var : param_name.second) {
      ss << param_name.first << ":(" << var->Desc() << ") ";
    }
  }
  for (auto& param_name : out_vars) {
    for (auto& var : param_name.second) {
      ss << param_name.first << ":(" << var->Desc() << ") ";
    }
  }
  return ss.str();
}

class Tape {
 public:
  void forward() {
    PADDLE_ENFORCE(!has_been_backwarded_);
    LOG(INFO) << "Starting forward -------------------------";
    while (current_position_ < tape_.size()) {
      OpHandle& op = tape_[current_position_];

      // Create Output Tensor, this is only necessary for OpWithKernel
      for (auto& param2var : op.out_vars_) {
        for (auto& var : param2var.second) {
          var->InitializeVariable();
        }
      }

      framework::OpDesc op_desc =
          CreateOpDesc(op.type_, op.in_vars_, op.out_vars_, op.attrs_);
      framework::OpRegistry::CreateOp(op_desc)->Run(get_global_scope(),
                                                    platform::CPUPlace());
      current_position_++;
    }

    LOG(INFO) << "Finishing forward -------------------------";
  }

  void backward(VariableHandle target) {
    PADDLE_ENFORCE(!has_been_backwarded_);

    forward();

    // TODO(tonyyang-svail): check output of last op is target

    backward_tape_.reset(new Tape());

    framework::AttributeMap attrs;
    LOG(INFO) << target->Desc().GetDataType();
    // FIXME(tonyyang-svail): the current infer_data_type logic is written in
    // python, Need a infer_data_type on the C++
    attrs["dtype"] = framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = 1.0f;
    backward_tape_->AddOp(
        "fill_constant", {}, {{"Out", {target->Grad()}}}, attrs);

    for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
      framework::OpDesc op_desc =
          CreateOpDesc(it->type_, it->in_vars_, it->out_vars_, it->attrs_);
      std::unordered_map<std::string, std::string> grad_to_var;
      std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs =
          framework::OpInfoMap::Instance()
              .Get(op_desc.Type())
              .GradOpMaker()(op_desc, {}, &grad_to_var, {});

      for (auto& op_desc : grad_op_descs) {
        std::unordered_map<std::string, VariableHandle> name2var;
        for (auto& param2vars : it->in_vars_) {
          for (auto& a : param2vars.second) {
            name2var[a->Name()] = a;
          }
        }
        for (auto& param2vars : it->out_vars_) {
          for (auto& a : param2vars.second) {
            name2var[a->Name()] = a;
          }
        }

        VariableHandleMap in_vars;
        VariableHandleMap out_vars;
        std::map<const framework::VariableNameMap*, VariableHandleMap*>
            loop_over{{&op_desc->Inputs(), &in_vars},
                      {&op_desc->Outputs(), &out_vars}};
        for (auto& each : loop_over) {
          auto& vmp = *each.first;
          auto& vhm = *each.second;
          for (auto& p2a : vmp) {
            for (auto& argu : p2a.second) {
              if (name2var.count(argu)) {
                vhm[p2a.first].push_back(name2var[argu]);
              } else {
                PADDLE_ENFORCE(ends_with(argu, framework::kGradVarSuffix),
                               argu.c_str());
                std::string name = argu.substr(
                    0, argu.size() - std::strlen(framework::kGradVarSuffix));
                PADDLE_ENFORCE(name2var.count(name), name.c_str());
                vhm[p2a.first].push_back(name2var[name]->Grad());
              }
            }
          }
        }

        backward_tape_->AddOp(
            op_desc->Type(), in_vars, out_vars, op_desc->GetAttrMap());
      }

      // TODO(tonyyang-svail): Sum var grad is necessary
    }

    backward_tape_->forward();
    has_been_backwarded_ = true;
  }

  void AddOp(const std::string& type,
             VariableHandleMap in_vars,
             VariableHandleMap out_vars,
             const framework::AttributeMap& attrs) {
    // InferShape and InferVarType --------------------------------
    framework::OpDesc op_desc = CreateOpDesc(type, in_vars, out_vars, attrs);

    // Create a temporary block for compile-time
    framework::ProgramDesc program_desc;
    framework::BlockDesc* block_desc = program_desc.MutableBlock(0);
    PADDLE_ENFORCE(block_desc);

    for (auto& param_name : in_vars) {
      for (auto& var : param_name.second) {
        *block_desc->Var(var->Name())->Proto() = *var->MutableDesc()->Proto();
      }
    }
    for (auto& param_name : out_vars) {
      for (auto& var : param_name.second) {
        *block_desc->Var(var->Name())->Proto() = *var->MutableDesc()->Proto();
      }
    }

    LOG(INFO) << "- " << to_string(type, in_vars, out_vars, attrs);
    op_desc.InferShape(*block_desc);
    op_desc.InferVarType(block_desc);
    for (auto& param_name : out_vars) {
      for (auto& var : param_name.second) {
        *var->MutableDesc()->Proto() = *block_desc->Var(var->Name())->Proto();
      }
    }
    LOG(INFO) << "+ " << to_string(type, in_vars, out_vars, attrs);

    tape_.emplace_back(type, in_vars, out_vars, attrs);
  }

 private:
  framework::OpDesc CreateOpDesc(const std::string& type,
                                 const VariableHandleMap& in_vars,
                                 const VariableHandleMap& out_vars,
                                 const framework::AttributeMap& attrs) {
    framework::VariableNameMap inputs;
    for (auto& param_name : in_vars) {
      for (auto& var : param_name.second) {
        inputs[param_name.first].emplace_back(var->Name());
      }
    }
    framework::VariableNameMap outputs;
    for (auto& param_name : out_vars) {
      for (auto& var : param_name.second) {
        outputs[param_name.first].emplace_back(var->Name());
      }
    }
    return framework::OpDesc(type, inputs, outputs, attrs);
  }

  bool has_been_backwarded_ = false;
  size_t current_position_ = 0;

  std::vector<OpHandle> tape_;
  std::shared_ptr<Tape> backward_tape_;
};

Tape& get_global_tape() {
  static Tape T;
  return T;
}
}
}
