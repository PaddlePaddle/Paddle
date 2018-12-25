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

#include <map>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

class OpBase;

class VarBase {
 public:
  VarBase()
      : pre_op_(nullptr),
        pre_op_out_idx_(-1),
        var_desc_(nullptr),
        var_(new framework::Variable()),
        grads_(new framework::Variable()) {}

  virtual ~VarBase() {
    if (var_) {
      delete var_;
      var_ = nullptr;
    }
    if (grads_) {
      delete grads_;
      grads_ = nullptr;
    }
  }

  void RunBackward();

  framework::LoDTensor& Grad();

  OpBase* pre_op_;
  std::string pre_op_out_name_;
  int pre_op_out_idx_;

  framework::VarDesc* var_desc_;
  framework::Variable* var_;
  framework::Variable* grads_;
};

class OpBase {
 public:
  OpBase()
      : pre_ops_(new std::map<std::string, std::vector<OpBase*>>()),
        pre_ops_out_idx_(new std::map<std::string, std::vector<int>>()),
        op_desc_(nullptr),
        grad_op_desc_(nullptr) {}

  virtual ~OpBase() {
    delete pre_ops_;
    delete pre_ops_out_idx_;

    if (grad_op_desc_) delete grad_op_desc_;
    if (grad_to_var_) delete grad_to_var_;
  }

  std::map<std::string, std::vector<VarBase*>> ApplyGrad();

  std::map<std::string, std::vector<VarBase*>> input_vars_;
  std::map<std::string, std::vector<VarBase*>> output_vars_;
  std::map<std::string, std::vector<OpBase*>>* pre_ops_;
  std::map<std::string, std::vector<int>>* pre_ops_out_idx_;
  framework::OpDesc* op_desc_;

  framework::OpDesc* grad_op_desc_;
  std::unordered_map<std::string, std::string>* grad_to_var_;
  std::map<std::string, std::vector<framework::Variable*>> grad_input_vars_;
  std::map<std::string, std::vector<framework::Variable*>> grad_output_vars_;
  framework::BlockDesc* block_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<VarBase> Forward(const std::vector<VarBase>& inputs) {
    std::vector<VarBase> vars;
    return vars;
  }

  virtual void Backward() { LOG(ERROR) << "To support customize"; }
};

}  // namespace imperative
}  // namespace paddle
