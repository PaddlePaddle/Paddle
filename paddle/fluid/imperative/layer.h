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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
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
        var_(nullptr),
        grads_(nullptr) {}

  virtual ~VarBase() {
    LOG(ERROR) << "deleting var";
    LOG(ERROR) << "done deleting var";
  }

  void ApplyGrad(framework::Scope* scope, framework::Variable* grad);

  void RunBackward(framework::Scope* scope);

  framework::LoDTensor& Grad();

  OpBase* pre_op_;
  int pre_op_out_idx_;

  framework::VarDesc* var_desc_;
  framework::Variable* var_;
  framework::Variable* grads_;
};

class OpBase {
 public:
  OpBase()
      : input_vars_(new std::vector<VarBase*>()),
        output_vars_(new std::vector<VarBase*>()),
        pre_ops_(new std::vector<OpBase*>()),
        pre_ops_out_idx_(new std::vector<int>()),
        op_desc_(nullptr),
        grad_op_desc_(nullptr) {}

  virtual ~OpBase() {
    delete input_vars_;
    delete output_vars_;

    delete pre_ops_;
    delete pre_ops_out_idx_;

    if (grad_op_desc_) delete grad_op_desc_;
    if (grad_to_var_) delete grad_to_var_;
  }

  std::vector<framework::Variable*> ApplyGrad(framework::Scope* scope);

  std::vector<VarBase*>* input_vars_;
  std::vector<VarBase*>* output_vars_;
  std::vector<OpBase*>* pre_ops_;
  std::vector<int>* pre_ops_out_idx_;
  framework::OpDesc* op_desc_;

  framework::OpDesc* grad_op_desc_;
  std::unordered_map<std::string, std::string>* grad_to_var_;
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
