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

#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace imperative {

class OpBase;

class VarBase {
 public:
  VarBase() {}
  virtual ~VarBase() {}

  OpBase* pre_op_;
  framework::VarDesc* var_desc_;
};

class OpBase {
 public:
  OpBase()
      : input_vars_(new std::vector<VarBase*>()),
        output_vars_(new std::vector<VarBase*>()) {}
  virtual ~OpBase() {
    delete input_vars_;
    delete output_vars_;
  }

  std::vector<VarBase*>* input_vars_;
  std::vector<VarBase*>* output_vars_;
  framework::OpDesc* op_desc_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<VarBase> Forward(const std::vector<VarBase>& inputs) {
    std::vector<VarBase> vars;
    return vars;
  }

  virtual void Backward() { LOG(ERROR) << "backward at cpp."; }
};

}  // namespace imperative
}  // namespace paddle
