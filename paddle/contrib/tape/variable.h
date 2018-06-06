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

#include <memory>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/operator.h"  // could be moved to .cc

namespace paddle {
namespace tape {

paddle::framework::Scope& get_global_scope() {
  static paddle::framework::Scope S;
  return S;
}

class Variable;
using VariableHandle = std::shared_ptr<Variable>;

/*
 * Currently it depends on framework::Scope and framework::Variable
 * Later on will only depend on framework::Variable
 */
class Variable {
 public:
  Variable(const std::string pre_fix)
      : desc_(pre_fix + std::to_string(count())) {
    get_global_scope().Var(desc_.Name());
  }

  Variable(const std::string pre_fix, bool is_grad)
      : desc_(pre_fix +
              (is_grad ? framework::kGradVarSuffix : std::to_string(count()))) {
    get_global_scope().Var(desc_.Name());
  }

  ~Variable() { get_global_scope().EraseVars({desc_.Name()}); }

  void InitializeVariable() {
    LOG(INFO) << "Initialzing " << desc_.Name() << " as " << desc_.GetType();
    framework::proto::VarType::Type var_type = desc_.GetType();
    if (var_type == framework::proto::VarType::LOD_TENSOR) {
      get_global_scope().Var(desc_.Name())->GetMutable<framework::LoDTensor>();
      //    } else if (var_type == framework::proto::VarType::SELECTED_ROWS) {
      //      var->GetMutable<SelectedRows>();
    } else {
      PADDLE_THROW("Variable type %d is not in [LOD_TENSOR, SELECTED_ROWS]",
                   var_type);
    }
  }

  VariableHandle Grad() {
    if (grad_ == nullptr) {
      grad_.reset(new Variable(desc_.Name(), true));
    }

    return grad_;
  }

  std::string Name() const { return desc_.Name(); }

  //  void init(const std::string& initializer,
  //            const framework::AttributeMap& attrs);

  framework::VarDesc desc_;

 private:
  int count() {
    static int counter = 0;
    return counter++;
  }

  VariableHandle grad_;
  //  framework::Variable* var_;
};
}
}
