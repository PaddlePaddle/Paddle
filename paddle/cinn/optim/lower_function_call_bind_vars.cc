// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/lower_function_call_bind_vars.h"

#include <string>
#include <vector>

#include "paddle/cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct LowerFunctionCallBindVarsMutator : public ir::IRMutator<> {
  LowerFunctionCallBindVarsMutator() = default;

  void operator()(Expr* m) {
    m_ = m->as_module();
    Expr module(m->get());
    ir::IRMutator<>::Visit(&module, &module);
  }

 private:
  void Visit(const ir::Call* op, Expr* expr) {
    auto* node = expr->As<ir::Call>();
    if (op->is_cinn_call()) {
      const std::string& target = op->name;
      auto it = std::find_if(
          m_->functions.begin(), m_->functions.end(), [&](const Expr& x) {
            return x.as_lowered_func()->name == target;
          });
      CHECK(it != m_->functions.end())
          << "The called function [" << target << "] is not exist";

      std::vector<Expr> extra_var_args;

      for (auto& arg : (*it).as_lowered_func()->args) {
        if (arg.is_var()) {
          extra_var_args.push_back(arg.var_arg());
        }
      }

      // insert the extra var arguments to the begining of the original call's
      // argument list.
      node->read_args.insert(std::begin(op->read_args),
                             extra_var_args.begin(),
                             extra_var_args.end());
    }

    ir::IRMutator<>::Visit(op, expr);
  }

 private:
  ir::_Module_* m_{};
};

}  // namespace

void LowerFunctionCallBindVars(Expr* m) {
  CHECK(m->as_module());
  LowerFunctionCallBindVarsMutator()(m);
}

}  // namespace optim
}  // namespace cinn
