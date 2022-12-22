// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/eager/api/prims/static_global_utils.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"
namespace paddle {
namespace prims {

/*
  This functor class is responsible for creating the gradient ops for the given
  operator fwd_op. After it is called (through operator()), the pairs of
  (gradient variable, corresponding input variable of fwd_op) will be added to
  grad_to_var. If an input variable of fwd_op is contained in no_grad_set, its
  gradient variable will be ignored or kEmptyVarName depending on the template
  argument DropEmptyIG in the derived classes.
 */

class GradCompositeOpMakerBase {
 public:
  explicit GradCompositeOpMakerBase(
      const framework::OpDesc& fwd_op,
      const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      framework::BlockDesc* current_block,
      const std::vector<framework::BlockDesc*>& grad_block =
          std::vector<framework::BlockDesc*>())
      : fwd_op_(fwd_op),
        no_grad_set_(no_grad_set),
        grad_to_var_(grad_to_var),
        current_block_(current_block),
        grad_block_(grad_block) {
    // TODO(jiabin): This should always execute by one thread...
    StaticCompositeContext::Instance().SetBlock(current_block);
  }

  virtual ~GradCompositeOpMakerBase() = default;

  virtual void operator()() = 0;

 protected:
  framework::VarDesc* SingleInputGrad(const std::string& name,
                                      bool drop_empty_grad = true) const {
    auto var_name = this->SingleForwardInputVarName(name);
    auto grad_var_name = framework::GradVarName(var_name);
    if (no_grad_set_.empty() || !no_grad_set_.count(grad_var_name)) {
      (*this->grad_to_var_)[grad_var_name] = var_name;
      VLOG(8) << "Valid gradients: " << grad_var_name;
    } else {
      // TODO(jiabin): Will this cause fill zeros error?
      grad_var_name = framework::kEmptyVarName;
    }
    return current_block_->Var(grad_var_name);
  }

  framework::VarDesc* SingleOutputGrad(const std::string& name) const {
    auto var_name = this->SingleForwardOutputVarName(name);
    auto grad_var_name = framework::GradVarName(var_name);
    if (no_grad_set_.empty() || !no_grad_set_.count(grad_var_name)) {
      (*this->grad_to_var_)[grad_var_name] = var_name;
      VLOG(8) << "Valid gradients: " << grad_var_name;
    } else {
      grad_var_name = framework::kEmptyVarName;
    }
    return current_block_->Var(grad_var_name);
  }

  std::vector<framework::VarDesc*> MultiInputGrad(
      const std::string& name, bool drop_empty_grad = true) const {
    std::vector<std::string> ret_val;
    std::vector<framework::VarDesc*> input_grads;
    auto var_names = this->MultiForwardInputVarName(name);
    ret_val.reserve(var_names.size());
    std::transform(var_names.begin(),
                   var_names.end(),
                   std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = framework::GradVarName(fwd_var_name);
                     if (no_grad_set_.empty() || !no_grad_set_.count(g_name)) {
                       (*this->grad_to_var_)[g_name] = fwd_var_name;
                       return g_name;
                     } else {
                       return framework::kEmptyVarName;
                     }
                   });
    if (!drop_empty_grad) {
      for (const auto& name : ret_val) {
        input_grads.emplace_back(current_block_->Var(name));
      }
      return input_grads;
    }
    PADDLE_ENFORCE_LE(
        var_names.size(),
        1UL,
        platform::errors::Unavailable(
            "BUG from operator developer:"
            " for input argument with a list of variables, "
            " drop_empty_grad is not allowed because it makes"
            " the correspondence bewteen a variable and its gradient"
            " ambiguous."));

    std::vector<std::string> dropped_ret_val;
    dropped_ret_val.reserve(ret_val.size());
    std::copy_if(
        ret_val.begin(),
        ret_val.end(),
        std::back_inserter(dropped_ret_val),
        [](const std::string& str) { return str != framework::kEmptyVarName; });
    for (const auto& name : dropped_ret_val) {
      // TODO(jiabin): Will this cause fill zeros error?
      input_grads.emplace_back(current_block_->Var(name));
    }
    return input_grads;
  }

  std::vector<framework::VarDesc*> MultiOutputGrad(
      const std::string& name) const {
    std::vector<std::string> ret_val;
    auto out_names = this->MultiForwardOutputVarName(name);
    ret_val.reserve(out_names.size());
    std::transform(out_names.begin(),
                   out_names.end(),
                   std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = framework::GradVarName(fwd_var_name);
                     (*this->grad_to_var_)[g_name] = fwd_var_name;
                     return g_name;
                   });
    std::vector<framework::VarDesc*> grad_out;
    for (const auto& name : ret_val) {
      // TODO(jiabin): Will this cause fill zeros error?
      grad_out.emplace_back(current_block_->Var(name));
    }
    return grad_out;
  }

  framework::VarDesc* SingleForwardInput(const std::string& name) const {
    return current_block_->FindVar(fwd_op_.Input(name).at(0));
  }

  framework::VarDesc* SingleForwardOutput(const std::string& name) const {
    return current_block_->FindVar(fwd_op_.Output(name).at(0));
  }

  std::vector<framework::VarDesc*> MultiForwardInput(
      const std::string& name) const {
    std::vector<framework::VarDesc*> result;
    for (const auto& n : fwd_op_.Input(name)) {
      result.emplace_back(current_block_->FindVar(n));
    }
    return result;
  }

  std::vector<framework::VarDesc*> MultiForwardOutput(
      const std::string& name) const {
    std::vector<framework::VarDesc*> result;
    for (const auto& n : fwd_op_.Output(name)) {
      result.emplace_back(current_block_->FindVar(n));
    }
    return result;
  }

  std::string SingleForwardInputVarName(const std::string& name) const {
    return fwd_op_.Input(name).at(0);
  }

  std::string SingleForwardOutputVarName(const std::string& name) const {
    return fwd_op_.Output(name).at(0);
  }

  std::vector<std::string> MultiForwardOutputVarName(
      const std::string& name) const {
    return fwd_op_.Output(name);
  }

  std::vector<std::string> MultiForwardInputVarName(
      const std::string& name) const {
    return fwd_op_.Input(name);
  }

  static std::vector<std::string> EmptyInput() { return {}; }

  static std::vector<std::string> EmptyOutput() { return {}; }

  static std::vector<std::string> EmptyInputGrad() { return {}; }

  static std::vector<std::string> EmptyOutputGrad() { return {}; }

  std::vector<std::string> InputNames() const {
    return this->fwd_op_.InputNames();
  }

  std::vector<std::string> OutputNames() const {
    return this->fwd_op_.OutputNames();
  }

  const std::unordered_map<std::string, framework::Attribute>& Attrs() const {
    return fwd_op_.GetAttrMap();
  }

  const std::unordered_map<std::string, framework::Attribute>& RuntimeAttrs()
      const {
    return fwd_op_.GetRuntimeAttrMap();
  }

  const framework::Attribute& GetAttr(const std::string& name) const {
    auto& map = fwd_op_.GetAttrMap();
    auto it = map.find(name);
    PADDLE_ENFORCE_NE(
        it,
        map.end(),
        platform::errors::NotFound("Cannot find attribute (%s).", name));
    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return PADDLE_GET_CONST(T, GetAttr(name));
  }

  std::string ForwardOpType() const { return this->fwd_op_.Type(); }
  const framework::BlockDesc* GetForwardOpBlock() const {
    return fwd_op_.Block();
  }

 protected:
  bool HasInput(const std::string& name) const {
    return (fwd_op_.Inputs().count(name) > 0);
  }

  bool HasOutput(const std::string& name) const {
    return (fwd_op_.Outputs().count(name) > 0);
  }

 private:
  const framework::OpDesc& fwd_op_;
  const std::unordered_set<std::string>& no_grad_set_;
  std::unordered_map<std::string, std::string>* grad_to_var_;
  framework::BlockDesc* current_block_;

 protected:
  std::vector<framework::BlockDesc*> grad_block_;
};

}  // namespace prims
}  // namespace paddle
