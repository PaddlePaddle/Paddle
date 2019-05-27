/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {

/*
  This functor class is responsible for creating the gradient ops for the given
  operator fwd_op. After it is called (through operator()), the pairs of
  (gradient variable, corresponding input variable of fwd_op) will be added to
  grad_to_var. If an input variable of fwd_op is contained in no_grad_set, its
  gradient varialbe will be ignored or kEmptyVarName depending on the template
  argument DropEmptyIG in the derived classes.
 */
class GradOpDescMakerBase {
 public:
  explicit GradOpDescMakerBase(
      const OpDesc& fwd_op, const std::unordered_set<std::string>& no_grad_set,
      std::unordered_map<std::string, std::string>* grad_to_var,
      const std::vector<BlockDesc*>& grad_block = std::vector<BlockDesc*>())
      : fwd_op_(fwd_op),
        no_grad_set_(no_grad_set),
        grad_to_var_(grad_to_var),
        grad_block_(grad_block) {}

  virtual ~GradOpDescMakerBase() = default;
  virtual std::vector<std::unique_ptr<OpDesc>> operator()() const = 0;

 protected:
  std::vector<std::string> InputGrad(const std::string& name,
                                     bool drop_empty_grad = true) const {
    std::vector<std::string> ret_val;
    auto var_names = this->Input(name);
    ret_val.reserve(var_names.size());
    std::transform(var_names.begin(), var_names.end(),
                   std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = GradVarName(fwd_var_name);
                     if (no_grad_set_.empty() || !no_grad_set_.count(g_name)) {
                       (*this->grad_to_var_)[g_name] = fwd_var_name;
                       return g_name;
                     } else {
                       return kEmptyVarName;
                     }
                   });
    if (!drop_empty_grad) {
      return ret_val;
    }
    PADDLE_ENFORCE_LE(var_names.size(), 1UL,
                      "BUG from operator developer:"
                      " for input argument with a list of variables, "
                      " drop_empty_grad is not allowed because it makes"
                      " the correspondence bewteen a variable and its gradient"
                      " ambiguous."
                      " Op type %s",
                      fwd_op_.Type());

    std::vector<std::string> dropped_ret_val;
    dropped_ret_val.reserve(ret_val.size());
    std::copy_if(ret_val.begin(), ret_val.end(),
                 std::back_inserter(dropped_ret_val),
                 [](const std::string& str) { return str != kEmptyVarName; });
    return dropped_ret_val;
  }

  std::vector<std::string> OutputGrad(const std::string& name) const {
    std::vector<std::string> ret_val;
    auto onames = this->Output(name);
    ret_val.reserve(onames.size());
    std::transform(onames.begin(), onames.end(), std::back_inserter(ret_val),
                   [this](const std::string& fwd_var_name) -> std::string {
                     auto g_name = GradVarName(fwd_var_name);
                     (*this->grad_to_var_)[g_name] = fwd_var_name;
                     return g_name;
                   });
    return ret_val;
  }

  std::vector<std::string> InputNames() const {
    return this->fwd_op_.InputNames();
  }

  std::vector<std::string> OutputNames() const {
    return this->fwd_op_.OutputNames();
  }

  std::vector<std::string> Input(const std::string& name) const {
    return fwd_op_.Input(name);
  }

  std::vector<std::string> Output(const std::string& name) const {
    return fwd_op_.Output(name);
  }

  const std::unordered_map<std::string, Attribute>& Attrs() const {
    return fwd_op_.GetAttrMap();
  }

  const Attribute& GetAttr(const std::string& name) const {
    auto& map = fwd_op_.GetAttrMap();
    auto it = map.find(name);
    PADDLE_ENFORCE(it != map.end(), "Cannot find attribute %s", name);
    return it->second;
  }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return boost::get<T>(GetAttr(name));
  }

  std::string ForwardOpType() const { return this->fwd_op_.Type(); }

 protected:
  const OpDesc& ForwardOp() const { return fwd_op_; }

 private:
  const OpDesc& fwd_op_;
  const std::unordered_set<std::string>& no_grad_set_;
  std::unordered_map<std::string, std::string>* grad_to_var_;

 protected:
  std::vector<BlockDesc*> grad_block_;
};

class SingleGradOpDescMaker : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<OpDesc>> operator()() const final {
    std::vector<std::unique_ptr<OpDesc>> retv;
    retv.emplace_back(this->Apply());
    return retv;
  }

 protected:
  virtual std::unique_ptr<OpDesc> Apply() const = 0;
};

template <bool DropEmptyIG = true>
class DefaultGradOpDescMaker final : public SingleGradOpDescMaker {
 public:
  using SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<OpDesc> Apply() const final {
    auto* grad = new OpDesc();
    grad->SetType(this->ForwardOpType() + "_grad");

    for (auto& input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(GradVarName(input_param),
                      this->InputGrad(input_param, DropEmptyIG));
    }

    for (auto& output_param : this->OutputNames()) {
      grad->SetInput(output_param, this->Output(output_param));
      grad->SetInput(GradVarName(output_param), this->OutputGrad(output_param));
    }

    grad->SetAttrMap(this->Attrs());

    return std::unique_ptr<OpDesc>(grad);
  }
};

class EmptyGradOpMaker final : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;
  std::vector<std::unique_ptr<OpDesc>> operator()() const final { return {}; }
};

}  // namespace framework
}  // namespace paddle
