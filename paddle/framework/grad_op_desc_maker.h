/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/op_desc.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {

class GradOpDescMakerBase {
 public:
  explicit GradOpDescMakerBase(const OpDescBind& fwd_op) : fwd_op_(fwd_op) {}

  virtual ~GradOpDescMakerBase() = default;
  virtual std::vector<std::unique_ptr<OpDescBind>> operator()() const = 0;

 protected:
  static std::vector<std::string> ToGradNames(
      const std::vector<std::string>& var_names) {
    std::vector<std::string> ret_val;
    ret_val.reserve(var_names.size());
    std::transform(var_names.begin(), var_names.end(),
                   std::back_inserter(ret_val), GradVarName);
    return ret_val;
  }

  std::vector<std::string> InputGrad(const std::string& name) const {
    return ToGradNames(fwd_op_.Input(name));
  }

  std::vector<std::string> OutputGrad(const std::string& name) const {
    return ToGradNames(fwd_op_.Output(name));
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

  std::string ForwardOpType() const { return this->fwd_op_.Type(); }

 private:
  const OpDescBind& fwd_op_;
};

class SingleGradOpDescMaker : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<OpDescBind>> operator()() const {
    std::vector<std::unique_ptr<OpDescBind>> retv;
    retv.emplace_back(this->Apply());
    return retv;
  }

 protected:
  virtual std::unique_ptr<OpDescBind> Apply() const = 0;
};

class DefaultGradOpDescMaker : public SingleGradOpDescMaker {
 public:
  using SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  virtual std::unique_ptr<OpDescBind> Apply() const {
    auto* grad = new OpDescBind();
    grad->SetType(this->GradOpType());

    for (auto& input_param : this->InputNames()) {
      grad->SetInput(input_param, this->Input(input_param));
      grad->SetOutput(GradVarName(input_param), this->InputGrad(input_param));
    }

    for (auto& output_param : this->OutputNames()) {
      grad->SetInput(output_param, this->Output(output_param));
      grad->SetInput(GradVarName(output_param), this->OutputGrad(output_param));
    }

    grad->SetAttrMap(this->Attrs());

    return std::unique_ptr<OpDescBind>(grad);
  }

  virtual std::string GradOpType() const {
    return this->ForwardOpType() + "_grad";
  }
};

}  // namespace framework
}  // namespace paddle
