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
#include "paddle/fluid/imperative/dygraph_grad_maker.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/type_defs.h"

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
      GradToVarMapType* grad_to_var,
      const std::vector<BlockDesc*>& grad_block = std::vector<BlockDesc*>());

  virtual ~GradOpDescMakerBase();
  virtual std::vector<std::unique_ptr<OpDesc>> operator()() const = 0;

 protected:
  std::vector<std::string> InputGrad(const std::string& name,
                                     bool drop_empty_grad = true) const;

  std::vector<std::string> OutputGrad(const std::string& name) const;

  std::vector<std::string> Empty() const;

  std::vector<std::string> InputNames() const;

  std::vector<std::string> OutputNames() const;

  std::vector<std::string> Input(const std::string& name) const;

  std::vector<std::string> Output(const std::string& name) const;

  const AttributeMap& Attrs() const;

  const Attribute& GetAttr(const std::string& name) const;

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return boost::get<T>(GetAttr(name));
  }

  std::string ForwardOpType() const;

 protected:
  bool HasInput(const std::string& name) const;

 private:
  const OpDesc& fwd_op_;
  const std::unordered_set<std::string>& no_grad_set_;
  // std::unordered_map<std::string, std::string>* grad_to_var_;
  GradToVarMapType* grad_to_var_;

 protected:
  std::vector<BlockDesc*> grad_block_;
};

template <typename T>
class SingleGradOpMaker {
 public:
  std::vector<std::unique_ptr<T>> operator()() const {
    PADDLE_ENFORCE(false, "should not call this function");
    return {};
  }

 protected:
  virtual std::unique_ptr<T> Apply() const = 0;
};

template <>
class SingleGradOpMaker<OpDesc> : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<OpDesc>> operator()() const {
    std::vector<std::unique_ptr<OpDesc>> retv;
    retv.emplace_back(this->Apply());
    return retv;
  }

 protected:
  virtual std::unique_ptr<OpDesc> Apply() const = 0;
};

template <>
class SingleGradOpMaker<imperative::OpBase>
    : public imperative::GradOpBaseMakerBase {
 public:
  using GradOpBaseMakerBase::GradOpBaseMakerBase;

 public:
  std::vector<std::unique_ptr<imperative::OpBase>> operator()() const {
    std::vector<std::unique_ptr<imperative::OpBase>> retv;
    retv.emplace_back(this->Apply());

    return retv;
  }

 protected:
  virtual std::unique_ptr<imperative::OpBase> Apply() const = 0;
};

template <typename T, bool DropEmptyIG = true>
class DefaultGradOpMaker final : public SingleGradOpMaker<T> {
 public:
  using SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const final {
    auto* grad = new T();
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

    return std::unique_ptr<T>(grad);
  }
};

template <typename T>
class EmptyGradOpMaker {
 public:
  virtual std::vector<std::unique_ptr<T>> operator()()
      const final { /* NOLINT */
    return {};
  }
};

template <>
class EmptyGradOpMaker<OpDesc> final : public GradOpDescMakerBase {
 public:
  using GradOpDescMakerBase::GradOpDescMakerBase;
  std::vector<std::unique_ptr<OpDesc>> operator()() const final { return {}; }
};

template <>
class EmptyGradOpMaker<imperative::OpBase> final
    : public imperative::GradOpBaseMakerBase {
 public:
  using GradOpBaseMakerBase::GradOpBaseMakerBase;
  std::vector<std::unique_ptr<imperative::OpBase>> operator()() const final {
    return {};
  }
};

void* AnyFunc(void*);

OpDesc* NewOpDesc();

imperative::OpBase* NewOpBase();

template <typename T>
T* CreateOp();

template <>
inline OpDesc* CreateOp<OpDesc>() {
  return NewOpDesc();
}

template <>
inline imperative::OpBase* CreateOp<imperative::OpBase>() {
  return NewOpBase();
}

}  // namespace framework
}  // namespace paddle
