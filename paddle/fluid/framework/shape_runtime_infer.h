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

#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_type.h"

namespace paddle {
namespace framework {

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  bool HasInput(const std::string& name) const override;
  bool HasOutput(const std::string& name) const override;
  bool HasInputs(const std::string& name) const override;
  bool HasOutputs(const std::string& name) const override;

  const OperatorBase& OpBase() const { return op_; }

  const Scope& InferScope() const { return scope_; }
  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override;

  void ShareLayout(const std::string& in, const std::string& out, size_t i = 0,
                   size_t j = 0) const;

  bool IsRuntime() const override { return true; }

 protected:
  DDim GetDim(const std::string& name) const override;
  void SetDim(const std::string& name, const DDim& dim) override;

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW("Only compile time support this method");
  }
  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW("Only compile time support this method");
  }

  proto::VarType::Type GetVarType(const std::string& name) const override {
    auto* var = scope_.FindVar(name);
    return ToVarType(var->Type());
  }

  InferShapeVarPtr GetVarPtr(const std::string& name) override {
    return scope_.FindVar(name);
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

}  // namespace framework
}  // namespace paddle
