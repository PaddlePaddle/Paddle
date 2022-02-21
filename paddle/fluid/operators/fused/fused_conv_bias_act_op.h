/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedConvBiasActOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class FusedConvBiasActGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class FusedConvBiasActOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T>
class FusedConvBiasActGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput("Output", this->Output("Output"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));

    op->SetAttrMap(this->Attrs());
  }
};

class FusedConvBiasActOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

template <typename DeviceContext, typename T>
class FusedConvBiasActKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class FusedConvBiasActGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
