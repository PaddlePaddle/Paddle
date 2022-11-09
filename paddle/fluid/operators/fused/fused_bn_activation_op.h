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

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

class FusedBatchNormActOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class FusedBatchNormActGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class FusedBatchNormActOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T>
class FusedBatchNormActGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("Scale", this->Input("Scale"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput("SavedMean", this->Output("SavedMean"));
    op->SetInput("SavedVariance", this->Output("SavedVariance"));
    op->SetInput("ReserveSpace", this->Output("ReserveSpace"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
  }
};

class FusedBatchNormActOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

template <typename DeviceContext, typename T>
class FusedBatchNormActKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class FusedBatchNormActGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
