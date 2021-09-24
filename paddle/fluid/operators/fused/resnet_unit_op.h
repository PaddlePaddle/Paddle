/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <numeric>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/grad_op_desc_maker.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/cudnn_desc.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class ResNetUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ResNetUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ResNetUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T>
class ResNetUnitGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("resnet_unit_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("FilterX", this->Input("FilterX"));
    op->SetInput("ConvX", this->Input("ConvX"));
    op->SetInput("ScaleX", this->Input("ScaleX"));
    op->SetInput("BiasX", this->Input("BiasX"));
    op->SetInput("SavedMeanX", this->Output("SavedMeanX"));
    op->SetInput("SavedInvstdX", this->Output("SavedInvstdX"));
    op->SetInput("Z", this->Input("Z"));
    op->SetInput("FilterZ", this->Input("FilterZ"));
    op->SetInput("ConvZ", this->Input("ConvZ"));
    op->SetInput("ScaleZ", this->Input("ScaleZ"));
    op->SetInput("BiasZ", this->Input("BiasZ"));
    op->SetInput("SavedMeanZ", this->Output("SavedMeanZ"));
    op->SetInput("SavedInvstdZ", this->Output("SavedInvstdZ"));
    op->SetInput("Y", this->Output("Y"));
    op->SetInput("BitMask", this->Output("BitMask"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("FilterX"),
                  this->InputGrad("FilterX"));
    op->SetOutput(framework::GradVarName("ScaleX"), this->InputGrad("ScaleX"));
    op->SetOutput(framework::GradVarName("BiasX"), this->InputGrad("BiasX"));
    op->SetOutput(framework::GradVarName("ConvX"), this->OutputGrad("ConvX"));
    op->SetOutput(framework::GradVarName("Z"), this->InputGrad("Z"));
    op->SetOutput(framework::GradVarName("ScaleZ"), this->InputGrad("ScaleZ"));
    op->SetOutput(framework::GradVarName("BiasZ"), this->InputGrad("BiasZ"));
    op->SetOutput(framework::GradVarName("ConvZ"), this->OutputGrad("ConvZ"));
    op->SetOutput(framework::GradVarName("FilterZ"),
                  this->InputGrad("FilterZ"));
  }
};

class ResNetUnitOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

template <typename DeviceContext, typename T>
class ResNetUnitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class ResNetUnitGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override;
};

}  // namespace operators
}  // namespace paddle
