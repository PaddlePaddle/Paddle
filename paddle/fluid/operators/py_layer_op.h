// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/python_headers.h"
#include "paddle/fluid/operators/py_layer_context/py_context.h"

namespace paddle {
namespace operators {
void test();
// class PyLayerOp;
using CtxPtr = std::shared_ptr<imperative::PyLayerContext>;

class PyLayerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    // todo:check
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      return framework::OpKernelType(tensor.type(), tensor.place(),
                                     tensor.layout());
    } else {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), tensor.layout());
    }
  }

 public:
  CtxPtr& GetMutablePyLayerContext() { return py_context; }
  const CtxPtr& GetMutablePyLayerContext() const { return py_context; }

 private:
  CtxPtr py_context;
};

template <typename T>
class PyLayerGradOpMaker {};
template <>
class PyLayerGradOpMaker<paddle::framework::OpDesc>
    : public framework::SingleGradOpMaker<paddle::framework::OpDesc> {
 public:
  using framework::SingleGradOpMaker<
      paddle::framework::OpDesc>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::framework::OpDesc> grad_op) const override {
    grad_op->SetType("py_layer");
  }
};

template <>
class PyLayerGradOpMaker<paddle::imperative::OpBase>
    : public framework::SingleGradOpMaker<paddle::imperative::OpBase> {
 public:
  using framework::SingleGradOpMaker<
      paddle::imperative::OpBase>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<paddle::imperative::OpBase> grad_op) const override {
    grad_op->SetType("py_layer");
    auto inner_op = grad_op->MutableInnerOp();
    auto py_layer_op = dynamic_cast<PyLayerOp*>(inner_op);

    if (py_layer_op) {
      py_layer_op->GetMutablePyLayerContext() = py_context;

    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "PyLayerGradOpMaker can only be matched with PyLayer."));
    }

    // All forward inputs
    auto fwd_ins = this->Input("X");
    // All forward outputs
    auto fwd_outs = this->Output("Out");

    auto fwd_out_grads = this->OutputGrad("Out");
    using return_type = decltype(fwd_out_grads);
    return_type bwd_ins;
    bwd_ins.reserve(fwd_ins.size() + fwd_outs.size());
    for (auto var : fwd_ins) {
      bwd_ins.emplace_back(var);
    }
    for (auto var : fwd_outs) {
      bwd_ins.emplace_back(var);
    }

    bwd_ins.reserve(bwd_ins.size() + fwd_out_grads.size());
    bwd_ins.insert(bwd_ins.end(), fwd_out_grads.begin(), fwd_out_grads.end());

    auto bwd_outs = this->InputGrad("X", false);

    grad_op->SetInput("X", bwd_ins);
    grad_op->SetOutput("Out", bwd_outs);
  }

 public:
  CtxPtr& GetMutablePyLayerContext() { return py_context; }

 private:
  CtxPtr py_context;
};

}  // namespace operators
}  // namespace paddle
