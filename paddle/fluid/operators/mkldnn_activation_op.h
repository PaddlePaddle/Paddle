/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(context.Input<framework::Tensor>("X") != nullptr,
                   "Cannot get input tensor X, variable name = %s",
                   context.op().Input("X"));
    PADDLE_ENFORCE(context.Output<framework::Tensor>("Out") != nullptr,
                   "Cannot find output tensor Out, variable name = %s",
                   context.op().Output("Out"));
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

template <typename Functor>
class MKLDNNActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    Functor functor;

    auto attrs = functor.GetAttrs();
    for (auto& attr : attrs) {
      *attr.second = context.Attr<float>(attr.first);
    }
    functor(context);
  }
};

namespace {  // NOLINT
framework::OpKernelType GetKernelType(const framework::ExecutionContext& ctx,
                                      const framework::OperatorWithKernel& oper,
                                      const std::string& name) {
  framework::LibraryType library{framework::LibraryType::kPlain};
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
  }
#endif
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<framework::Tensor>(name)->type()),
      ctx.GetPlace(), layout, library);
}
}  // anonymous namespace

class ActivationWithMKLDNNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "X");
  }
};

class ActivationWithMKLDNNOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return GetKernelType(ctx, *this, "Out");
  }
};

}  // namespace operators
}  // namespace paddle
