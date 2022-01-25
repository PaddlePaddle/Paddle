// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/trace_grad_kernel.h"
#include "paddle/pten/kernels/trace_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class TraceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int offset = context.Attr<int>("offset");
    const int dim1 = context.Attr<int>("axis1");
    const int dim2 = context.Attr<int>("axis2");

    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::TraceKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, offset, dim1, dim2, out);
  }
};

template <typename DeviceContext, typename T>
class TraceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    auto x = context.Output<framework::Tensor>("Input");

    int offset = context.Attr<int>("offset");
    int dim1 = context.Attr<int>("axis1");
    int dim2 = context.Attr<int>("axis2");

    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::TraceGradKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *d_out, *x, offset, dim1, dim2, d_x);
  }
};

}  // namespace operators
}  // namespace paddle
