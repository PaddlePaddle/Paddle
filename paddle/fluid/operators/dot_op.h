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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

// only can include the headers in paddle/phi/api dirs
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/kernels/dot_grad_kernel.h"
#include "paddle/phi/kernels/dot_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// See Note [ Why still keep the original kernel implementation? ]
template <typename DeviceContext, typename T>
class DotKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    auto& dev_ctx = ctx.device_context<DeviceContext>();
    out->mutable_data<T>(x->place());

    // call new kernel
    phi::DotKernel<T, typename paddle::framework::ConvertToPhiContext<
                          DeviceContext>::TYPE>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *x, *y, out);
  }
};

template <typename DeviceContext, typename T>
class DotGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* tensor_x = ctx.Input<Tensor>("X");
    auto* tensor_y = ctx.Input<Tensor>("Y");
    auto* tensor_dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* tensor_dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* tensor_dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    if (tensor_dx) tensor_dx->mutable_data<T>(ctx.GetPlace());
    if (tensor_dy) tensor_dy->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // call new kernel
    phi::DotGradKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *tensor_x, *tensor_y, *tensor_dout, tensor_dx, tensor_dy);
  }
};

}  // namespace operators
}  // namespace paddle
