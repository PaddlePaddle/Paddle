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
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

// only can include the headers in paddle/pten/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/kernels/dot_grad_kernel.h"
#include "paddle/pten/kernels/dot_kernel.h"

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

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    // call new kernel
    pten::DotKernel<T, typename paddle::framework::ConvertToPtenContext<
                           DeviceContext>::TYPE>(
        static_cast<const typename paddle::framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *pt_x.get(), *pt_y.get(), pt_out.get());
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

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*tensor_x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*tensor_y);
    auto pt_dout = paddle::experimental::MakePtenDenseTensor(*tensor_dout);
    auto pt_dx = paddle::experimental::MakePtenDenseTensor(*tensor_dx);
    auto pt_dy = paddle::experimental::MakePtenDenseTensor(*tensor_dy);

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // call new kernel
    pten::DotGradKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *pt_x, *pt_y, *pt_dout, pt_dx.get(), pt_dy.get());
  }
};

}  // namespace operators
}  // namespace paddle
