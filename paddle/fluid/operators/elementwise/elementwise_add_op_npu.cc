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

#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_npu.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
class ElementwiseAddNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    Tensor transformed_x, transformed_y;
    if (x->dims() != y->dims()) {
      int axis = ctx.Attr<int>("axis");
      NpuElementWiseOpBroadcast<T>(dev_ctx, x, y, axis, &transformed_x,
                                   &transformed_y);
    } else {
      transformed_x.ShareDataWith(*x);
      transformed_y.ShareDataWith(*y);
    }
    const auto& runner =
        NpuOpRunner("Add", {transformed_x, transformed_y}, {*out}, {});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class ElementwiseAddGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    if (dx && !dy && dx->dims() == dout->dims()) {
      dx->mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dx);
    } else if (dy && !dx && dy->dims() == dout->dims()) {
      dy->mutable_data<T>(ctx.GetPlace());
      framework::TensorCopy(*dout, ctx.GetPlace(), dev_ctx, dy);
    } else {
      auto* x = ctx.Input<framework::Tensor>("X");
      auto* y = ctx.Input<framework::Tensor>("Y");

      // skip out
      auto* out = dout;
      auto cpu_dev_ctx = platform::CPUDeviceContext(platform::CPUPlace());
      framework::Tensor cpu_out, cpu_dout, cpu_dx, cpu_dy;
      framework::Tensor *cpu_dx_ptr = nullptr, *cpu_dy_ptr = nullptr;
      if (dx) {
        dx->mutable_data<T>(ctx.GetPlace());
        cpu_dx.mutable_data<T>(dx->dims(), cpu_dev_ctx.GetPlace());
        cpu_dx_ptr = &cpu_dx;
      }
      if (dy) {
        dy->mutable_data<T>(ctx.GetPlace());
        cpu_dy.mutable_data<T>(dy->dims(), cpu_dev_ctx.GetPlace());
        cpu_dy_ptr = &cpu_dy;
      }
      cpu_out.mutable_data<T>(out->dims(), cpu_dev_ctx.GetPlace());
      cpu_dout.mutable_data<T>(dout->dims(), cpu_dev_ctx.GetPlace());
      framework::TensorCopy(*out, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_out);
      framework::TensorCopy(*dout, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_dout);
      dev_ctx.Wait();

      int axis = ctx.Attr<int>("axis");
      const framework::DDim x_dim = x->dims();
      const framework::DDim y_dim = y->dims();
      auto cpu_ctx =
          framework::ExecutionContext(ctx.GetOp(), ctx.scope(), cpu_dev_ctx,
                                      framework::RuntimeContext({}, {}));
      if (x_dim == y_dim) {
        ElemwiseGradComputeNoBroadcast<platform::CPUDeviceContext, T,
                                       IdentityGrad<T>, IdentityGrad<T>>(
            cpu_ctx, x_dim, y_dim, cpu_dout, cpu_dout, cpu_out, cpu_dout, axis,
            cpu_dx_ptr, cpu_dy_ptr, IdentityGrad<T>(), IdentityGrad<T>());
      } else {
        ElemwiseGradComputeWithBroadcast<platform::CPUDeviceContext, T,
                                         IdentityGrad<T>, IdentityGrad<T>>(
            cpu_ctx, x_dim, y_dim, cpu_dout, cpu_dout, cpu_out, cpu_dout, axis,
            cpu_dx_ptr, cpu_dy_ptr, IdentityGrad<T>(), IdentityGrad<T>());
      }
      if (dx) {
        framework::TensorCopy(cpu_dx, dev_ctx.GetPlace(), dev_ctx, dx);
      }
      if (dy) {
        framework::TensorCopy(cpu_dy, dev_ctx.GetPlace(), dev_ctx, dy);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(elementwise_add, ops::ElementwiseAddNPUKernel<float>,
                       ops::ElementwiseAddNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(elementwise_add_grad,
                       ops::ElementwiseAddGradNPUKernel<float>,
                       ops::ElementwiseAddGradNPUKernel<plat::float16>);
