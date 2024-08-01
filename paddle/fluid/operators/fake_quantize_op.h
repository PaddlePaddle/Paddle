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

#include "paddle/common/hostdevice.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class MovingAverageAbsMaxScaleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<phi::DenseTensor>("X");
    auto &dev_ctx = context.template device_context<DeviceContext>();

    if (context.HasOutput("Out")) {
      auto *out = context.Output<phi::DenseTensor>("Out");
      out->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*in, context.GetPlace(), dev_ctx, out);
    }

    bool is_test = context.Attr<bool>("is_test");
    // testing
    if (is_test) {
      return;
    }

    // training
    auto *in_accum = context.Input<phi::DenseTensor>("InAccum");
    auto *in_state = context.Input<phi::DenseTensor>("InState");
    phi::DenseTensor tmp_scale;
    tmp_scale.Resize(common::make_dim(1));
    T *cur_scale_data = dev_ctx.template Alloc<T>(&tmp_scale);

    phi::funcs::FindAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx, in->data<T>(), in->numel(), cur_scale_data);

    auto *out_state = context.Output<phi::DenseTensor>("OutState");
    auto *out_accum = context.Output<phi::DenseTensor>("OutAccum");
    auto *out_scale = context.Output<phi::DenseTensor>("OutScale");
    out_state->mutable_data<T>(context.GetPlace());
    out_accum->mutable_data<T>(context.GetPlace());
    out_scale->mutable_data<T>(context.GetPlace());
    float moving_rate = context.Attr<float>("moving_rate");

    phi::funcs::FindMovingAverageAbsMaxFunctor<DeviceContext, T>()(
        dev_ctx,
        *in_accum,
        *in_state,
        cur_scale_data,
        moving_rate,
        out_state,
        out_accum,
        out_scale);
  }
};

template <typename T, typename DeviceContext>
class StraightThroughEstimatorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    auto *d_x = context.Output<phi::DenseTensor>(x_grad_name);
    PADDLE_ENFORCE_NOT_NULL(
        d_x,
        common::errors::PreconditionNotMet("StraightThroughEstimatorGradKernel "
                                           "doesn't have the output named %s.",
                                           x_grad_name));

    // Initialize dx as same as d_out
    d_x->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(*d_out, context.GetPlace(), d_x);
  }
};

}  // namespace operators
}  // namespace paddle
