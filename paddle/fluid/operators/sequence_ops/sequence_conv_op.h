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
#include <algorithm>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math/context_project.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class SequenceConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<phi::DenseTensor>("X");
    auto* out = context.Output<phi::DenseTensor>("Out");
    auto filter = *context.Input<phi::DenseTensor>("Filter");

    out->mutable_data<T>(context.GetPlace());

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(in->lod().empty(),
                      false,
                      phi::errors::InvalidArgument(
                          "Input(X) phi::DenseTensor of SequenceConvOp "
                          "does not contain LoD information."));
    PADDLE_ENFORCE_EQ(
        in->lod().size(),
        1UL,
        phi::errors::InvalidArgument(
            "Only support input sequence with lod level equal to 1 at "
            "present. But received: lod level %u.",
            in->lod().size()));

    const phi::DenseTensor* padding_data = nullptr;
    if (padding_trainable) {
      padding_data = context.Input<phi::DenseTensor>("PaddingData");
    }

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    auto sequence_width = static_cast<int64_t>(in->dims()[1]);

    phi::DDim col_shape = {in->dims()[0], context_length * sequence_width};
    phi::DenseTensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // Because if padding_trainable is false, padding data should be zeros.
    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
    set_zero(dev_ctx, &col, static_cast<T>(0));
    phi::math::ContextProjectFunctor<DeviceContext, T> seq_project_functor;

    seq_project_functor(dev_ctx,
                        *in,
                        padding_data,
                        padding_trainable,
                        context_start,
                        context_length,
                        context_stride,
                        up_pad,
                        down_pad,
                        &col);

    blas.MatMul(col, filter, out);
  }
};

template <typename T, typename DeviceContext>
class SequenceConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_g = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* out_g =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* filter_g =
        context.Output<phi::DenseTensor>(framework::GradVarName("Filter"));
    auto* padding_data_g =
        context.Output<phi::DenseTensor>(framework::GradVarName("PaddingData"));
    auto* in = context.Input<phi::DenseTensor>("X");
    auto* filter = context.Input<phi::DenseTensor>("Filter");

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(
        in->lod().size(),
        1UL,
        phi::errors::InvalidArgument(
            "Only support input sequence with lod level equal to 1 at "
            "present. But received: lod level %u.",
            in->lod().size()));
    auto lod_g_level_0 = in->lod()[0];

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    auto sequence_width = static_cast<int64_t>(in->dims()[1]);

    phi::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);
    // use col_shape in the im2col calculation
    phi::DDim col_shape = {in->dims()[0], sequence_width * context_length};
    phi::DenseTensor col;

    if (in_g || filter_g || (padding_trainable && padding_data_g)) {
      col.mutable_data<T>(col_shape, context.GetPlace());
      // Because if padding_trainable is false, padding data should be zeros.
      set_zero(dev_ctx, &col, static_cast<T>(0));
      blas.MatMul(*out_g, false, *filter, true, &col);
    }
    phi::math::ContextProjectFunctor<DeviceContext, T> seq_project_functor;
    phi::math::ContextProjectGradFunctor<DeviceContext, T>
        seq_project_grad_functor;

    if (in_g) {
      in_g->mutable_data<T>(context.GetPlace());
      in_g->set_lod(in->lod());
      set_zero(dev_ctx, in_g, static_cast<T>(0));

      seq_project_grad_functor(dev_ctx,
                               *in_g,
                               padding_trainable,
                               context_start,
                               context_length,
                               context_stride,
                               up_pad,
                               down_pad,
                               false,
                               true,
                               padding_data_g,
                               &col);
    }

    if (padding_trainable && padding_data_g) {
      padding_data_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, padding_data_g, static_cast<T>(0));

      phi::DenseTensor* input = const_cast<phi::DenseTensor*>(in);
      seq_project_grad_functor(dev_ctx,
                               *input,
                               padding_trainable,
                               context_start,
                               context_length,
                               context_stride,
                               up_pad,
                               down_pad,
                               true,
                               false,
                               padding_data_g,
                               &col);
    }

    if (filter_g) {
      filter_g->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_g, static_cast<T>(0));

      phi::DenseTensor filter_grad = *filter_g;
      phi::DenseTensor out_grad = *out_g;

      const phi::DenseTensor* padding_data = nullptr;
      if (padding_trainable) {
        padding_data = context.Input<phi::DenseTensor>("PaddingData");
      }

      seq_project_functor(dev_ctx,
                          *in,
                          padding_data,
                          padding_trainable,
                          context_start,
                          context_length,
                          context_stride,
                          up_pad,
                          down_pad,
                          &col);

      blas.MatMul(col, true, out_grad, false, &filter_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
