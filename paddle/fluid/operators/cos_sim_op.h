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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/cos_sim_functor.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class CosSimKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* in_x = context.Input<framework::LoDTensor>("X");
    auto* in_y = context.Input<phi::DenseTensor>("Y");
    auto* out_z = context.Output<framework::LoDTensor>("Out");
    auto* out_x_norm = context.Output<phi::DenseTensor>("XNorm");
    auto* out_y_norm = context.Output<phi::DenseTensor>("YNorm");

    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];
    out_z->Resize({rows_x, 1});
    out_x_norm->Resize({rows_x, 1});
    out_y_norm->Resize({rows_y, 1});
    out_z->mutable_data<T>(context.GetPlace());
    out_x_norm->mutable_data<T>(context.GetPlace());
    out_y_norm->mutable_data<T>(context.GetPlace());
    out_z->set_lod(in_x->lod());

    int cols = phi::product(in_x->dims()) / rows_x;

    if (rows_x == rows_y) {
      math::CosSimFunctor<T, true> functor(in_x->data<T>(),
                                           in_y->data<T>(),
                                           out_x_norm->data<T>(),
                                           out_y_norm->data<T>(),
                                           out_z->data<T>(),
                                           cols);
      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(context.device_context()), rows_x);
      for_range(functor);
    } else {
      math::CosSimFunctor<T, false> functor(in_x->data<T>(),
                                            in_y->data<T>(),
                                            out_x_norm->data<T>(),
                                            out_y_norm->data<T>(),
                                            out_z->data<T>(),
                                            cols);
      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(context.device_context()), rows_x);
      for_range(functor);
    }
  }
};

template <typename DeviceContext, typename T>
class CosSimGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get Tensor
    auto* in_x = context.Input<phi::DenseTensor>("X");
    auto* in_y = context.Input<phi::DenseTensor>("Y");
    auto* in_z = context.Input<phi::DenseTensor>("Out");
    auto* in_x_norm = context.Input<phi::DenseTensor>("XNorm");
    auto* in_y_norm = context.Input<phi::DenseTensor>("YNorm");
    auto* out_grad_x =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* out_grad_y =
        context.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* in_grad_z =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    // compute gradident
    int rows_x = in_x->dims()[0];
    int rows_y = in_y->dims()[0];
    int cols = phi::product(in_x->dims()) / rows_x;

    if (rows_x == rows_y) {
      if (out_grad_x) {
        out_grad_x->Resize(in_x->dims());
        math::CosSimGradFunctor<T> functor(
            in_x_norm->data<T>(),
            in_y_norm->data<T>(),
            in_x->data<T>(),
            in_y->data<T>(),
            in_z->data<T>(),
            in_grad_z->data<T>(),
            out_grad_x->mutable_data<T>(context.GetPlace()),
            cols);
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(context.device_context()),
            rows_x);
        for_range(functor);
      }
      if (out_grad_y) {
        out_grad_y->Resize(in_y->dims());
        math::CosSimGradFunctor<T> functor(
            in_y_norm->data<T>(),
            in_x_norm->data<T>(),
            in_y->data<T>(),
            in_x->data<T>(),
            in_z->data<T>(),
            in_grad_z->data<T>(),
            out_grad_y->mutable_data<T>(context.GetPlace()),
            cols);
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(context.device_context()),
            rows_x);
        for_range(functor);
      }
    } else {
      if (out_grad_x) {
        out_grad_x->Resize(in_x->dims());
        math::CosSimDxFunctor<T> functor(
            in_x_norm->data<T>(),
            in_y_norm->data<T>(),
            in_x->data<T>(),
            in_y->data<T>(),
            in_z->data<T>(),
            in_grad_z->data<T>(),
            out_grad_x->mutable_data<T>(context.GetPlace()),
            cols);
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(context.device_context()),
            rows_x);
        for_range(functor);
      }
      if (out_grad_y) {
        out_grad_y->Resize(in_y->dims());
        out_grad_y->mutable_data<T>(context.GetPlace());
        phi::funcs::SetConstant<DeviceContext, T> set_zero;
        auto& dev_ctx = context.template device_context<DeviceContext>();
        set_zero(dev_ctx, out_grad_y, static_cast<T>(0));

        math::CosSimDyFunctor<DeviceContext, T> functor;
        functor(dev_ctx,
                in_x_norm->data<T>(),
                in_y_norm->data<T>(),
                in_x->data<T>(),
                in_y->data<T>(),
                in_z->data<T>(),
                in_grad_z->data<T>(),
                static_cast<size_t>(rows_x),
                static_cast<size_t>(cols),
                out_grad_y->data<T>());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
