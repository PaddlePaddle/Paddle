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

#include <vector>
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

namespace paddle {
namespace operators {

template <typename T>
struct DivFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a / b; }
};

template <typename DeviceContext, typename T>
class ElementwiseDivKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::LoDTensor>("X");
    auto* y = ctx.Input<framework::LoDTensor>("Y");
    auto* z = ctx.Output<framework::LoDTensor>("Out");

    z->mutable_data<T>(ctx.GetPlace());
    int axis = ctx.Attr<int>("axis");
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          DivFunctor<T>(), z);
  }
};

template <typename T>
struct DivGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout / y; }
};

template <typename T>
struct DivGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return -dout * out / y;
  }
};

template <typename T>
struct DivDoubleDYBase {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return y * out * dout;
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");

    auto* x = dout;  // Fake x, not used

    ElemwiseGradCompute<DeviceContext, T, DivGradDX<T>, DivGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, DivGradDX<T>(), DivGradDY<T>());
  }
};

class ElementwiseDivOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput("DOut")) {
      ctx->ShareDim("DX", "DOut");
      ctx->ShareLoD("DX", "DOut");
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", y_grad_name);
      ctx->ShareLoD("Y", y_grad_name);
    }
    if (ctx->HasOutput("DDOut")) {
      ctx->ShareDim("DX", "DDOut");
      ctx->ShareLoD("DX", "DDOut");
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("DDX")->type();

#ifdef PADDLE_WITH_MKLDNN
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Input<Tensor>("Out");
    auto* ddX = ctx.Input<Tensor>("DDX");
    auto* ddY = ctx.Input<Tensor>("DDY");
    auto* dX = ctx.Input<Tensor>("DX");

    auto* dY = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* dOut = ctx.Output<Tensor>("DOut");
    auto* ddOut = ctx.Output<Tensor>("DDOut");

    if (dOut) dOut->mutable_data<T>(Out->dims(), ctx.GetPlace());
    if (ddOut) ddOut->mutable_data<T>(Out->dims(), ctx.GetPlace());

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    int axis = ctx.Attr<int>("axis");

    Tensor ddX_tmp, ddY_tmp, dX_tmp;
    if (ddX) {
      ddX_tmp.mutable_data<T>(ddX->dims(), ctx.GetPlace());
      ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
          ctx, ddX, Y, axis, DivFunctor<T>(), &ddX_tmp);
    }
    if (ddY) {
      ddY_tmp.mutable_data<T>(ddY->dims(), ctx.GetPlace());
      ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
          ctx, ddY, Y, 0, DivFunctor<T>(), &ddY_tmp);
    }

    dX_tmp.mutable_data<T>(Out->dims(), ctx.GetPlace());
    ElementwiseComputeEx<DivFunctor<T>, DeviceContext, T>(
        ctx, dX, Y, axis, DivFunctor<T>(), &dX_tmp);

    if (dOut && ddY) {
      Tensor dOut_tmp;
      dOut_tmp.mutable_data<T>(Out->dims(), ctx.GetPlace());
      default_elementwise_mul<DeviceContext, T>(ctx, dX, ddY, &dOut_tmp);
      auto dout = framework::EigenVector<T>::Flatten(*dOut);
      auto dout_tmp = framework::EigenVector<T>::Flatten(dOut_tmp);
      dout.device(place) = static_cast<T>(-1) * dout_tmp;
    }

    Tensor ddX_safe, ddY_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, Out, ddX, &ddX_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, Y, ddY, &ddY_safe);

    if (dY) {
      dY->mutable_data<T>(Y->dims(), ctx.GetPlace());
      auto dy = framework::EigenVector<T>::Flatten(*dY);
      int pre, n, post;
      std::vector<int> dims = {0, 2};
      bool keep_dim = false;

      get_mid_dims(Out->dims(), dY->dims(), axis, &pre, &n, &post);

      if (ddX && ddY) {
        Tensor dY_tmp1, dY_tmp2, dY_tmp1_sum, dY_tmp2_sum;
        dY_tmp1.mutable_data<T>(Out->dims(), ctx.GetPlace());
        dY_tmp2.mutable_data<T>(Out->dims(), ctx.GetPlace());
        dY_tmp1_sum.mutable_data<T>(framework::make_ddim({n}), ctx.GetPlace());
        dY_tmp2_sum.mutable_data<T>(framework::make_ddim({n}), ctx.GetPlace());
        ElemwiseGradCompute<DeviceContext, T, DivDoubleDYBase<T>, MulGradDY<T>>(
            ctx, ddX_safe, ddY_safe, *Out, dX_tmp, axis, &dY_tmp1, &dY_tmp2,
            DivDoubleDYBase<T>(), MulGradDY<T>());

        dY_tmp1.Resize(paddle::framework::make_ddim({pre, n, post}));

        ReduceFunctor<DeviceContext, T, 3, 2, SumFunctor>(
            ctx.template device_context<DeviceContext>(), dY_tmp1, &dY_tmp1_sum,
            dims, keep_dim);

        dY_tmp2.Resize(paddle::framework::make_ddim({pre, n, post}));

        ReduceFunctor<DeviceContext, T, 3, 2, SumFunctor>(
            ctx.template device_context<DeviceContext>(), dY_tmp2, &dY_tmp2_sum,
            dims, keep_dim);

        auto dy_tmp1 = framework::EigenVector<T>::Flatten(dY_tmp1_sum);
        auto dy_tmp2 = framework::EigenVector<T>::Flatten(dY_tmp2_sum);
        dy.device(place) = dy_tmp1 + static_cast<T>(-1) * dy_tmp2;
      } else {
        if (ddX) {
          Tensor dY_tmp1, dY_tmp1_sum;
          dY_tmp1.mutable_data<T>(Out->dims(), ctx.GetPlace());
          default_elementwise_mul<DeviceContext, T>(ctx, &ddX_tmp, dX,
                                                    &dY_tmp1);
          dY_tmp1.Resize(paddle::framework::make_ddim({pre, n, post}));
          dY_tmp1_sum.mutable_data<T>(framework::make_ddim({n}),
                                      ctx.GetPlace());

          ReduceFunctor<DeviceContext, T, 3, 2, SumFunctor>(
              ctx.template device_context<DeviceContext>(), dY_tmp1,
              &dY_tmp1_sum, dims, keep_dim);
          auto dy_tmp1 = framework::EigenVector<T>::Flatten(dY_tmp1_sum);
          dy.device(place) = static_cast<T>(-1) * dy_tmp1;
        }
        if (ddY) {
          Tensor dY_tmp1, tmp;
          default_elementwise_mul<DeviceContext, T>(ctx, &dX_tmp, ddY, &tmp);
          default_elementwise_mul<DeviceContext, T>(ctx, Out, &tmp, &dY_tmp1);

          dY_tmp1.Resize(paddle::framework::make_ddim({pre, n, post}));

          ReduceFunctor<DeviceContext, T, 3, 2, SumFunctor>(
              ctx.template device_context<DeviceContext>(), dY_tmp1, dY, dims,
              keep_dim);
        }
      }
    }
    if (ddOut) {
      if (ddX && ddY) {
        Tensor ddOut_tmp;
        ddOut_tmp.mutable_data<T>(Out->dims(), ctx.GetPlace());
        default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_tmp,
                                                  &ddOut_tmp);
        auto ddout_tmp2 = framework::EigenVector<T>::Flatten(ddX_tmp);
        auto ddout_tmp = framework::EigenVector<T>::Flatten(ddOut_tmp);
        auto ddout = framework::EigenVector<T>::Flatten(*ddOut);
        ddout.device(place) = static_cast<T>(-1) * ddout_tmp + ddout_tmp2;
      } else {
        if (ddX) {
          framework::TensorCopy(ddX_tmp, ctx.GetPlace(), ddOut);
        }
        if (ddY) {
          Tensor ddOut_tmp;
          ddOut_tmp.mutable_data<T>(Out->dims(), ctx.GetPlace());
          default_elementwise_mul<DeviceContext, T>(ctx, Out, &ddY_tmp,
                                                    &ddOut_tmp);
          auto ddout_tmp = framework::EigenVector<T>::Flatten(ddOut_tmp);
          auto ddout = framework::EigenVector<T>::Flatten(*ddOut);
          ddout.device(place) = static_cast<T>(-1) * ddout_tmp;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
