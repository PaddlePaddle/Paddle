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
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/cpu_info.h"
#ifdef PADDLE_WITH_MKLDNN
#include "mkldnn.hpp"
#endif

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_MKLDNN
static bool AreDimsAndFormatCorrect(const framework::ExecutionContext& ctx,
                                    int simd_width,
                                    mkldnn::memory::format x_format) {
  using Tensor = framework::Tensor;
  using paddle::framework::vectorize2int;
  using mkldnn::memory;
  auto* x = ctx.Input<Tensor>("X");
  auto* y = ctx.Input<Tensor>("Y");
  auto x_dims = vectorize2int(x->dims());
  const bool are_dims_divisable = !(x_dims[1] % simd_width);
  const bool is_x_format_correct = x->format() == x_format;
  const bool is_y_format_correct = y->format() == memory::format::nc;
  return are_dims_divisable && is_x_format_correct && is_y_format_correct;
}
#endif

class ElementwiseMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using Tensor = framework::Tensor;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of elementwise op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of elementwise op should not be null.");

    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Y").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s [%s]",
        ctx->GetInputsVarType("Y").front(), ctx->Inputs("Y").front());

    if (ctx->GetInputsVarType("X").front() ==
        framework::proto::VarType::LOD_TENSOR) {
      auto x_dim = ctx->GetInputDim("X");
      auto y_dim = ctx->GetInputDim("Y");
      PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                        "Rank of first input must >= rank of second input.");
    } else if (ctx->GetInputsVarType("X").front() ==
               framework::proto::VarType::SELECTED_ROWS) {
      PADDLE_ENFORCE((ctx->GetInputDim("Y").size() == 1u) &&
                         (ctx->GetInputDim("Y")[0] == 1),
                     "For elementwise_op, if X is Sparse, "
                     "Y must be scalar.");
    } else {
      PADDLE_THROW("X's type[%s] is not supported by elementwise_op.",
                   ctx->GetInputsVarType("X").front());
    }

    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));

#ifdef PADDLE_WITH_MKLDNN
    using mkldnn::memory;
    if (platform::CanMKLDNNBeUsed(ctx)) {
      bool can_use_avx512_kernel =
          platform::MayIUse(platform::avx512f) &&
          AreDimsAndFormatCorrect(ctx, 16, memory::format::nChw16c);
      bool can_use_avx2_kernel =
          platform::MayIUse(platform::avx2) &&
          AreDimsAndFormatCorrect(ctx, 8, memory::format::nChw8c);
      if (can_use_avx512_kernel || can_use_avx2_kernel) {
        return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                       framework::DataLayout::kMKLDNN,
                                       framework::LibraryType::kMKLDNN);
      }
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename DeviceContext, typename T>
void default_elementwise_mul(const framework::ExecutionContext& ctx,
                             const framework::Tensor* x,
                             const framework::Tensor* y, framework::Tensor* z) {
  int axis = ctx.Attr<int>("axis");
  ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                        MulFunctor<T>(), z);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul_same_dims(const framework::ExecutionContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* y, framework::Tensor* z) {
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  blas.VMUL(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value ||
    !std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_mul_same_dims(const framework::ExecutionContext& ctx,
                          const framework::Tensor* x,
                          const framework::Tensor* y, framework::Tensor* z) {
  auto eigen_x = framework::EigenVector<T>::Flatten(*x);
  auto eigen_y = framework::EigenVector<T>::Flatten(*y);
  auto eigen_z = framework::EigenVector<T>::Flatten(*z);

  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  eigen_z.device(place) = eigen_x * eigen_y;
}

template <typename DeviceContext, typename T>
class ElementwiseMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x_var = ctx.InputVar("X");
    PADDLE_ENFORCE(x_var != nullptr,
                   "Cannot get input Variable X, variable name = %s",
                   ctx.op().Input("X"));
    auto* y = ctx.Input<framework::LoDTensor>("Y");

    framework::Tensor x, *z;
    if (x_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(y->dims().size() == 1 && y->dims()[0] == 1,
                     "For elementwise_op, if X is Sparse, Y must be scalar.");
      auto& x_sele = x_var->Get<framework::SelectedRows>();
      auto out_sele = ctx.Output<framework::SelectedRows>("Out");
      x = x_sele.value();
      out_sele->set_rows(x_sele.rows());
      out_sele->set_height(x_sele.height());
      out_sele->mutable_value()->Resize(x_sele.value().dims());
      out_sele->mutable_value()->mutable_data(ctx.GetPlace(), x.type());
      z = ctx.Output<framework::SelectedRows>("Out")->mutable_value();
    } else if (x_var->IsType<framework::LoDTensor>()) {
      x = x_var->Get<framework::LoDTensor>();
      z = ctx.Output<framework::LoDTensor>("Out");
    } else {
      PADDLE_THROW("X's type[%s] is not supported by elementwise_op.",
                   framework::ToTypeName(x_var->Type()));
    }

    z->mutable_data<T>(ctx.GetPlace());
    if (x.numel() == y->numel()) {
      elementwise_mul_same_dims<DeviceContext, T>(ctx, &x, y, z);
    } else {
      default_elementwise_mul<DeviceContext, T>(ctx, &x, y, z);
    }
  }
};

template <typename T>
struct MulGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * y; }
};

template <typename T>
struct MulGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout * x; }
};

template <typename DeviceContext, typename T>
class ElementwiseMulGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* out = dout;  // out is not necessary
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, MulGradDX<T>(), MulGradDY<T>());
  }
};

template <typename DeviceContext, typename T>
class ElementwiseMulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* dout = ctx.Input<Tensor>("DOut");
    auto* ddx = ctx.Input<Tensor>("DDX");
    auto* ddy = ctx.Input<Tensor>("DDY");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto* ddout = ctx.Output<Tensor>("DDOut");

    if (ddout) ddout->mutable_data<T>(ctx.GetPlace());

    // dx = dout * ddy
    // dy = dout * ddx
    Tensor ddx_safe, ddy_safe;
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, x, ddx, &ddx_safe);
    GetDoubleGradSafeTensor<DeviceContext, T>(ctx, y, ddy, &ddy_safe);
    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, MulGradDX<T>, MulGradDY<T>>(
        ctx, ddx_safe, ddy_safe, *dout, *dout, axis, dx, dy, MulGradDX<T>(),
        MulGradDY<T>());

    // ddout = ddx * y + x * ddy
    if (ddout) {
      if (ddx && ddy) {
        Tensor ddout_tmp;
        ddout_tmp.mutable_data<T>(ddout->dims(), ctx.GetPlace());

        default_elementwise_mul<DeviceContext, T>(ctx, ddx, y, ddout);
        default_elementwise_mul<DeviceContext, T>(ctx, x, ddy, &ddout_tmp);

        auto& place =
            *ctx.template device_context<DeviceContext>().eigen_device();
        auto ddout_t = framework::EigenVector<T>::Flatten(*ddout);
        auto ddout_tmp_t = framework::EigenVector<T>::Flatten(ddout_tmp);
        ddout_t.device(place) = ddout_t + ddout_tmp_t;
      } else {
        if (ddx) default_elementwise_mul<DeviceContext, T>(ctx, ddx, y, ddout);
        if (ddy) default_elementwise_mul<DeviceContext, T>(ctx, x, ddy, ddout);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
