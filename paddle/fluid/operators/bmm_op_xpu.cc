//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include <string>
#include <vector>
#include "paddle/fluid/operators/matmul_v2_op.h"

#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename T, typename FCT>
static void MatMulXPUFunction(const Tensor* x, const Tensor* y, Tensor* out,
                              bool trans_x, bool trans_y,
                              const paddle::framework::ExecutionContext& ctx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x->dims();
  const auto& y_dims = y->dims();
  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(
      RowMatrixFromVector(x_dims), 0, trans_x);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(
      ColumnMatrixFromVector(y_dims), 0, trans_y);

  T* data_c = out->data<T>();
  int m = mat_dim_a.height_;
  int n = mat_dim_b.width_;
  int k = mat_dim_a.width_;
  int batch_size = mat_dim_a.batch_size_;
  // batch matmul
  int r = xpu::fc_batched<XPUType, XPUType, XPUType, FCT>(
      dev_ctx.x_context(),                             // Context* ctx,
      batch_size,                                      // int batch_size,
      mat_dim_a.trans_,                                // bool x_trans,
      mat_dim_b.trans_,                                // bool w_trans,
      m,                                               // int m,
      n,                                               // int n,
      k,                                               // int k,
      1.0,                                             // float alpha,
      reinterpret_cast<const XPUType*>(x->data<T>()),  // const TX* x,
      mat_dim_a.stride_,                               // int stride_a,
      reinterpret_cast<const XPUType*>(y->data<T>()),  // const TW* w,
      mat_dim_b.stride_,                               // int stride_b,
      0.0,                                             // float beta,
      reinterpret_cast<XPUType*>(data_c),              // TY* y,
      m * n,                                           // int stride_c,
      nullptr,                                         // const float* x_maxptr,
      nullptr);                                        // const float* w_maxptr

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_batched");
}

template <typename T>
class BmmXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    if (x->numel() == 0 || y->numel() == 0) {
      return;
    }
    bool trans_x = false;
    bool trans_y = false;

    auto x_dims = x->dims();
    auto y_dims = y->dims();

    PADDLE_ENFORCE_EQ(x_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "Input(X) of BmmOp must be 3-dimensional in BmmOp, "
                          "but received X's shape: [%s].",
                          x_dims));
    PADDLE_ENFORCE_EQ(y_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "Input(Y) of BmmOp must be 3-dimensional in BmmOp, "
                          "but received Y's shape: [%s].",
                          y_dims));
    PADDLE_ENFORCE_EQ(
        x_dims[0], y_dims[0],
        platform::errors::InvalidArgument(
            "Input(X) and Input(Y) must have the same batch size in BmmOp, "
            "but received X's batch size: [%s],"
            "Y's batch size [%s]",
            x_dims[0], y_dims[0]));
    PADDLE_ENFORCE_EQ(
        x_dims[2], y_dims[1],
        platform::errors::InvalidArgument(
            "Input(X)'s width must be equal with Input(Y)'s height in BmmOp,"
            "but receive X's width: [%s],"
            "Y's height: [%s].",
            x_dims[2], y_dims[1]));

    if (std::is_same<paddle::platform::float16, T>::value) {
      MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, ctx);
    } else {
      if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
        MatMulXPUFunction<T, int32_t>(x, y, out, trans_x, trans_y, ctx);
      } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
        MatMulXPUFunction<T, float>(x, y, out, trans_x, trans_y, ctx);
      } else {
        MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, ctx);
      }
    }
  }
};

template <typename T>
class BmmXPUGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& ctx,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(ctx.GetPlace());
    if (std::is_same<paddle::platform::float16, T>::value) {
      MatMulXPUFunction<T, int16_t>(&a, &b, out, trans_a, trans_b, ctx);
    } else {
      if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
        MatMulXPUFunction<T, int32_t>(&a, &b, out, trans_a, trans_b, ctx);
      } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
        MatMulXPUFunction<T, float>(&a, &b, out, trans_a, trans_b, ctx);
      } else {
        MatMulXPUFunction<T, int16_t>(&a, &b, out, trans_a, trans_b, ctx);
      }
    }
  }

  void CalcInputGrad(const framework::ExecutionContext& context,
                     const framework::Tensor& a, bool trans_a,
                     const framework::Tensor& b, bool trans_b,
                     framework::Tensor* out) const {
    if (out == nullptr) return;
    MatMul(context, a, trans_a, b, trans_b, out);
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, false, false);

    framework::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    framework::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    CalcInputGrad(context, dout, false, y, true, dx);
    CalcInputGrad(context, x, true, dout, false, dy);

    // CalcInputGrad(context, dout, false, false, y, true, false, dx);
    // CalcInputGrad(context, x, true, true, dout, false, true, dy);

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
      }
    }

    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(bmm, ops::BmmXPUKernel<float>,
                       ops::BmmXPUKernel<plat::float16>);
REGISTER_OP_XPU_KERNEL(bmm_grad, ops::BmmXPUGradKernel<float>,
                       ops::BmmXPUGradKernel<plat::float16>);

#endif
