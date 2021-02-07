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

#include "paddle/fluid/operators/matmul_v2_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

template <typename T, typename FCT>
static void MatMulXPUFunction(const Tensor* x, const Tensor* y, Tensor* out,
                              bool trans_x, bool trans_y,
                              const paddle::framework::ExecutionContext& ctx) {
  const auto& x_dims = x->dims();
  const auto& y_dims = y->dims();
  auto& dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  auto mat_dim_a =
      math::CreateMatrixDescriptor(RowMatrixFromVector(x_dims), 0, trans_x);
  auto mat_dim_b =
      math::CreateMatrixDescriptor(ColumnMatrixFromVector(y_dims), 0, trans_y);

  if (x_dims.size() == 3 && y_dims.size() <= 2) {
    // if transpose_X is true, the transpose cost much time
    if (!trans_x) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    } else {
      mat_dim_b.batch_size_ = mat_dim_a.batch_size_;
      mat_dim_b.height_ = mat_dim_b.height_ / mat_dim_b.batch_size_;
    }
  }

  if (mat_dim_a.width_ == mat_dim_b.height_) {
    if (mat_dim_a.batch_size_ == 0 && mat_dim_b.batch_size_ == 1) {
      mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
    }
    if (mat_dim_a.batch_size_ == 1 && mat_dim_b.batch_size_ == 0) {
      mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
    }
  }

  PADDLE_ENFORCE_EQ(mat_dim_a.width_, mat_dim_b.height_,
                    platform::errors::InvalidArgument(
                        "Shape mistake in matmul_v2_op xdims = %s ydims = %s",
                        x_dims.to_str(), y_dims.to_str()));
  PADDLE_ENFORCE_EQ(mat_dim_a.batch_size_, mat_dim_b.batch_size_,
                    platform::errors::InvalidArgument(
                        "Shape mistake in matmul_v2_op xdims = %s ydims = %s",
                        x_dims.to_str(), y_dims.to_str()));

  float* data_c = out->data<T>();
  int m = mat_dim_a.height_;
  int n = mat_dim_b.width_;
  int k = mat_dim_a.width_;
  int batch_size = mat_dim_a.batch_size_;

  if (batch_size == 0) {
    int r = xpu::fc<float, float, float, FCT>(
        dev_ctx.x_context(), x->data<T>(), y->data<T>(), data_c, m, n, k,
        mat_dim_a.trans_, mat_dim_b.trans_, nullptr, nullptr, nullptr);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU fc_fusion kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  } else {
    // batch matmul
    int x_stride = mat_dim_a.stride_;
    int y_stride = mat_dim_b.stride_;
    int out_stride = m * n;
    for (int i = 0; i < batch_size; ++i) {
      const float* x_data = x->data<T>() + x_stride * i;
      const float* y_data = y->data<T>() + y_stride * i;
      float* out_data = data_c + out_stride * i;
      int r = xpu::fc<float, float, float, FCT>(
          dev_ctx.x_context(), x_data, y_data, out_data, m, n, k,
          mat_dim_a.trans_, mat_dim_b.trans_, nullptr, nullptr, nullptr);
      PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                        platform::errors::External(
                            "XPU fc_fusion kernel return wrong value[%d %s]", r,
                            XPUAPIErrorMsg[r]));
    }
  }
}

template <typename T>
class MatMulV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    out->mutable_data<T>(ctx.GetPlace());
    if (std::getenv("XPU_PADDLE_MAT_MUL_V2_FCINT32") != nullptr) {
      MatMulXPUFunction<T, int32_t>(x, y, out, trans_x, trans_y, ctx);
    } else {
      MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, ctx);
    }
  }
};

template <typename DeviceContext, typename T>
static framework::Tensor XPUFoldHeadAndLastDims(
    const DeviceContext& context, const framework::Tensor& input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }

  framework::Tensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> in_shape_host = {static_cast<int>(in_dims[0]),
                                    static_cast<int>(in_dims[1]),
                                    static_cast<int>(in_dims[2])};
  std::vector<int> axis_host = {1, 0, 2};

  int r = xpu::transpose(context.x_context(), input.data<T>(), output.data<T>(),
                         in_shape_host.data(), axis_host.data(), /*ndims=*/3);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU transpose kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});

  return output;
}

template <typename T>
class MatMulV2XPUGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& ctx,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(ctx.GetPlace());
    if (std::getenv("XPU_PADDLE_MAT_MUL_GRAD_V2_FCINT32") != nullptr) {
      MatMulXPUFunction<T, int32_t>(&a, &b, out, trans_a, trans_b, ctx);
    } else {
      MatMulXPUFunction<T, int16_t>(&a, &b, out, trans_a, trans_b, ctx);
    }
  }

  void CalcInputGrad(const framework::ExecutionContext& context,
                     const framework::Tensor& a, bool trans_a,
                     bool is_fold_init_dims_a, const framework::Tensor& b,
                     bool trans_b, bool is_fold_init_dims_b,
                     framework::Tensor* out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      auto& dev_ctx =
          context.template device_context<paddle::platform::XPUDeviceContext>();
      MatMul(
          context,
          is_fold_init_dims_a
              ? FoldInitDims(a)
              : XPUFoldHeadAndLastDims<paddle::platform::XPUDeviceContext, T>(
                    dev_ctx, a),
          trans_a,
          is_fold_init_dims_b
              ? FoldInitDims(b)
              : XPUFoldHeadAndLastDims<paddle::platform::XPUDeviceContext, T>(
                    dev_ctx, b),
          trans_b, out);
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    bool transpose_x = context.Attr<bool>("trans_x");
    bool transpose_y = context.Attr<bool>("trans_y");

    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);
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

    if (transpose_x && transpose_y) {
      CalcInputGrad(context, y, true, true, dout, true, false, dx);
      CalcInputGrad(context, dout, true, true, x, true, false, dy);
    } else if (transpose_x) {
      CalcInputGrad(context, y, false, false, dout, true, false, dx);
      CalcInputGrad(context, x, false, false, dout, false, true, dy);
    } else if (transpose_y) {
      CalcInputGrad(context, dout, false, false, y, false, true, dx);
      CalcInputGrad(context, dout, true, true, x, false, true, dy);
    } else {
      CalcInputGrad(context, dout, false, false, y, true, false, dx);
      CalcInputGrad(context, x, true, true, dout, false, true, dy);
    }

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

REGISTER_OP_XPU_KERNEL(matmul_v2, ops::MatMulV2XPUKernel<float>);
REGISTER_OP_XPU_KERNEL(matmul_v2_grad, ops::MatMulV2XPUGradKernel<float>);

#endif
