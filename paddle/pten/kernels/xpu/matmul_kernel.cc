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

#ifdef PADDLE_WITH_XPU

#include "paddle/pten/kernels/matmul_kernel.h"

#include "paddle/pten/backends/all_context.h"
#include "paddle/pten/core/kernel_registry.h"

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"

namespace pten {

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static DDim RowMatrixFromVector(const DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return paddle::framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static DDim ColumnMatrixFromVector(const DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return paddle::framework::make_ddim({y_dim[0], 1});
}

template <typename T, typename FCT>
static void MatMulXPUFunction(const XPUContext& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              DenseTensor* out,
                              bool trans_x,
                              bool trans_y) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();

  auto mat_dim_a = paddle::operators::math::CreateMatrixDescriptor(
      RowMatrixFromVector(x_dims), 0, trans_x);
  auto mat_dim_b = paddle::operators::math::CreateMatrixDescriptor(
      ColumnMatrixFromVector(y_dims), 0, trans_y);

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

  PADDLE_ENFORCE_EQ(mat_dim_a.width_,
                    mat_dim_b.height_,
                    paddle::platform::errors::InvalidArgument(
                        "Shape mistake in matmul_v2_op xdims = %s ydims = %s "
                        "x_trans = %d y_trans = %d",
                        x_dims.to_str(),
                        y_dims.to_str(),
                        mat_dim_a.trans_,
                        mat_dim_b.trans_));
  PADDLE_ENFORCE_EQ(mat_dim_a.batch_size_,
                    mat_dim_b.batch_size_,
                    paddle::platform::errors::InvalidArgument(
                        "Shape mistake in matmul_v2_op xdims = %s ydims = %s "
                        "x_trans = %d y_trans = %d",
                        x_dims.to_str(),
                        y_dims.to_str(),
                        mat_dim_a.trans_,
                        mat_dim_b.trans_));

  T* data_c = out->mutable_data<T>();
  int m = mat_dim_a.height_;
  int n = mat_dim_b.width_;
  int k = mat_dim_a.width_;
  int batch_size = mat_dim_a.batch_size_;
  int ldx = mat_dim_a.trans_ ? m : k;
  int ldy = mat_dim_b.trans_ ? k : n;
  int ldout = n;
  if (batch_size <= 1) {
    int r = 0;
    r = xpu_fc_wrapper<XPUType, FCT>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(x.data<T>()),
        reinterpret_cast<const XPUType*>(y.data<T>()),
        reinterpret_cast<XPUType*>(data_c),
        m,
        n,
        k,
        mat_dim_a.trans_,
        mat_dim_b.trans_,
        nullptr,
        nullptr,
        nullptr,
        ldx,
        ldy,
        ldout,
        1.0,
        0,
        nullptr,
        xpu::Activation_t::LINEAR);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        paddle::platform::errors::External(
            "XPU fc kernel return wrong value[%d %s] , m = %d, n = "
            "%d, "
            "k = %d, a_tr = %d, b_tr = %d",
            r,
            XPUAPIErrorMsg[r],
            m,
            n,
            k,
            mat_dim_a.trans_,
            mat_dim_b.trans_));
  } else {
    // batch matmul
    int r = xpu::fc_batched<XPUType, XPUType, XPUType, FCT>(
        dev_ctx.x_context(),                            // Context* ctx,
        batch_size,                                     // int batch_size,
        mat_dim_a.trans_,                               // bool x_trans,
        mat_dim_b.trans_,                               // bool w_trans,
        m,                                              // int m,
        n,                                              // int n,
        k,                                              // int k,
        1.0,                                            // float alpha,
        reinterpret_cast<const XPUType*>(x.data<T>()),  // const TX* x,
        mat_dim_a.stride_,                              // int stride_a,
        reinterpret_cast<const XPUType*>(y.data<T>()),  // const TW* w,
        mat_dim_b.stride_,                              // int stride_b,
        0.0,                                            // float beta,
        reinterpret_cast<XPUType*>(data_c),             // TY* y,
        m * n,                                          // int stride_c,
        nullptr,   // const float* x_maxptr,
        nullptr);  // const float* w_maxptr

    PADDLE_ENFORCE_EQ(r,
                      XPU_SUCCESS,
                      paddle::platform::errors::External(
                          "XPU fc_batched kernel return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
}

template <typename T, typename DevCtx>
void Matmul(const DevCtx& dev_ctx,
            const DenseTensor& x,
            const DenseTensor& y,
            bool transpose_x,
            bool transpose_y,
            DenseTensor* out) {
  if (std::is_same<paddle::platform::float16, T>::value) {
    MatMulXPUFunction<T, int16_t>(dev_ctx, x, y, out, transpose_x, transpose_y);
  } else {
    if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
      MatMulXPUFunction<T, int32_t>(
          dev_ctx, x, y, out, transpose_x, transpose_y);
    } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
      MatMulXPUFunction<T, float>(dev_ctx, x, y, out, transpose_x, transpose_y);
    } else {
      MatMulXPUFunction<T, int16_t>(
          dev_ctx, x, y, out, transpose_x, transpose_y);
    }
  }
}

}  // namespace pten

PT_REGISTER_CTX_KERNEL(
    matmul, XPU, ALL_LAYOUT, pten::Matmul, float, paddle::platform::float16) {}

#endif
