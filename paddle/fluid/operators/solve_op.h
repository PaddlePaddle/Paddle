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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/dot_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "paddle/fluid/operators/matmul_v2_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

constexpr int kMULMKLDNNINT8 = 1;

template <typename DeviceContext, typename T>
class SolveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("X");
    Tensor tmp_x(input->type());
    tmp_x.Resize(input->dims());
    tmp_x.mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, &tmp_x);

    const Tensor x_matrix = tmp_x;
    const Tensor* y = context.Input<Tensor>("Y");
    const Tensor y_matrix = *y;

    Tensor* z = context.Output<Tensor>("Out");
    z->mutable_data<T>(context.GetPlace());

    // special case: input likes A(2,3,3) B(2,3)
    auto x_dim = input->dims();
    auto y_dim = y->dims();
    auto x_dim_size = x_dim.size();
    auto y_dim_size = y_dim.size();
    bool is_special = false;
    if (x_dim_size - y_dim_size == 1) {
      is_special = true;
      for (int i = 0; i < y_dim_size; i++) {
        if (x_dim[i] != y_dim[i]) {
          is_special = false;
        }
      }
    }

    Tensor tmp_y;
    if (is_special) {  // reshape operation
      std::vector<int64_t> y_dims_vec = paddle::framework::vectorize(y_dim);
      y_dims_vec.push_back(1);
      auto y_new_dims = framework::make_ddim(y_dims_vec);
      tmp_y.ShareDataWith(*y).Resize(y_new_dims);
      MatMulFunction<DeviceContext, T>(&x_matrix, &tmp_y, z, false, false,
                                       context);
      z->Resize(y_dim);
    } else {  // normal case
      PADDLE_ENFORCE_EQ(
          x_dim[x_dim_size - 1], y_dim[y_dim_size - 2],
          platform::errors::InvalidArgument(
              "Matrix X1 with dimension greater than 2 and any matrix Y1,"
              "the matrix X1's width must be equal with matrix Y1's "
              "height. But received X's shape = [%s], X1's shape = [%s], X1's "
              "width = %s; Y's shape = [%s], Y1's shape = [%s], Y1's height = "
              "%s.",
              x_dim, x_dim, x_dim[x_dim_size - 1], y_dim, y_dim,
              y_dim[y_dim_size - 2]));
      // defined in matmul_v2_op.h
      MatMulFunction<DeviceContext, T>(&x_matrix, &y_matrix, z, false, false,
                                       context);
    }
  }
};

template <typename DeviceContext, typename T>
class SolveGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& context,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = math::CreateMatrixDescriptor(b.dims(), 0, trans_b);
    if (a.dims().size() == 3 && b.dims().size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!trans_a) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
    blas.MatMul(a, mat_dim_a, b, mat_dim_b, static_cast<T>(1), out,
                static_cast<T>(0));
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
      auto& ctx = context.template device_context<DeviceContext>();
      MatMul(context, is_fold_init_dims_a
                          ? FoldInitDims(a)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, a),
             trans_a, is_fold_init_dims_b
                          ? FoldInitDims(b)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, b),
             trans_b, out);
    }
  }
  void InvBackwardGrad(const framework::ExecutionContext& ctx,
                       const framework::Tensor& x_inv,
                       framework::Tensor* dx) const {
    Tensor x_inv_grad = *dx;
    x_inv_grad.mutable_data<T>(ctx.GetPlace());
    Tensor* dx_matrix = dx;
    dx_matrix->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    framework::Tensor tmp_out =
        ctx.AllocateTmpTensor<T, DeviceContext>(x_inv.dims(), dev_ctx);

    // dx = -(x^-1)'dout(x^-1)'
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto mat_dim_a0 = math::CreateMatrixDescriptor(x_inv_grad.dims(), 0, false);
    auto mat_dim_b0 = math::CreateMatrixDescriptor(x_inv.dims(), 0, true);
    blas.MatMul(x_inv_grad, mat_dim_a0, x_inv, mat_dim_b0, T(1), &tmp_out,
                T(0));
    auto mat_dim_a1 = math::CreateMatrixDescriptor(x_inv.dims(), 0, true);
    auto mat_dim_b1 = math::CreateMatrixDescriptor(tmp_out.dims(), 0, false);
    blas.MatMul(x_inv, mat_dim_a1, tmp_out, mat_dim_b1, T(-1), dx_matrix, T(0));
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = false;
    bool transpose_y = false;

    auto* input = ctx.Input<framework::LoDTensor>("X");

    Tensor x_inv(input->type());  // temporary tensor x_inv
    x_inv.Resize(input->dims());
    x_inv.mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, &x_inv);  // inverse operation for input X
    auto x = static_cast<const Tensor&>(x_inv);

    auto tmp_y = *ctx.Input<framework::Tensor>("Y");

    // special operation for input y
    // special case: input likes A(2,3,3) B(2,3)
    auto x_dim = input->dims();
    auto y_dim = tmp_y.dims();
    auto x_dim_size = x_dim.size();
    auto y_dim_size = y_dim.size();
    bool is_special = false;
    if (x_dim_size - y_dim_size == 1) {
      is_special = true;
      for (int i = 0; i < y_dim_size; i++) {
        if (x_dim[i] != y_dim[i]) {
          is_special = false;
        }
      }
    }
    Tensor y;
    if (is_special) {  // reshape
      std::vector<int64_t> y_dims_vec = paddle::framework::vectorize(y_dim);
      y_dims_vec.push_back(1);
      auto y_new_dims = framework::make_ddim(y_dims_vec);
      y.ShareDataWith(tmp_y).Resize(y_new_dims);
    } else {
      y.ShareDataWith(tmp_y);
    }

    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor y_conj(y.type());
    framework::Tensor x_conj(y.type());

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(x.dims());
    std::vector<std::int64_t> y_dims = vectorize(y.dims());
    std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int ndim = dout_dims.size();

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    // Case1 : x's or y's dim = 1
    if (x_ndim == 1 && y_ndim == 1) {
      if (dx) dx->mutable_data<T>(ctx.GetPlace());
      if (dy) dy->mutable_data<T>(ctx.GetPlace());
      if (dout.numel() == 1) {
        DotGradFunction<DeviceContext, T>()(&x, &y, &dout, dx, dy, ctx);
        InvBackwardGrad(ctx, x_inv, dx);
        if (is_special) {
          dy->Resize(y_dim);
        }
        return;
      }
    }

    bool is_broadcast = true;
    if (x_ndim <= 2 || y_ndim <= 2) {
      is_broadcast = false;
    } else if (x_ndim != y_ndim) {
      is_broadcast = true;
    } else {
      is_broadcast = !std::equal(x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2,
                                 y_dims.cbegin());
    }
    // Case2: no broadcast or no batch size, it aims to speed and it is same as
    // matmul in old version.
    if (!is_broadcast) {
      ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);
      framework::DDim dx_dims;
      if (dx) {
        dx_dims = dx->dims();
        if (dx_dims != x.dims()) {
          dx->Resize(x.dims());
        }
        // for complex
        ConjHelper<DeviceContext, T> conj_helper(ctx);
        conj_helper(y, y_conj);
      }
      framework::DDim dy_dims;
      if (dy) {
        dy_dims = dy->dims();
        if (dy_dims != y.dims()) {
          dy->Resize(y.dims());
        }
        // for complex
        ConjHelper<DeviceContext, T> conj_helper(ctx);
        conj_helper(x, x_conj);
      }
      if (transpose_x && transpose_y) {
        CalcInputGrad(ctx, y_conj, true, true, dout, true, false, dx);
        CalcInputGrad(ctx, dout, true, true, x_conj, true, false, dy);
      } else if (transpose_x) {
        CalcInputGrad(ctx, y_conj, false, false, dout, true, false, dx);
        CalcInputGrad(ctx, x_conj, false, false, dout, false, true, dy);
      } else if (transpose_y) {
        CalcInputGrad(ctx, dout, false, false, y_conj, false, true, dx);
        CalcInputGrad(ctx, dout, true, true, x_conj, false, true, dy);
      } else {
        CalcInputGrad(ctx, dout, false, false, y_conj, true, false, dx);
        if (dx) InvBackwardGrad(ctx, x_inv, dx);
        CalcInputGrad(ctx, x_conj, true, true, dout, false, true, dy);
      }

      if (dx) {
        if (dx_dims != x.dims()) {
          dx->Resize(dx_dims);
        }
      }
      if (dy) {
        if (dy_dims != y.dims()) {
          dy->Resize(dy_dims);

          if (is_special) {
            dy->Resize(y_dim);
          }
        }
      }
    } else {
      // Case3: broadcast. It need cost much time to reduce sum for the
      // broadcast and wastes the memory.
      // So we should avoid the case in reality.
      VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
                 "wastes the memory. So we should avoid the case in reality";
      Tensor dx_help, dy_help;

      ConjHelper<DeviceContext, T> conj_helper(ctx);
      conj_helper(x, x_conj);
      conj_helper(y, y_conj);
      if (transpose_x) {
        if (transpose_y) {
          // X'Y': dA = Y'G', dB = G'X'
          if (dx)
            MatMulFunction<DeviceContext, T>(&y_conj, &dout, y_dims, dout_dims,
                                             &dx_help, true, true, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&dout, &x_conj, dout_dims, x_dims,
                                             &dy_help, true, true, ctx);
        } else {
          // X'Y: dX = YG', dY = XG
          if (dx)
            MatMulFunction<DeviceContext, T>(&y_conj, &dout, y_dims, dout_dims,
                                             &dx_help, false, true, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&x_conj, &dout, x_dims, dout_dims,
                                             &dy_help, false, false, ctx);
        }
      } else {
        if (transpose_y) {
          // XY': dX = GY, dY = G'X
          if (dx)
            MatMulFunction<DeviceContext, T>(&dout, &y_conj, dout_dims, y_dims,
                                             &dx_help, false, false, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&dout, &x_conj, dout_dims, x_dims,
                                             &dy_help, true, false, ctx);
        } else {
          // XY: dX = GY', dY = X'G
          if (dx) {
            MatMulFunction<DeviceContext, T>(&dout, &y_conj, dout_dims, y_dims,
                                             &dx_help, false, true, ctx);
            InvBackwardGrad(ctx, x_inv, dx);
          }
          if (dy) {
            MatMulFunction<DeviceContext, T>(&x_conj, &dout, x_dims, dout_dims,
                                             &dy_help, true, false, ctx);
            if (is_special) {
              dy->Resize(y_dim);
            }
          }
        }
      }

      // get help dims
      const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
      const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

      std::vector<std::int64_t> dx_broadcast_dims(ndim);
      std::vector<std::int64_t> dy_broadcast_dims(ndim);

      std::fill(dx_broadcast_dims.data(),
                dx_broadcast_dims.data() + ndim - x_ndim, 1);
      std::fill(dy_broadcast_dims.data(),
                dy_broadcast_dims.data() + ndim - y_ndim, 1);
      std::copy(x_dims.data(), x_dims.data() + x_ndim,
                dx_broadcast_dims.data() + ndim - x_ndim);
      std::copy(y_dims.data(), y_dims.data() + y_ndim,
                dy_broadcast_dims.data() + ndim - y_ndim);

      std::vector<int> dx_reduce_dims;
      std::vector<int> dy_reduce_dims;
      for (int idx = 0; idx <= ndim - 3; idx++) {
        if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
          dx_reduce_dims.push_back(idx);
        }
        if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
          dy_reduce_dims.push_back(idx);
        }
      }
      // reduce sum to get grad by ReduceSum
      if (dx) {
        if (dx_reduce_dims.empty()) {
          *dx = std::move(dx_help);
        } else {
          ReduceSumForMatmulGrad<DeviceContext, T>(&dx_help, dx, dx_reduce_dims,
                                                   ctx);
        }
        dx->Resize(x.dims());
        InvBackwardGrad(ctx, x_inv, dx);
      }
      if (dy) {
        if (dy_reduce_dims.empty()) {
          *dy = std::move(dy_help);
        } else {
          ReduceSumForMatmulGrad<DeviceContext, T>(&dy_help, dy, dy_reduce_dims,
                                                   ctx);
        }
        dy->Resize(y.dims());
        if (is_special) {
          dy->Resize(y_dim);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
