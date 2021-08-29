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

template <typename T>
void PrintTensor(const framework::LoDTensor& src,
                 const framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < 10; ++i) {  // static_cast<int>(vec.size())
    VLOG(3) << "vec[" << i << "] : " << vec[i];
  }
}
template <typename T>
void PrintTensor(const framework::Tensor& src,
                 const framework::ExecutionContext& ctx) {
  std::vector<T> vec(src.numel());
  TensorToVector(src, ctx.device_context(), &vec);
  for (int i = 0; i < 10; ++i) {  // static_cast<int>(vec.size())
    VLOG(3) << "vec[" << i << "] : " << vec[i];
  }
}

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
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out = ctx.Input<framework::Tensor>("Out");

    VLOG(3) << "print input === 10 ==== ";
    PrintTensor<T>(*input, ctx);

    VLOG(3) << "print output === 10 ==== ";
    PrintTensor<T>(*out, ctx);

    auto out_ = *out;
    out_.mutable_data<T>(ctx.GetPlace());

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    if (dx) dx->mutable_data<T>(ctx.GetPlace());

    Tensor x_inv(input->type());  // temporary tensor x_inv
    x_inv.Resize(input->dims());
    x_inv.mutable_data<T>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    math::MatrixInverseFunctor<DeviceContext, T> mat_inv;

    mat_inv(dev_ctx, *input, &x_inv);  // inverse operation for input X

    auto x = static_cast<const Tensor&>(x_inv);

    auto x_dim = input->dims();
    auto y_dim = y->dims();
    auto x_dim_size = x_dim.size();
    auto y_dim_size = y_dim.size();
    std::vector<int64_t> x_dims_vec = paddle::framework::vectorize(x_dim);
    std::vector<int64_t> y_dims_vec = paddle::framework::vectorize(y_dim);

    std::vector<int64_t>::const_iterator f = x_dims_vec.begin();
    std::vector<int64_t>::const_iterator l = x_dims_vec.end() - 1;
    std::vector<int64_t> x_dims_vec_cut(f, l);

    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());

      if (y_dim_size == 1 ||
          (x_dim_size - 1 == y_dim_size) && y_dims_vec == x_dims_vec_cut) {
        VLOG(3) << "========== dy special case ===============";
        std::vector<int64_t> dout_dims_vec =
            paddle::framework::vectorize(dout.dims());
        dout_dims_vec.push_back(1);
        auto dout_new_dims = framework::make_ddim(dout_dims_vec);
        dout.Resize(dout_new_dims);
      }

      MatMulFunction<DeviceContext, T>(&x, &dout, dy, true, false, ctx);
      if (y_dim != dy->dims()) {
        dy->Resize(y_dim);
      }
    }
    if (dx) {
      auto blas = math::GetBlas<DeviceContext, T>(ctx);
      // Tensor dout_inv(out->type());  // temporary tensor
      // dout_inv.Resize(out->dims());
      // dout_inv.mutable_data<T>(ctx.GetPlace());

      // out->mutable_data<T>(ctx.GetPlace());
      VLOG(3) << "====== in dx before mat_inv ======";
      // mat_inv(dev_ctx, dout, &dout_inv);  // inverse operation for dout
      VLOG(3) << "====== in dx after mat_inv ======";

      Tensor grad_self(out->type());  // temporary tensor
      grad_self.Resize(dout.dims());
      grad_self.mutable_data<T>(ctx.GetPlace());

      MatMulFunction<DeviceContext, T>(&x, &dout, &grad_self, true, false, ctx);

      if (x_dim_size == 2 && y_dim_size == 2) {
        auto mat_dim_a1 =
            math::CreateMatrixDescriptor(grad_self.dims(), 0, false);
        auto mat_dim_b1 = math::CreateMatrixDescriptor(out->dims(), 0, true);
        blas.MatMul(grad_self, mat_dim_a1, *out, mat_dim_b1, T(-1), dx, T(0));
        // MatMulFunction<DeviceContext, T>(&grad_self, out, dx, false, true,
        // ctx);
        return;
      }

      if (y_dim_size == 1 ||
          (x_dim_size - 1 == y_dim_size) && y_dims_vec == x_dims_vec_cut) {
        VLOG(3) << "========== dx special case ===============";
        //  std::vector<int64_t> grad_self_dims_vec =
        //  paddle::framework::vectorize(grad_self.dims());
        //  grad_self_dims_vec.push_back(1);
        //  auto grad_self_new_dims = framework::make_ddim(grad_self_dims_vec);
        //  grad_self.Resize(grad_self_new_dims);

        std::vector<int64_t> out_dims_vec =
            paddle::framework::vectorize(out->dims());
        out_dims_vec.push_back(1);
        auto out_new_dims = framework::make_ddim(out_dims_vec);
        out_.Resize(out_new_dims);

        auto mat_dim_a1 =
            math::CreateMatrixDescriptor(grad_self.dims(), 0, false);
        auto mat_dim_b1 = math::CreateMatrixDescriptor(out_.dims(), 0, true);
        blas.MatMul(grad_self, mat_dim_a1, out_, mat_dim_b1, T(-1), dx, T(0));
        // MatMulFunction<DeviceContext, T>(&grad_self, &out_, dx, false, true,
        // ctx);
        return;
      }
      auto mat_dim_a1 =
          math::CreateMatrixDescriptor(grad_self.dims(), 0, false);
      auto mat_dim_b1 = math::CreateMatrixDescriptor(out->dims(), 0, true);
      blas.MatMul(grad_self, mat_dim_a1, *out, mat_dim_b1, T(-1), dx, T(0));
      // MatMulFunction<DeviceContext, T>(&grad_self, out, dx, false, true,
      // ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle
