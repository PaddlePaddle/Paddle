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
#include "glog/logging.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/solve_op.h"
#include "paddle/fluid/operators/tril_triu_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
static void triangular_solve(const DeviceContext& context, const Tensor& x,
                             const Tensor& y, Tensor* out, bool upper,
                             bool transpose, bool unitriangular) {
  // Tensor broadcast use eigen
  std::vector<int64_t> x_bst_dims_vec;
  std::vector<int64_t> y_bst_dims_vec;
  std::tie(x_bst_dims_vec, y_bst_dims_vec) = get_broadcast_dims(x, y);

  Tensor x_bst(x.type());
  TensorExpand<T, DeviceContext>(context, x, &x_bst, x_bst_dims_vec);

  Tensor y_bst(y.type());
  TensorExpand<T, DeviceContext>(context, y, &y_bst, y_bst_dims_vec);

  // TriangularSolveFunctor performs calculations in-place
  // x_clone should be a copy of 'x' after broadcast
  // out should be a copy of 'y' after broadcast
  Tensor x_clone(x.type());
  x_clone.Resize(framework::make_ddim(x_bst_dims_vec));
  x_clone.mutable_data<T>(context.GetPlace());
  framework::TensorCopy(x_bst, context.GetPlace(), context, &x_clone);

  out->Resize(framework::make_ddim(y_bst_dims_vec));
  out->mutable_data<T>(context.GetPlace());
  framework::TensorCopy(y_bst, context.GetPlace(), context, out);

  math::TriangularSolveFunctor<DeviceContext, T> functor;
  functor(context, &x_clone, out, /*left=*/true, upper, transpose,
          unitriangular);
}

template <typename DeviceContext, typename T>
class MatrixReduceSumFunctor {
 public:
  void operator()(const Tensor& input, Tensor* output,
                  const framework::ExecutionContext& ctx);
};

template <typename T>
class MatrixReduceSumFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const Tensor& in, Tensor* out,
                  const framework::ExecutionContext& ctx) {
    // For example: in's dim = [5, 3, 2, 7, 3] ; out's dim = [3, 1, 7, 3]
    // out_reduce_dim should be [0, 2]
    const std::vector<std::int64_t> in_dims = framework::vectorize(in.dims());
    auto in_size = in_dims.size();
    const std::vector<std::int64_t> out_dims =
        framework::vectorize(out->dims());
    auto out_size = out_dims.size();

    std::vector<std::int64_t> out_bst_dims(in_size);

    std::fill(out_bst_dims.data(), out_bst_dims.data() + in_size - out_size, 1);
    std::copy(out_dims.data(), out_dims.data() + out_size,
              out_bst_dims.data() + in_size - out_size);
    out->Resize(framework::make_ddim(out_bst_dims));

    std::vector<int> out_reduce_dims;
    for (size_t idx = 0; idx <= in_size - 3; idx++) {
      if (in_dims[idx] != 1 && out_bst_dims[idx] == 1) {
        out_reduce_dims.push_back(idx);
      }
    }

    ReduceKernelFunctor<platform::CPUDeviceContext, T, SumFunctor>(
        &in, out, out_reduce_dims, true, false, ctx)
        .template apply<T>();
    out->Resize(framework::make_ddim(out_dims));
  }
};

template <typename DeviceContext, typename T>
class TriangularSolveKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<framework::Tensor>("X");
    const auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");

    bool upper = ctx.template Attr<bool>("upper");
    bool transpose = ctx.template Attr<bool>("transpose");
    bool unitriangular = ctx.template Attr<bool>("unitriangular");

    const auto& dev_ctx = ctx.template device_context<DeviceContext>();
    triangular_solve<DeviceContext, T>(dev_ctx, *x, *y, out, upper, transpose,
                                       unitriangular);
  }
};

template <typename DeviceContext, typename T>
class TriangularSolveGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<framework::Tensor>("X");
    const auto* y = ctx.Input<framework::Tensor>("Y");
    const auto* out = ctx.Input<framework::Tensor>("Out");
    const auto* dout =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));

    bool upper = ctx.template Attr<bool>("upper");
    bool transpose = ctx.template Attr<bool>("transpose");
    bool unitriangular = ctx.template Attr<bool>("unitriangular");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    std::vector<int64_t> x_bst_dims_vec;
    std::vector<int64_t> y_bst_dims_vec;
    std::tie(x_bst_dims_vec, y_bst_dims_vec) = get_broadcast_dims(*x, *y);

    Tensor dy_bst(y->type());
    if (dy) {
      dy->mutable_data<T>(y->dims(), dev_ctx.GetPlace());
      dy_bst.Resize(framework::make_ddim(y_bst_dims_vec));
      dy_bst.mutable_data<T>(dev_ctx.GetPlace());

      // calculate x's conjugate for complex
      Tensor x_conj(x->type());
      platform::ForRange<DeviceContext> x_for_range(dev_ctx, x->numel());
      math::ConjFunctor<T> x_functor(
          x->data<T>(), x->numel(),
          x_conj.mutable_data<T>(x->dims(), dev_ctx.GetPlace()));
      x_for_range(x_functor);

      // reuse forward to get dy_bst, and the result has been broadcated.
      triangular_solve<DeviceContext, T>(dev_ctx, x_conj, *dout, &dy_bst, upper,
                                         !transpose, unitriangular);

      if (dy_bst.dims() == dy->dims()) {
        framework::TensorCopy(dy_bst, dev_ctx.GetPlace(), dev_ctx, dy);
      } else {
        MatrixReduceSumFunctor<DeviceContext, T> functor;
        functor(dy_bst, dy, ctx);
        dy->Resize(y->dims());
      }
    }

    Tensor dx_bst(x->type());
    if (dx) {
      dx->mutable_data<T>(x->dims(), dev_ctx.GetPlace());
      dx_bst.Resize(framework::make_ddim(x_bst_dims_vec));
      dx_bst.mutable_data<T>(dev_ctx.GetPlace());

      // calculate out's conjugate for complex
      Tensor out_conj(out->type());
      platform::ForRange<DeviceContext> out_for_range(dev_ctx, out->numel());
      math::ConjFunctor<T> out_functor(
          out->data<T>(), out->numel(),
          out_conj.mutable_data<T>(out->dims(), dev_ctx.GetPlace()));
      out_for_range(out_functor);

      auto blas = math::GetBlas<DeviceContext, T>(ctx);
      if (transpose) {
        auto mat_dim_a =
            math::CreateMatrixDescriptor(out_conj.dims(), 0, false);
        auto mat_dim_b = math::CreateMatrixDescriptor(dy_bst.dims(), 0, true);
        blas.MatMul(out_conj, mat_dim_a, dy_bst, mat_dim_b, static_cast<T>(-1),
                    &dx_bst, static_cast<T>(0));
      } else {
        auto mat_dim_a = math::CreateMatrixDescriptor(dy_bst.dims(), 0, false);
        auto mat_dim_b = math::CreateMatrixDescriptor(out_conj.dims(), 0, true);
        blas.MatMul(dy_bst, mat_dim_a, out_conj, mat_dim_b, static_cast<T>(-1),
                    &dx_bst, static_cast<T>(0));
      }

      Tensor dx_bst_upper(x->type());
      // get upper or lower triangular
      dx_bst_upper.Resize(dx_bst.dims());
      dx_bst_upper.mutable_data<T>(dev_ctx.GetPlace());

      const auto& dims = dx_bst.dims();
      const auto H = dims[dims.size() - 2];
      const auto W = dims[dims.size() - 1];
      platform::ForRange<DeviceContext> x_for_range(dev_ctx, dx_bst.numel());
      TrilTriuCompute<T> tril_triu_computer(dx_bst.data<T>(), unitriangular,
                                            !upper, H, W,
                                            dx_bst_upper.data<T>());
      x_for_range(tril_triu_computer);

      if (dx_bst_upper.dims() == dx->dims()) {
        framework::TensorCopy(dx_bst_upper, dev_ctx.GetPlace(), dev_ctx, dx);
      } else {
        MatrixReduceSumFunctor<DeviceContext, T> functor;
        functor(dx_bst_upper, dx, ctx);
        dx->Resize(x->dims());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
