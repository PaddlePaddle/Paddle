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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"
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
  std::vector<int64_t> x_broadcast_dims;
  std::vector<int64_t> y_broadcast_dims;
  std::tie(x_broadcast_dims, y_broadcast_dims) = _broadcast_batch_dims(x, y);

  Tensor x_bst(x.type());
  TensorExpand<T, DeviceContext>(context, x, &x_bst, x_broadcast_dims);

  Tensor y_bst(x.type());
  TensorExpand<T, DeviceContext>(context, y, &y_bst, y_broadcast_dims);

  // TriangularSolveFunctor performs calculations in-place
  // x_clone should be a copy of 'x' after broadcast
  // out should be a copy of 'y' after broadcast
  Tensor x_clone(x.type());
  x_clone.Resize(framework::make_ddim(x_broadcast_dims));
  x_clone.mutable_data<T>(context.GetPlace());
  framework::TensorCopy(x_bst, context.GetPlace(), context, &x_clone);

  out->Resize(framework::make_ddim(y_broadcast_dims));
  out->mutable_data<T>(context.GetPlace());
  framework::TensorCopy(y_bst, context.GetPlace(), context, out);

  math::TriangularSolveFunctor<DeviceContext，, T> functor;
  functor(context, &x_clone, out, /*left=*/true, upper, transpose,
          unitriangular);
}

template <typename DeviceContext, typename T>
static void ReduceSumForTriangularSolve(
    const Tensor& input, Tensor* output,
    const framework::ExecutionContext& ctx) {
  // For example: input's dim = [5, 3, 2, 7, 3] ; output's dim = [3, 1, 7, 3]
  // output_reduce_dims should be [0, 2]
  const std::vector<std::int64_t> input_dims =
      framework::vectorize(input.dims());
  auto input_size = input_dims.size();
  const std::vector<std::int64_t> output_dims =
      framework::vectorize(output->dims());
  auto output_size = output_dims.size();

  std::vector<std::int64_t> output_bst_dims(input_size);

  std::fill(output_bst_dims.data(),
            output_bst_dims.data() + input_size - output_size, 1);
  std::copy(output_dims.data(), output_dims.data() + output_size,
            output_bst_dims.data() + input_size - output_size);

  std::vector<int> output_reduce_dims;
  for (int idx = 0; idx <= input_size - 3; idx++) {
    if (input_dims[idx] != 1 && output_bst_dims[idx] == 1) {
      output_reduce_dims.push_back(idx);
    }
  }
#if defined(__NVCC__) || defined(__HIPCC__)
  auto stream = ctx.cuda_device_context().stream();
  TensorReduceFunctorImpl<T, T, CustomSum>(*input, output, output_reduce_dims,
                                           stream);
#else
  ReduceKernelFunctor<DeviceContext, T, SumFunctor>(
      input, output, output_reduce_dims, false, false, ctx)
      .template apply<T>();
#endif
}

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

    /*
    Tensor x_trans;
    auto transpose_axis = framework::vectorize(x->dims());
    std::swap(transpose_axis[-2], transpose_axis[-1]);
    TransCompute<DeviceContext, T>(x->dims(), dev_ctx, *x, &x_trans,
    transpose_axis);
    */

    Tensor dy_bst(y->type());
    if (dy) {
      dy->mutable_data<T>(y->dims(), dev_ctx.GetPlace());

      // calculate x's conjugate for complex
      Tensor x_conj(x->type());
      platform::ForRange<DeviceContext> for_range(dev_ctx, x->numel());
      math::ConjFunctor<T> functor(
          x->data<T>(), x->numel(),
          x_conj.mutable_data<T>(x->dims(), dev_ctx.GetPlace()));
      for_range(functor);

      // reuse triangular_solve's forward to get dy_bst, the result is
      // broadcated.
      triangular_solve<DeviceContext, T>(dev_ctx, x_conj, *dout, &dy_bst, upper,
                                         !transpose, unitriangular);

      if (dy_bst.dims() == dy->dims()) {
        framework::TensorCopy(dy_bst, dev_ctx.GetPlace(), dev_ctx, dy);
      } else {
        ReduceSumForTriangularSolve(dy_bst, dy, ctx);
        dy->Resize(y->dims());
      }
    }

    Tensor dx_bst(x->type());
    if (dx) {
      dx->mutable_data<T>(x->dims(), dev_ctx.GetPlace());

      // calculate out's conjugate for complex
      Tensor out_conj(out->type());
      platform::ForRange<DeviceContext> for_range(dev_ctx, out->numel());
      math::ConjFunctor<T> functor(
          out->data<T>(), out->numel(),
          out_conj.mutable_data<T>(out->dims(), dev_ctx.GetPlace()));
      for_range(functor);

      auto blas = math::GetBlas<DeviceContext, T>(ctx);
      if (transpose) {
        blas.MatMul(out_conj, false, dy_bst, true, static_cast<T>(-1), &dx_bst,
                    static_cast<T>(0));
      } else {
        blas.MatMul(dy_bst, false, out_conj, true, static_cast<T>(-1), &dx_bst,
                    static_cast<T>(0));
      }

      Tensor dx_bst_upper(x.type());
      // get upper or lower triangular
      dx_bst_upper.Resize(dx_bst.dims());
      dx_bst_upper.mutable_data<T>(dev_ctx.GetPlace());
      TrilTriuCompute<T> tril_triu_computer(dx_bst.data<T>(), unitriangular,
                                            !upper, a->dims()[-1],
                                            a->dims()[-1], &dx_bst_upper);
      platform::ForRange<DeviceContext> for_range(
          dev_ctx, static_cast<size_t>(x->numel()));
      for_range(tril_triu_computer);

      if (dx_bst_upper.dims() == dx->dims()) {
        framework::TensorCopy(dx_bst_upper, dev_ctx.GetPlace(), dev_ctx, dx);
      } else {
        ReduceSumForTriangularSolve(dx_bst_upper, dx, ctx);
        dx->Resize(x->dims());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
