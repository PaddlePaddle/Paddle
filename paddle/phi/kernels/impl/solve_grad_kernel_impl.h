/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
// #include "paddle/phi/kernels/solve_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/cpu/reduce.h"
#include "paddle/phi/kernels/expand_as_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_solve.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/solve_kernel_impl.h"
#include "paddle/phi/kernels/squeeze_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/gpu/reduce.h"
#endif

namespace phi {

// template <typename DeviceContext, typename T>
// void ReduceSumForSolve(const Tensor* input,
//                        Tensor* output,
//                        const std::vector<int>& reduce_dims,
//                        bool keep_dim,
//                        const paddle::framework::ExecutionContext& ctx) {
// #if defined(__NVCC__) || defined(__HIPCC__)
//   auto stream = ctx.cuda_device_context().stream();
//   TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
//       ctx.cuda_device_context(),
//       *input,
//       output,
//       kps::IdentityFunctor<T>(),
//       reduce_dims,
//       stream);
// #else
//   ReduceKernelFunctor<DeviceContext, T, ops::SumFunctor>(
//       input, output, reduce_dims, keep_dim, false, ctx)
//       .template apply<T>();
// #endif
// }

template <typename Context, typename T>
struct ReduceSumForSolvelGrad {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims,
                  bool keep_dims);
};

template <typename T>
struct ReduceSumForSolvelGrad<CPUContext, T> {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims,
                  bool keep_dims) {
    std::vector<int64_t> reduce_dims_tmp(reduce_dims.begin(),
                                         reduce_dims.end());
    phi::ReduceKernelImpl<CPUContext, T, T, phi::funcs::SumFunctor>(
        dev_ctx, input, output, reduce_dims_tmp, keep_dims, false);
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
struct ReduceSumForSolvelGrad<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims,
                  bool keep_dims) {
    phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        dev_ctx, input, output, kps::IdentityFunctor<T>(), reduce_dims);
  }
};
#endif

template <typename T, typename Context>
void SolveGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     const DenseTensor& dout,
                     const DenseTensor& out,
                     DenseTensor* dx,
                     DenseTensor* dy) {
  bool is_vector = false;
  is_vector = is_vector_rhs(x, y);

  DenseTensor tmp_y;
  if (is_vector) {
    dev_ctx.Alloc(&tmp_y, y.dtype());
    //   tmp_y.mutable_data(dev_ctx.GetPlace(), y.dtype());
    phi::SqueezeKernel<T, Context>(dev_ctx, y, {-1}, &tmp_y, nullptr);
    //   to_unsqueeze(dev_ctx, *y, &tmp_y);
  } else {
    tmp_y.Resize(y.dims());
    dev_ctx.Alloc(&tmp_y, y.dtype());
    phi::Copy(dev_ctx, y, dev_ctx.GetPlace(), false, &tmp_y);
    //   tmp_y.mutable_data(dev_ctx.GetPlace(), y->dtype());
    //   framework::TensorCopy(
    //       *y,
    //       dev_ctx.GetPlace(),
    //       dev_ctx.template device_context<platform::DeviceContext>(),
    //       &tmp_y);
  }

  DenseTensor tmp_x;
  tmp_x.Resize(x.dims());
  dev_ctx.Alloc(&tmp_x, x.dtype());
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &tmp_x);
  // tmp_x.mutable_data(dev_ctx.GetPlace(), x->dtype());
  // framework::TensorCopy(
  //     *x,
  //     dev_ctx.GetPlace(),
  //     dev_ctx.template device_context<platform::DeviceContext>(),
  //     &tmp_x);

  std::vector<int64_t> x_broadcast_dims;
  std::vector<int64_t> y_broadcast_dims;
  std::tie(x_broadcast_dims, y_broadcast_dims) =
      get_broadcast_dims(tmp_x, tmp_y);

  // tmp_dx
  DenseTensor tmp_dx;
  tmp_dx.Resize(phi::make_ddim(x_broadcast_dims));
  // tmp_dx.mutable_data<T>(dev_ctx.GetPlace());
  dev_ctx.template Alloc<T>(&tmp_dx);

  // tmp_dy
  DenseTensor tmp_dy;
  tmp_dy.Resize(phi::make_ddim(y_broadcast_dims));
  // tmp_dy.mutable_data<T>(dev_ctx.GetPlace());
  dev_ctx.template Alloc<T>(&tmp_dy);

  DenseTensor tmp_input(x.dtype());
  const auto& new_dims_vec = phi::funcs::getNewDimsVec(x.dims());
  tmp_input.Resize(phi::make_ddim(new_dims_vec));
  // tmp_x.mutable_data<T>(dev_ctx.GetPlace());
  dev_ctx.template Alloc<T>(&tmp_input);

  phi::funcs::TransposeNormal<Context, T> trans;
  std::vector<int> new_axis = phi::funcs::getNewAxis(x.dims().size());
  trans(dev_ctx, x, &tmp_input, new_axis);

  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    //   dy->mutable_data<T>(dev_ctx.GetPlace());
    // reuse linalg_solve forward logics to get tmp_dy
    linalg_solve<Context, T>(dev_ctx, tmp_input, dout, &tmp_dy);
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    //   dx->mutable_data<T>(dev_ctx.GetPlace());
    // to get dx
    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
    if (x.dims().size() == 2 && y.dims().size() == 2) {
      auto mat_dim_a1 =
          phi::funcs::CreateMatrixDescriptor(tmp_dy.dims(), 0, false);
      auto mat_dim_b1 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      blas.MatMul(tmp_dy, mat_dim_a1, out, mat_dim_b1, T(-1), &tmp_dx, T(0));
    } else if (is_vector_rhs(x, y)) {
      DenseTensor tmp_dy_;
      dev_ctx.Alloc(&tmp_dy_, y.dtype());
      // tmp_dy_.mutable_data(dev_ctx.GetPlace(), y->dtype());
      phi::UnsqueezeKernel<T, Context>(dev_ctx,
                                       tmp_dy,
                                       paddle::experimental::IntArray({-1}),
                                       &tmp_dy_,
                                       nullptr);
      // to_unsqueeze(dev_ctx, tmp_dy, &tmp_dy_);

      DenseTensor tmp_out_;
      dev_ctx.Alloc(&tmp_out_, out.dtype());
      // tmp_out_.mutable_data(dev_ctx.GetPlace(), out->dtype());
      phi::UnsqueezeKernel<T, Context>(dev_ctx,
                                       out,
                                       paddle::experimental::IntArray({-1}),
                                       &tmp_out_,
                                       nullptr);
      // to_unsqueeze(dev_ctx, *out, &tmp_out_);

      auto mat_dim_a1 =
          phi::funcs::CreateMatrixDescriptor(tmp_dy_.dims(), 0, false);
      auto mat_dim_b1 =
          phi::funcs::CreateMatrixDescriptor(tmp_out_.dims(), 0, true);
      blas.MatMul(
          tmp_dy_, mat_dim_a1, tmp_out_, mat_dim_b1, T(-1), &tmp_dx, T(0));
    } else {
      auto mat_dim_a1 =
          phi::funcs::CreateMatrixDescriptor(tmp_dy.dims(), 0, false);
      auto mat_dim_b1 = phi::funcs::CreateMatrixDescriptor(out.dims(), 0, true);
      blas.MatMul(tmp_dy, mat_dim_a1, out, mat_dim_b1, T(-1), &tmp_dx, T(0));
    }
  }

  if (y.dims() != tmp_dy.dims()) {
    DenseTensor dy_help;
    dy_help.Resize(tmp_dy.dims());
    dev_ctx.Alloc(&dy_help, tmp_dy.dtype());
    //   dy_help.mutable_data(dev_ctx.GetPlace(), tmp_dy.dtype());

    phi::Copy(dev_ctx, tmp_dy, dev_ctx.GetPlace(), false, &dy_help);
    //   framework::TensorCopy(
    //       tmp_dy,
    //       dev_ctx.GetPlace(),
    //       dev_ctx.template device_context<platform::DeviceContext>(),
    //       &dy_help);

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(x.dims());
    std::vector<std::int64_t> y_dims = vectorize(y.dims());
    std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

    if (is_vector_rhs(x, y)) {
      dout_dims.push_back(1);
    }

    int y_ndim = y_dims.size();
    int ndim = dout_dims.size();

    const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
              dy_broadcast_dims.data() + ndim - y_ndim);

    std::vector<int64_t> dy_reduce_dims;
    for (int64_t idx = 0; idx <= ndim - 3; idx++) {
      if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
        dy_reduce_dims.push_back(int64_t(idx));
      }
    }
    // reduce sum to get grad by ReduceSum
    if (dy) {
      if (dy_reduce_dims.empty()) {
        *dy = std::move(dy_help);
      } else {
        bool keep_dim = true;
        if (dy_help.dims().size() != dy->dims().size()) {
          keep_dim = false;
        }
        // ReduceSumForSolve<DeviceContext, T>(
        //     &dy_help, dy, dy_reduce_dims, keep_dim, dev_ctx);
        // call Reduce kernel
        // phi::Reduce<Context, T, phi::funcs::SumFunctor>(dev_ctx,
        //     dy_help,
        //     true,
        //     dy_reduce_dims,
        //     keep_dim,
        //     dy->dtype(),
        //     dy);
        // #if defined(__NVCC__) || defined(__HIPCC__)
        //     phi::funcs::ReduceKernel<T, T, kps::AddFunctor,
        //     kps::IdentityFunctor<T>>(
        //   dev_ctx, dy_help, dx, kps::IdentityFunctor<T>(), dy_reduce_dims);
        // #else
        //     phi::ReduceKernelImpl<Context, T, T, phi::funcs::SumFunctor>(
        //     dev_ctx, dy_help, dx, dy_reduce_dims, keep_dim, false);
        // #endif

        ReduceSumForSolvelGrad<Context, T>()(
            dev_ctx, dy_help, dy, dy_reduce_dims, keep_dim);
        //     void Reduce(const DeviceContext& dev_ctx,
        // const DenseTensor& x,
        // bool reduce_all,
        // const std::vector<int64_t>& dims,
        // bool keep_dim,
        // DataType out_dtype,
        // DenseTensor* out) {
      }
      dy->Resize(y.dims());
    }
  } else {
    phi::Copy(dev_ctx, tmp_dy, dev_ctx.GetPlace(), false, dy);
    //   framework::TensorCopy(
    //       tmp_dy,
    //       dev_ctx.GetPlace(),
    //       dev_ctx.template device_context<platform::DeviceContext>(),
    //       dy);
  }

  if (x.dims() != tmp_dx.dims()) {
    DenseTensor dx_help;
    dx_help.Resize(tmp_dx.dims());
    dev_ctx.Alloc(&dx_help, tmp_dx.dtype());
    //   dx_help.mutable_data(dev_ctx.GetPlace(), tmp_dx.dtype());

    phi::Copy(dev_ctx, tmp_dx, dev_ctx.GetPlace(), false, &dx_help);
    //   framework::TensorCopy(
    //       tmp_dx,
    //       dev_ctx.GetPlace(),
    //       dev_ctx.template device_context<platform::DeviceContext>(),
    //       &dx_help);

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(x.dims());
    std::vector<std::int64_t> y_dims = vectorize(y.dims());

    int x_ndim = x_dims.size();
    int ndim = x_broadcast_dims.size();

    const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
    std::vector<std::int64_t> dx_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);

    std::vector<int64_t> dx_reduce_dims;
    for (int64_t idx = 0; idx <= ndim - 3; idx++) {
      if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
        dx_reduce_dims.push_back(idx);
      }
    }
    // reduce sum to get grad by ReduceSum
    if (dx) {
      dev_ctx.template Alloc<T>(dx);
      // dx->mutable_data<T>(dev_ctx.GetPlace());
      if (dx_reduce_dims.empty()) {
        *dx = std::move(dx_help);
      } else {
        bool keep_dim = true;
        if (dx_help.dims().size() != dx->dims().size()) {
          keep_dim = false;
        }
        // ReduceSumForSolve<Context, T>(
        //     &dx_help, dx, dx_reduce_dims, keep_dim, dev_ctx);

        ReduceSumForSolvelGrad<Context, T>()(
            dev_ctx, dx_help, dx, dx_reduce_dims, keep_dim);
        // #if defined(__NVCC__) || defined(__HIPCC__)
        //     phi::funcs::ReduceKernel<T, T, kps::AddFunctor,
        //     kps::IdentityFunctor<T>>(
        //   dev_ctx, dx_help, dx, kps::IdentityFunctor<T>(), dx_reduce_dims);
        // #else
        //     phi::ReduceKernelImpl<Context, T, T, phi::funcs::SumFunctor>(
        //     dev_ctx, dx_help, dx, dx_reduce_dims, keep_dim, false);
        // #endif

        // phi::Reduce<Context, T, phi::funcs::SumFunctor>(dev_ctx,
        //         dx_help,
        //         true,
        //         dx_reduce_dims,
        //         keep_dim,
        //         dx->dtype(),
        //         dx);
      }
      dx->Resize(x.dims());
    }
  } else {
    phi::Copy(dev_ctx, tmp_dx, dev_ctx.GetPlace(), false, dx);
  }
}

}  // namespace phi
