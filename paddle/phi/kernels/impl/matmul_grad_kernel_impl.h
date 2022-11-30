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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/reduce_functor.h"
#include "paddle/phi/kernels/impl/dot_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/gpu/reduce.h"
#endif

namespace phi {

template <typename Context, typename T>
struct ReduceSumForMatmulGrad {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims);
};

template <typename T>
struct ReduceSumForMatmulGrad<CPUContext, T> {
  void operator()(const CPUContext& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims) {
    std::vector<int64_t> reduce_dims_tmp(reduce_dims.begin(),
                                         reduce_dims.end());
    funcs::ReduceKernelImpl<CPUContext, T, T, phi::funcs::SumFunctor>(
        dev_ctx, input, output, reduce_dims_tmp, true, false);
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T>
struct ReduceSumForMatmulGrad<GPUContext, T> {
  void operator()(const GPUContext& dev_ctx,
                  const DenseTensor& input,
                  DenseTensor* output,
                  const std::vector<int>& reduce_dims) {
    funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        dev_ctx, input, output, kps::IdentityFunctor<T>(), reduce_dims);
  }
};
#endif

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static DenseTensor FoldInitDims(const DenseTensor& input) {
  DenseTensor output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename Context, typename T>
static DenseTensor FoldHeadAndLastDims(const Context& dev_ctx,
                                       const DenseTensor& input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }
  DenseTensor output = EmptyLike<T, Context>(dev_ctx, input);
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  std::vector<int> axis = {1, 0, 2};
  funcs::Transpose<Context, T, 3> trans;
  trans(dev_ctx, input, &output, axis);
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});
  return output;
}

template <typename Context, typename T>
void MatMul(const Context& dev_ctx,
            const DenseTensor& a,
            bool trans_a,
            const DenseTensor& b,
            bool trans_b,
            DenseTensor* out,
            bool flag = false) {
  dev_ctx.template Alloc<T>(out);
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a.dims(), 0, trans_a);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b.dims(), 0, trans_b);
  if (a.dims().size() == 3 && b.dims().size() <= 2) {
    // the transpose_X must be false, if is true, the transpose cost much time
    if (!trans_a) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    }
  }
  blas.MatMul(a.data<T>(),
              mat_dim_a,
              b.data<T>(),
              mat_dim_b,
              static_cast<T>(1),
              dev_ctx.template Alloc<T>(out),
              static_cast<T>(flag));
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static DDim RowMatrixFromVector(const DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return phi::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static DDim ColumnMatrixFromVector(const DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return phi::make_ddim({y_dim[0], 1});
}

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    DenseTensor* x, const phi::funcs::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

static void ReshapeXYOutIntoMatrixSequence(DenseTensor* x,
                                           DenseTensor* y,
                                           DenseTensor* out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({(std::max)(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename T, typename Context>
void CalcInputGrad(const Context& dev_ctx,
                   const DenseTensor& a,
                   bool trans_a,
                   bool is_fold_init_dims_a,
                   const DenseTensor& b,
                   bool trans_b,
                   bool is_fold_init_dims_b,
                   DenseTensor* out,
                   bool flag = false) {
  if (out == nullptr) return;
  bool need_combine =
      (a.dims().size() == 3 || b.dims().size() == 3) && out->dims().size() == 2;
  if (!need_combine) {
    MatMul<Context, T>(dev_ctx, a, trans_a, b, trans_b, out, flag);
  } else {
    MatMul<Context, T>(
        dev_ctx,
        is_fold_init_dims_a ? FoldInitDims(a)
                            : FoldHeadAndLastDims<Context, T>(dev_ctx, a),
        trans_a,
        is_fold_init_dims_b ? FoldInitDims(b)
                            : FoldHeadAndLastDims<Context, T>(dev_ctx, b),
        trans_b,
        out,
        flag);
  }
}

template <typename T, typename Context>
void MatmulGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& out_grad,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy) {
  // get dims
  std::vector<std::int64_t> x_dims = vectorize(x.dims());
  std::vector<std::int64_t> y_dims = vectorize(y.dims());
  std::vector<std::int64_t> dout_dims = vectorize(out_grad.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  // Case1 : x's or y's dim = 1
  if (x_ndim == 1 && y_ndim == 1) {
    if (dx) dev_ctx.template Alloc<T>(dx);
    if (dy) dev_ctx.template Alloc<T>(dy);
    if (out_grad.numel() == 1) {
      DotGradFunction<Context, T>()(dev_ctx, &x, &y, &out_grad, dx, dy);
      return;
    }
  }

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  // for complex
  DenseTensor x_conj;
  DenseTensor y_conj;

  // Case2: no broadcast or no batch size, it aims to speed and it is same as
  // matmul in old version.
  if (!is_broadcast) {
    DenseTensor x_help = x;
    DenseTensor y_help = y;
    DenseTensor out_grad_help = out_grad;

    ReshapeXYOutIntoMatrixSequence(
        &x_help, &y_help, &out_grad_help, transpose_x, transpose_y);

    DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x_help.dims()) {
        dx->Resize(x_help.dims());
      }

      y_conj = Conj<T>(dev_ctx, y_help);
    }

    DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y_help.dims()) {
        dy->Resize(y_help.dims());
      }

      x_conj = Conj<T>(dev_ctx, x_help);
    }

    if (transpose_x && transpose_y) {
      CalcInputGrad<T>(
          dev_ctx, y_conj, true, true, out_grad_help, true, false, dx);
      CalcInputGrad<T>(
          dev_ctx, out_grad_help, true, true, x_conj, true, false, dy);
    } else if (transpose_x) {
      CalcInputGrad<T>(
          dev_ctx, y_conj, false, false, out_grad_help, true, false, dx);
      CalcInputGrad<T>(
          dev_ctx, x_conj, false, false, out_grad_help, false, true, dy);
    } else if (transpose_y) {
      CalcInputGrad<T>(
          dev_ctx, out_grad_help, false, false, y_conj, false, true, dx);
      CalcInputGrad<T>(
          dev_ctx, out_grad_help, true, true, x_conj, false, true, dy);
    } else {
      CalcInputGrad<T>(
          dev_ctx, out_grad_help, false, false, y_conj, true, false, dx);
      CalcInputGrad<T>(
          dev_ctx, x_conj, true, true, out_grad_help, false, true, dy);
    }

    if (dx) {
      if (dx_dims != x_help.dims()) {
        dx->Resize(dx_dims);
      }
    }
    if (dy) {
      if (dy_dims != y_help.dims()) {
        dy->Resize(dy_dims);
      }
    }
  } else {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";
    x_conj = Conj<T>(dev_ctx, x);
    y_conj = Conj<T>(dev_ctx, y);

    DenseTensor dx_help;
    DenseTensor dy_help;

    if (transpose_x) {
      if (transpose_y) {
        // X'Y': dA = Y'G', dB = G'X'
        if (dx)
          MatMulFunction<Context, T>(dev_ctx,
                                     y_conj,
                                     out_grad,
                                     y_dims,
                                     dout_dims,
                                     &dx_help,
                                     true,
                                     true);
        if (dy)
          MatMulFunction<Context, T>(dev_ctx,
                                     out_grad,
                                     x_conj,
                                     dout_dims,
                                     x_dims,
                                     &dy_help,
                                     true,
                                     true);
      } else {
        // X'Y: dX = YG', dY = XG
        if (dx)
          MatMulFunction<Context, T>(dev_ctx,
                                     y_conj,
                                     out_grad,
                                     y_dims,
                                     dout_dims,
                                     &dx_help,
                                     false,
                                     true);
        if (dy)
          MatMulFunction<Context, T>(dev_ctx,
                                     x_conj,
                                     out_grad,
                                     x_dims,
                                     dout_dims,
                                     &dy_help,
                                     false,
                                     false);
      }
    } else {
      if (transpose_y) {
        // XY': dX = GY, dY = G'X
        if (dx)
          MatMulFunction<Context, T>(dev_ctx,
                                     out_grad,
                                     y_conj,
                                     dout_dims,
                                     y_dims,
                                     &dx_help,
                                     false,
                                     false);
        if (dy)
          MatMulFunction<Context, T>(dev_ctx,
                                     out_grad,
                                     x_conj,
                                     dout_dims,
                                     x_dims,
                                     &dy_help,
                                     true,
                                     false);
      } else {
        // XY: dX = GY', dY = X'G
        if (dx)
          MatMulFunction<Context, T>(dev_ctx,
                                     out_grad,
                                     y_conj,
                                     dout_dims,
                                     y_dims,
                                     &dx_help,
                                     false,
                                     true);
        if (dy)
          MatMulFunction<Context, T>(dev_ctx,
                                     x_conj,
                                     out_grad,
                                     x_dims,
                                     dout_dims,
                                     &dy_help,
                                     true,
                                     false);
      }
    }

    // get help dims
    const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
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
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dx_help, dx, dx_reduce_dims);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      if (dy_reduce_dims.empty()) {
        *dy = std::move(dy_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dy_help, dy, dy_reduce_dims);
      }
      dy->Resize(y.dims());
    }
    // Get the OutputGrad(out)
  }
}

template <typename T, typename Context>
void MatmulDoubleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& dout,
                            const paddle::optional<DenseTensor>& ddx_opt,
                            const paddle::optional<DenseTensor>& ddy_opt,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* dx,
                            DenseTensor* dy,
                            DenseTensor* ddout) {
  paddle::optional<DenseTensor> ddx;
  paddle::optional<DenseTensor> ddy;
  if (!ddx_opt && (dy || ddout)) {
    DenseTensor ddx_tmp = phi::FullLike<T, Context>(dev_ctx, x, Scalar(0.0));
    ddx = paddle::make_optional<DenseTensor>(ddx_tmp);
  } else {
    ddx = ddx_opt;
  }
  if (!ddy_opt && (dx || ddout)) {
    DenseTensor ddy_tmp = phi::FullLike<T, Context>(dev_ctx, y, Scalar(0.0));
    ddy = paddle::make_optional<DenseTensor>(ddy_tmp);
  } else {
    ddy = ddy_opt;
  }
  // Get dims from the input x, y, output_grad
  std::vector<std::int64_t> x_dims = vectorize(x.dims());
  std::vector<std::int64_t> y_dims = vectorize(y.dims());
  std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  // Case1 : x's or y's dim = 1
  if (x_ndim == 1 && y_ndim == 1) {
    DotDoubleGradFunction<Context, T>()(
        dev_ctx, &x, &y, &dout, ddx.get_ptr(), ddy.get_ptr(), dx, dy, ddout);
    return;
  }

  DenseTensor x_conj;
  DenseTensor y_conj;
  DenseTensor dout_conj;

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  if (!is_broadcast) {
    // Case2: no broadcast or no batch size
    DenseTensor x_help = x;
    DenseTensor y_help = y;
    DenseTensor dout_help = dout;
    ReshapeXYOutIntoMatrixSequence(
        &x_help, &y_help, &dout_help, transpose_x, transpose_y);
    DDim dx_dims;

    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x_help.dims()) {
        dx->Resize(x_help.dims());
      }
    }

    DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y_help.dims()) {
        dy->Resize(y_help.dims());
      }
    }

    DDim ddout_dims;
    if (ddout) {
      ddout_dims = ddout->dims();
      if (ddout_dims != dout_help.dims()) {
        ddout->Resize(dout_help.dims());
      }

      x_conj = Conj<T>(dev_ctx, x_help);
      y_conj = Conj<T>(dev_ctx, y_help);
    }

    if (dx || dy) {
      dout_conj = Conj<T>(dev_ctx, dout_help);
    }

    bool ddout_flag = false;
    if (ddx) {
      auto ddx_mat = ddx.get();
      if (ddx_mat.dims() != x_help.dims()) {
        ddx_mat.Resize(x_help.dims());
      }
      if (dy) {
        if (transpose_x && transpose_y) {
          // dy = dout' * ddx'
          CalcInputGrad<T>(
              dev_ctx, dout_conj, true, true, ddx_mat, true, false, dy, false);
        } else if (transpose_x) {
          // dy = ddx * dout
          CalcInputGrad<T>(dev_ctx,
                           ddx_mat,
                           false,
                           false,
                           dout_conj,
                           false,
                           true,
                           dy,
                           false);
        } else if (transpose_y) {
          // dy = dout' * ddx
          CalcInputGrad<T>(
              dev_ctx, dout_conj, true, true, ddx_mat, false, true, dy, false);
        } else {
          // dy = ddx' * dout
          CalcInputGrad<T>(
              dev_ctx, ddx_mat, true, true, dout_conj, false, true, dy, false);
        }
      }

      if (ddout) {
        CalcInputGrad<T>(dev_ctx,
                         ddx_mat,
                         transpose_x,
                         true,
                         y_conj,
                         transpose_y,
                         false,
                         ddout,
                         ddout_flag);
        ddout_flag = true;
      }
    }
    if (ddy) {
      auto ddy_mat = ddy.get();
      if (ddy_mat.dims() != y_help.dims()) {
        ddy_mat.Resize(y_help.dims());
      }
      if (dx) {
        if (transpose_x && transpose_y) {
          // dx = ddy' * dout'
          CalcInputGrad<T>(
              dev_ctx, ddy_mat, true, true, dout_conj, true, false, dx, false);
        } else if (transpose_x) {
          // dx = ddy * dout'
          CalcInputGrad<T>(dev_ctx,
                           ddy_mat,
                           false,
                           false,
                           dout_conj,
                           true,
                           false,
                           dx,
                           false);
        } else if (transpose_y) {
          // dx = dout * ddy
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           false,
                           false,
                           ddy_mat,
                           false,
                           true,
                           dx,
                           false);
        } else {
          // dx = dout * ddy'
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           false,
                           false,
                           ddy_mat,
                           true,
                           false,
                           dx,
                           false);
        }
      }

      if (ddout) {
        CalcInputGrad<T>(dev_ctx,
                         x_conj,
                         transpose_x,
                         true,
                         ddy_mat,
                         transpose_y,
                         false,
                         ddout,
                         ddout_flag);
      }
    }

    if (dx) {
      if (dx_dims != x_help.dims()) {
        dx->Resize(dx_dims);
      }
    }

    if (dy) {
      if (dy_dims != y_help.dims()) {
        dy->Resize(dy_dims);
      }
    }

    if (ddout) {
      if (ddout_dims != dout_help.dims()) {
        ddout->Resize(ddout_dims);
      }
    }
  } else {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";
    if (dx || dy) {
      dout_conj = Conj<T>(dev_ctx, dout);
    }
    if (ddout) {
      x_conj = Conj<T>(dev_ctx, x);
      y_conj = Conj<T>(dev_ctx, y);
    }

    DenseTensor dx_help;
    DenseTensor dy_help;

    if (transpose_x) {
      if (transpose_y) {
        if (dx && ddy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     ddy.get(),
                                     dout_conj,
                                     y_dims,
                                     dout_dims,
                                     &dx_help,
                                     true,
                                     true);
        }
        if (dy && ddx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     ddx.get(),
                                     dout_dims,
                                     x_dims,
                                     &dy_help,
                                     true,
                                     true);
        }
      } else {
        if (dx && ddy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     ddy.get(),
                                     dout_conj,
                                     y_dims,
                                     dout_dims,
                                     &dx_help,
                                     false,
                                     true);
        }
        if (dy && ddx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     ddx.get(),
                                     dout_conj,
                                     x_dims,
                                     dout_dims,
                                     &dy_help,
                                     false,
                                     false);
        }
      }
    } else {
      if (transpose_y) {
        if (dx && ddy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     ddy.get(),
                                     dout_dims,
                                     y_dims,
                                     &dx_help,
                                     false,
                                     false);
        }
        if (dy && ddx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     ddx.get(),
                                     dout_dims,
                                     x_dims,
                                     &dy_help,
                                     true,
                                     false);
        }
      } else {
        if (dx && ddy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     ddy.get(),
                                     dout_dims,
                                     y_dims,
                                     &dx_help,
                                     false,
                                     true);
        }
        if (dy && ddx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     ddx.get(),
                                     dout_conj,
                                     x_dims,
                                     dout_dims,
                                     &dy_help,
                                     true,
                                     false);
        }
      }
    }

    // get help dims
    const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
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
    // Reduce sum to get grad by ReduceSum
    if (dx) {
      if (dx_reduce_dims.empty()) {
        *dx = std::move(dx_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dx_help, dx, dx_reduce_dims);
      }
      dx->Resize(x.dims());
    }
    if (dy) {
      if (dy_reduce_dims.empty()) {
        *dy = std::move(dy_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, dy_help, dy, dy_reduce_dims);
      }
      dy->Resize(y.dims());
    }

    if (ddout) {
      // Calculate the gradient of OutputGrad(Out)
      if (ddx) {
        MatMulFunction<Context, T>(dev_ctx,
                                   ddx.get(),
                                   y_conj,
                                   x_dims,
                                   y_dims,
                                   ddout,
                                   transpose_x,
                                   transpose_y);
      }

      if (ddy) {
        MatMulFunction<Context, T>(dev_ctx,
                                   x_conj,
                                   ddy.get(),
                                   x_dims,
                                   y_dims,
                                   ddout,
                                   transpose_x,
                                   transpose_y,
                                   true);
      }
    }
  }
}

template <typename T, typename Context>
void MatmulTripleGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            const DenseTensor& dout,
                            const paddle::optional<DenseTensor>& ddx_opt,
                            const paddle::optional<DenseTensor>& ddy_opt,
                            const paddle::optional<DenseTensor>& d_dx_opt,
                            const paddle::optional<DenseTensor>& d_dy_opt,
                            const paddle::optional<DenseTensor>& d_ddout_opt,
                            bool transpose_x,
                            bool transpose_y,
                            DenseTensor* out_d_x,
                            DenseTensor* out_d_y,
                            DenseTensor* out_d_dout,
                            DenseTensor* out_d_ddx,
                            DenseTensor* out_d_ddy) {
  paddle::optional<DenseTensor> ddx;
  paddle::optional<DenseTensor> ddy;
  paddle::optional<DenseTensor> d_dx;
  paddle::optional<DenseTensor> d_dy;
  paddle::optional<DenseTensor> d_ddout;

  if (!ddx_opt && (out_d_y || out_d_dout)) {
    DenseTensor ddx_tmp =
        phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(0.0));
    ddx = paddle::make_optional<DenseTensor>(ddx_tmp);
  } else {
    ddx = ddx_opt;
  }
  if (!ddy_opt && (out_d_x || out_d_dout)) {
    DenseTensor ddy_tmp =
        phi::FullLike<T, Context>(dev_ctx, y, static_cast<T>(0.0));
    ddy = paddle::make_optional<DenseTensor>(ddy_tmp);
  } else {
    ddy = ddy_opt;
  }

  if (!d_ddout_opt && (out_d_y || out_d_x || out_d_ddy || out_d_ddx)) {
    DenseTensor d_ddout_tmp =
        phi::FullLike<T, Context>(dev_ctx, dout, static_cast<T>(0.0));
    d_ddout = paddle::make_optional<DenseTensor>(d_ddout_tmp);
  } else {
    d_ddout = d_ddout_opt;
  }

  if (!d_dx_opt && (out_d_ddy || out_d_dout)) {
    DenseTensor d_dx_tmp =
        phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(0.0));
    d_dx = paddle::make_optional<DenseTensor>(d_dx_tmp);
  } else {
    d_dx = d_dx_opt;
  }

  if (!d_dy_opt && (out_d_ddx || out_d_dout)) {
    DenseTensor d_dy_tmp =
        phi::FullLike<T, Context>(dev_ctx, y, static_cast<T>(0.0));
    d_dy = paddle::make_optional<DenseTensor>(d_dy_tmp);
  } else {
    d_dy = d_dy_opt;
  }
  // Get dims from the input x, y, output_grad
  std::vector<std::int64_t> x_dims = vectorize(x.dims());
  std::vector<std::int64_t> y_dims = vectorize(y.dims());
  std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();
  int ndim = dout_dims.size();

  // Case1 : x's and y's dim = 1
  if (x_ndim == 1 && y_ndim == 1) {
    VLOG(3) << "========  MatMulV2TripleGradKernel, Compute ====== Case 1";
    DotTripleGradFunction<Context, T>()(dev_ctx,
                                        &x,
                                        &y,
                                        ddx.get_ptr(),
                                        ddy.get_ptr(),
                                        d_dx.get_ptr(),
                                        d_dy.get_ptr(),
                                        &dout,
                                        d_ddout.get_ptr(),
                                        out_d_x,
                                        out_d_y,
                                        out_d_dout,
                                        out_d_ddx,
                                        out_d_ddy);
    return;
  }

  DenseTensor x_conj;
  DenseTensor y_conj;
  DenseTensor dout_conj;
  DenseTensor ddx_conj;
  DenseTensor ddy_conj;

  bool is_broadcast = true;
  if (x_ndim <= 2 || y_ndim <= 2) {
    is_broadcast = false;
  } else if (x_ndim != y_ndim) {
    is_broadcast = true;
  } else {
    is_broadcast = !std::equal(
        x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2, y_dims.cbegin());
  }

  if (!is_broadcast) {
    // Case2: no broadcast or no batch size
    VLOG(3) << "========  MatMulV2TripleGradKernel, Compute ====== Case 2";
    DenseTensor x_help = x;
    DenseTensor y_help = y;
    DenseTensor dout_help = dout;

    DenseTensor ddx_help;
    DenseTensor ddy_help;
    ReshapeXYOutIntoMatrixSequence(
        &x_help, &y_help, &dout_help, transpose_x, transpose_y);
    if (ddx) {
      ddx_help = ddx.get();
      if (ddx_help.dims() != x_help.dims()) {
        ddx_help.Resize(x_help.dims());
      }
    }

    if (ddy) {
      ddy_help = ddy.get();
      if (ddy_help.dims() != y_help.dims()) {
        ddy_help.Resize(y_help.dims());
      }
    }

    DDim out_dx_dims;
    if (out_d_x) {
      out_dx_dims = out_d_x->dims();
      if (out_dx_dims != x_help.dims()) {
        out_d_x->Resize(x_help.dims());
      }
      if (ddy) {
        ddy_conj = Conj<T>(dev_ctx, ddy_help);
      }
    }
    DDim out_dy_dims;
    if (out_d_y) {
      out_dy_dims = out_d_y->dims();
      if (out_dy_dims != y_help.dims()) {
        out_d_y->Resize(y_help.dims());
      }
      if (ddx) {
        ddx_conj = Conj<T>(dev_ctx, ddx_help);
      }
    }
    DDim out_d_dout_dims;
    if (out_d_dout) {
      out_d_dout_dims = out_d_dout->dims();
      if (out_d_dout_dims != dout_help.dims()) {
        out_d_dout->Resize(dout_help.dims());
      }
      if (ddx && !ddx_conj.IsInitialized()) {
        ddx_conj = Conj<T>(dev_ctx, ddx_help);
      }
      if (ddy && !ddy_conj.IsInitialized()) {
        ddy_conj = Conj<T>(dev_ctx, ddy_help);
      }
    }
    DDim out_d_ddx_dims;
    if (out_d_ddx) {
      out_d_ddx_dims = out_d_ddx->dims();
      if (out_d_ddx_dims != x_help.dims()) {
        out_d_ddx->Resize(x_help.dims());
      }
      dout_conj = Conj<T>(dev_ctx, dout_help);
      y_conj = Conj<T>(dev_ctx, y_help);
    }
    DDim out_d_ddy_dims;
    if (out_d_ddy) {
      out_d_ddy_dims = out_d_ddy->dims();
      if (out_d_ddy_dims != y_help.dims()) {
        out_d_ddy->Resize(y_help.dims());
      }
      if (dout_conj.IsInitialized()) {
        dout_conj = Conj<T>(dev_ctx, dout_help);
      }
      x_conj = Conj<T>(dev_ctx, x_help);
    }

    bool d_dout_flag = false;
    bool d_ddx_flag = false;
    bool d_ddy_flag = false;
    if (d_ddout) {
      auto d_ddout_mat = d_ddout.get();
      if (d_ddout_mat.dims() != dout_help.dims()) {
        d_ddout_mat.Resize(dout_help.dims());
      }

      if (out_d_y && ddx) {
        if (transpose_x && transpose_y) {
          // out_d_y = d_ddout' * ddx'
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           true,
                           true,
                           ddx_conj,
                           true,
                           false,
                           out_d_y,
                           false);
        } else if (transpose_x) {
          // out_d_y = ddx * d_ddout
          CalcInputGrad<T>(dev_ctx,
                           ddx_conj,
                           false,
                           false,
                           d_ddout_mat,
                           false,
                           true,
                           out_d_y,
                           false);
        } else if (transpose_y) {
          // out_d_y = d_ddout' * ddx
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           true,
                           true,
                           ddx_conj,
                           false,
                           true,
                           out_d_y,
                           false);
        } else {
          // out_d_y = ddx' * d_ddout
          CalcInputGrad<T>(dev_ctx,
                           ddx_conj,
                           true,
                           true,
                           d_ddout_mat,
                           false,
                           true,
                           out_d_y,
                           false);
        }
      }
      if (out_d_x && ddy) {
        if (transpose_x && transpose_y) {
          // out_d_x = ddy' * d_ddout'
          CalcInputGrad<T>(dev_ctx,
                           ddy_conj,
                           true,
                           true,
                           d_ddout_mat,
                           true,
                           false,
                           out_d_x,
                           false);
        } else if (transpose_x) {
          // out_d_x = ddy * d_ddout'
          CalcInputGrad<T>(dev_ctx,
                           ddy_conj,
                           false,
                           false,
                           d_ddout_mat,
                           true,
                           false,
                           out_d_x,
                           false);
        } else if (transpose_y) {
          // out_d_x = d_ddout * ddy
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           false,
                           false,
                           ddy_conj,
                           false,
                           true,
                           out_d_x,
                           false);
        } else {
          // out_d_x = d_ddout * ddy'
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           false,
                           false,
                           ddy_conj,
                           true,
                           false,
                           out_d_x,
                           false);
        }
      }

      // equations:
      // d_ddx = DOut * D_DY + Y * D_DDOut
      // Let: d_ddx1 = Y * D_DDOut
      // Let: d_ddx2 = DOut * D_DY

      // d_ddy = DOut * D_DX + X * D_DDOut
      // Let: d_ddy1 = X * D_DDOut
      // Let: d_ddy2 = DOut * D_DX

      // d_dout = DDY * D_DX + DDX * D_DY
      // Let: d_dout1 = DDX * D_DY
      // Let: d_dout2 = DDY * D_DX

      // compute d_ddx1
      if (out_d_ddx) {
        if (transpose_x && transpose_y) {
          // out_d_ddx1 = y' * d_ddout'
          CalcInputGrad<T>(dev_ctx,
                           y_conj,
                           true,
                           true,
                           d_ddout_mat,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        } else if (transpose_x) {
          // out_d_ddx1 = y * d_ddout'
          CalcInputGrad<T>(dev_ctx,
                           y_conj,
                           false,
                           false,
                           d_ddout_mat,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        } else if (transpose_y) {
          // out_d_ddx1 = d_ddout * y
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           false,
                           false,
                           y_conj,
                           false,
                           true,
                           out_d_ddx,
                           d_ddx_flag);
        } else {
          // out_d_ddx1 = d_ddout * y'
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           false,
                           false,
                           y_conj,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        }
        d_ddx_flag = true;
      }

      // compute d_ddy1
      if (out_d_ddy) {
        if (transpose_x && transpose_y) {
          // out_d_ddy1 = d_ddout' * x'
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           true,
                           true,
                           x_conj,
                           true,
                           false,
                           out_d_ddy,
                           false);
        } else if (transpose_x) {
          // out_d_ddy1 = x * d_ddout
          CalcInputGrad<T>(dev_ctx,
                           x_conj,
                           false,
                           false,
                           d_ddout_mat,
                           false,
                           true,
                           out_d_ddy,
                           false);
        } else if (transpose_y) {
          // out_d_ddy1 = d_ddout' * x
          CalcInputGrad<T>(dev_ctx,
                           d_ddout_mat,
                           true,
                           true,
                           x_conj,
                           false,
                           true,
                           out_d_ddy,
                           false);
        } else {
          // out_d_ddy1 = x' * d_ddout
          CalcInputGrad<T>(dev_ctx,
                           x_conj,
                           true,
                           true,
                           d_ddout_mat,
                           false,
                           true,
                           out_d_ddy,
                           false);
        }
        d_ddy_flag = true;
      }
    }

    if (d_dy) {
      auto d_dy_mat = d_dy.get();
      if (d_dy_mat.dims() != y_help.dims()) {
        d_dy_mat.Resize(y_help.dims());
      }

      // compute d_dout1
      if (out_d_dout && ddx) {
        CalcInputGrad<T>(dev_ctx,
                         ddx_conj,
                         transpose_x,
                         true,
                         d_dy_mat,
                         transpose_y,
                         false,
                         out_d_dout,
                         d_dout_flag);
        d_dout_flag = true;
      }

      // compute d_ddx2
      if (out_d_ddx) {
        if (transpose_x && transpose_y) {
          // out_d_ddx2 = D_DY' * DOut'
          CalcInputGrad<T>(dev_ctx,
                           d_dy_mat,
                           true,
                           true,
                           dout_conj,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        } else if (transpose_x) {
          // out_d_ddx2 = D_DY * Dout'
          CalcInputGrad<T>(dev_ctx,
                           d_dy_mat,
                           false,
                           false,
                           dout_conj,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        } else if (transpose_y) {
          // out_d_ddx2 = Dout * D_DY
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           false,
                           false,
                           d_dy_mat,
                           false,
                           true,
                           out_d_ddx,
                           d_ddx_flag);
        } else {
          // out_d_ddx2 = Dout * D_DY'
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           false,
                           false,
                           d_dy_mat,
                           true,
                           false,
                           out_d_ddx,
                           d_ddx_flag);
        }
      }
    }

    if (d_dx) {
      auto d_dx_mat = d_dx.get();
      if (d_dx_mat.dims() != x_help.dims()) {
        d_dx_mat.Resize(x_help.dims());
      }

      // compute d_dout2
      if (out_d_dout && ddy) {
        CalcInputGrad<T>(dev_ctx,
                         d_dx_mat,
                         transpose_x,
                         true,
                         ddy_conj,
                         transpose_y,
                         false,
                         out_d_dout,
                         d_dout_flag);
      }

      // compute d_ddy2
      if (out_d_ddy) {
        if (transpose_x && transpose_y) {
          // out_d_ddy2 = dout' * d_dx'
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           true,
                           true,
                           d_dx_mat,
                           true,
                           false,
                           out_d_ddy,
                           d_ddy_flag);
        } else if (transpose_x) {
          // out_d_ddy2 = d_dx * dout
          CalcInputGrad<T>(dev_ctx,
                           d_dx_mat,
                           false,
                           false,
                           dout_conj,
                           false,
                           true,
                           out_d_ddy,
                           d_ddy_flag);
        } else if (transpose_y) {
          // out_d_ddy2 = dout' * d_dx
          CalcInputGrad<T>(dev_ctx,
                           dout_conj,
                           true,
                           true,
                           d_dx_mat,
                           false,
                           true,
                           out_d_ddy,
                           d_ddy_flag);
        } else {
          // out_d_ddy2 = d_dx' * dout
          CalcInputGrad<T>(dev_ctx,
                           d_dx_mat,
                           true,
                           true,
                           dout_conj,
                           false,
                           true,
                           out_d_ddy,
                           d_ddy_flag);
        }
      }
    }

    if (out_d_x) {
      if (out_dx_dims != x_help.dims()) {
        out_d_x->Resize(out_dx_dims);
      }
    }

    if (out_d_y) {
      if (out_dy_dims != y_help.dims()) {
        out_d_y->Resize(out_dy_dims);
      }
    }

    if (out_d_dout) {
      if (out_d_dout_dims != dout_help.dims()) {
        out_d_dout->Resize(out_d_dout_dims);
      }
    }

    if (out_d_ddx) {
      if (out_d_ddx_dims != x_help.dims()) {
        out_d_ddx->Resize(out_d_ddx_dims);
      }
    }

    if (out_d_ddy) {
      if (out_d_ddy_dims != y_help.dims()) {
        out_d_ddy->Resize(out_d_ddy_dims);
      }
    }
  } else {
    // Case3: broadcast. It need cost much time to reduce sum for the
    // broadcast and wastes the memory.
    // So we should avoid the case in reality.
    VLOG(3) << "========  MatMulV2TripleGradKernel, Compute ====== Case 3";
    VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
               "wastes the memory. So we should avoid the case in reality";

    DenseTensor out_dx_help;
    DenseTensor out_dy_help;
    DenseTensor out_d_ddx_help;
    DenseTensor out_d_ddy_help;

    if (out_d_dout) {
      if (ddx) {
        ddx_conj = Conj<T>(dev_ctx, ddx.get());
      }
      if (ddy) {
        ddy_conj = Conj<T>(dev_ctx, ddy.get());
      }
    }
    if (out_d_ddx || out_d_ddy) {
      x_conj = Conj<T>(dev_ctx, x);
      y_conj = Conj<T>(dev_ctx, y);
      dout_conj = Conj<T>(dev_ctx, dout);
    }

    if (transpose_x) {
      if (transpose_y) {
        // dX = ddY' d_ddout’, dY = d_ddout’ ddX'
        if (out_d_x && ddy && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     ddy_conj,
                                     d_ddout.get(),
                                     y_dims,
                                     dout_dims,
                                     &out_dx_help,
                                     true,
                                     true);
        if (out_d_y && ddx && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     ddx_conj,
                                     dout_dims,
                                     x_dims,
                                     &out_dy_help,
                                     true,
                                     true);
      } else {
        // dX = ddY d_ddout', dY = ddX d_ddout
        if (out_d_x && ddy && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     ddy_conj,
                                     d_ddout.get(),
                                     y_dims,
                                     dout_dims,
                                     &out_dx_help,
                                     false,
                                     true);
        if (out_d_y && ddx && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     ddx_conj,
                                     d_ddout.get(),
                                     x_dims,
                                     dout_dims,
                                     &out_dy_help,
                                     false,
                                     false);
      }

    } else {
      if (transpose_y) {
        // dX = d_ddout ddY, dY = d_ddout’ ddX
        if (out_d_x && ddy && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     ddy_conj,
                                     dout_dims,
                                     y_dims,
                                     &out_dx_help,
                                     false,
                                     false);
        if (out_d_y && ddx && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     ddx_conj,
                                     dout_dims,
                                     x_dims,
                                     &out_dy_help,
                                     true,
                                     false);
      } else {
        // dX = d_ddout ddY', dY = ddX' d_ddout
        if (out_d_x && ddy && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     ddy_conj,
                                     dout_dims,
                                     y_dims,
                                     &out_dx_help,
                                     false,
                                     true);
        if (out_d_y && ddx && d_ddout)
          MatMulFunction<Context, T>(dev_ctx,
                                     ddx_conj,
                                     d_ddout.get(),
                                     x_dims,
                                     dout_dims,
                                     &out_dy_help,
                                     true,
                                     false);
      }
    }

    // get help dims
    const std::vector<std::int64_t> dx_help_dims =
        vectorize(out_dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims =
        vectorize(out_dx_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(
        dx_broadcast_dims.data(), dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(
        dy_broadcast_dims.data(), dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(),
              x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(),
              y_dims.data() + y_ndim,
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

    // Reduce sum to get grad by ReduceSum
    if (out_d_x) {
      if (dx_reduce_dims.empty()) {
        *out_d_x = std::move(out_dx_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, out_dx_help, out_d_x, dx_reduce_dims);
      }
      out_d_x->Resize(x.dims());
    }

    if (out_d_y) {
      if (dy_reduce_dims.empty()) {
        *out_d_y = std::move(out_dy_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, out_dy_help, out_d_y, dy_reduce_dims);
      }
      out_d_y->Resize(y.dims());
    }

    // compute d_dout
    if (out_d_dout) {
      if (d_dx && ddy) {
        MatMulFunction<Context, T>(dev_ctx,
                                   d_dx.get(),
                                   ddy_conj,
                                   x_dims,
                                   y_dims,
                                   out_d_dout,
                                   transpose_x,
                                   transpose_y);
      }
      if (d_dy && ddx) {
        MatMulFunction<Context, T>(dev_ctx,
                                   ddx_conj,
                                   d_dy.get(),
                                   x_dims,
                                   y_dims,
                                   out_d_dout,
                                   transpose_x,
                                   transpose_y,
                                   true);
      }
    }

    // compute d_ddx
    if (out_d_ddx) {
      if (transpose_x && transpose_y) {
        // out_d_ddx1 = y' * d_ddout'
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     y_conj,
                                     d_ddout.get(),
                                     y_dims,
                                     dout_dims,
                                     &out_d_ddx_help,
                                     true,
                                     true);
        }

        // out_d_ddx2 = D_DY' * DOut'
        if (d_dy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_dy.get(),
                                     dout_conj,
                                     y_dims,
                                     dout_dims,
                                     &out_d_ddx_help,
                                     true,
                                     true,
                                     true);
        }

      } else if (transpose_x) {
        // out_d_ddx1 = y * d_ddout'
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     y_conj,
                                     d_ddout.get(),
                                     y_dims,
                                     dout_dims,
                                     &out_d_ddx_help,
                                     false,
                                     true);
        }

        // out_d_ddx2 = D_DY * Dout'
        if (d_dy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_dy.get(),
                                     dout_conj,
                                     y_dims,
                                     dout_dims,
                                     &out_d_ddx_help,
                                     false,
                                     true,
                                     true);
        }

      } else if (transpose_y) {
        // out_d_ddx1 = d_ddout * y
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     y_conj,
                                     dout_dims,
                                     y_dims,
                                     &out_d_ddx_help,
                                     false,
                                     false);
        }

        // out_d_ddx2 = Dout * D_DY
        if (d_dy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     d_dy.get(),
                                     dout_dims,
                                     y_dims,
                                     &out_d_ddx_help,
                                     false,
                                     false,
                                     true);
        }
      } else {
        // out_d_ddx1 = d_ddout * y'
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     y_conj,
                                     dout_dims,
                                     y_dims,
                                     &out_d_ddx_help,
                                     false,
                                     true);
        }

        // out_d_ddx2 = Dout * D_DY'
        if (d_dy) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     d_dy.get(),
                                     dout_dims,
                                     y_dims,
                                     &out_d_ddx_help,
                                     false,
                                     true,
                                     true);
        }
      }

      if (dx_reduce_dims.empty()) {
        *out_d_ddx = std::move(out_d_ddx_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, out_d_ddx_help, out_d_ddx, dx_reduce_dims);
      }
      out_d_ddx->Resize(x.dims());
    }

    // compute d_ddy
    if (out_d_ddy) {
      if (transpose_x && transpose_y) {
        // out_d_ddy1 = d_ddout' * x'
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     x_conj,
                                     dout_dims,
                                     x_dims,
                                     &out_d_ddy_help,
                                     true,
                                     true);
        }

        // out_d_ddy2 = dout' * d_dx'
        if (d_dx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     d_dx.get(),
                                     dout_dims,
                                     x_dims,
                                     &out_d_ddy_help,
                                     true,
                                     true,
                                     true);
        }

      } else if (transpose_x) {
        // out_d_ddy1 = x * d_ddout
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     x_conj,
                                     d_ddout.get(),
                                     x_dims,
                                     dout_dims,
                                     &out_d_ddy_help,
                                     false,
                                     false);
        }

        // out_d_ddy2 = d_dx * dout
        if (d_dx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_dx.get(),
                                     dout_conj,
                                     x_dims,
                                     dout_dims,
                                     &out_d_ddy_help,
                                     false,
                                     false,
                                     true);
        }

      } else if (transpose_y) {
        // out_d_ddy1 = d_ddout' * x
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_ddout.get(),
                                     x_conj,
                                     dout_dims,
                                     x_dims,
                                     &out_d_ddy_help,
                                     true,
                                     false);
        }

        // out_d_ddy2 = dout' * d_dx
        if (d_dx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     dout_conj,
                                     d_dx.get(),
                                     dout_dims,
                                     x_dims,
                                     &out_d_ddy_help,
                                     true,
                                     false,
                                     true);
        }

      } else {
        // out_d_ddy1 = x' * d_ddout
        if (d_ddout) {
          MatMulFunction<Context, T>(dev_ctx,
                                     x_conj,
                                     d_ddout.get(),
                                     x_dims,
                                     dout_dims,
                                     &out_d_ddy_help,
                                     true,
                                     false);
        }

        // out_d_ddy2 = d_dx' * dout
        if (d_dx) {
          MatMulFunction<Context, T>(dev_ctx,
                                     d_dx.get(),
                                     dout_conj,
                                     x_dims,
                                     dout_dims,
                                     &out_d_ddy_help,
                                     true,
                                     false,
                                     true);
        }
      }

      if (dy_reduce_dims.empty()) {
        *out_d_ddy = std::move(out_d_ddy_help);
      } else {
        ReduceSumForMatmulGrad<Context, T>()(
            dev_ctx, out_d_ddy_help, out_d_ddy, dy_reduce_dims);
      }
      out_d_ddy->Resize(y.dims());
    }
  }
}

template <typename T, typename Context>
void MatmulWithFlattenGradKernel(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 const DenseTensor& out_grad,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor* x_grad,
                                 DenseTensor* y_grad) {
  auto x_matrix = x.dims().size() > 2
                      ? paddle::framework::ReshapeToMatrix(x, x_num_col_dims)
                      : x;
  auto y_matrix = y.dims().size() > 2
                      ? paddle::framework::ReshapeToMatrix(y, y_num_col_dims)
                      : y;
  auto* dout = &out_grad;

  DenseTensor dout_mat(*dout);
  dout_mat.Resize({phi::flatten_to_2d(x.dims(), x_num_col_dims)[0],
                   phi::flatten_to_2d(y.dims(), y_num_col_dims)[1]});

  auto* dx = x_grad;
  auto* dy = y_grad;

  if (dx != nullptr) {
    dx->set_lod(x.lod());
  }
  if (dy != nullptr) {
    dy->set_lod(y.lod());
  }

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    DenseTensor dx_matrix =
        dx->dims().size() > 2
            ? paddle::framework::ReshapeToMatrix(*dx, x_num_col_dims)
            : *dx;

    // dx = dout * y'. dx: M x K, dout : M x N, y : K x N
    blas.MatMul(dout_mat, false, y_matrix, true, &dx_matrix);
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    DenseTensor dy_matrix =
        dy->dims().size() > 2
            ? paddle::framework::ReshapeToMatrix(*dy, y_num_col_dims)
            : *dy;
    // dy = x' * dout. dy K x N, dout : M x N, x : M x K
    blas.MatMul(x_matrix, true, dout_mat, false, &dy_matrix);
  }
}

template <typename T, typename Context>
void MatmulWithFlattenDoubleGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    const DenseTensor& out_grad,
    const paddle::optional<DenseTensor>& x_grad_grad,
    const paddle::optional<DenseTensor>& y_grad_grad,
    int x_num_col_dims,
    int y_num_col_dims,
    DenseTensor* x_grad,
    DenseTensor* y_grad,
    DenseTensor* out_grad_grad) {
  auto x_mat = x.dims().size() > 2
                   ? paddle::framework::ReshapeToMatrix(x, x_num_col_dims)
                   : x;
  auto y_mat = y.dims().size() > 2
                   ? paddle::framework::ReshapeToMatrix(y, y_num_col_dims)
                   : y;

  const int m = phi::flatten_to_2d(x.dims(), x_num_col_dims)[0];
  const int n = phi::flatten_to_2d(y.dims(), y_num_col_dims)[1];

  auto* dout = &out_grad;
  DenseTensor dout_mat(*dout);
  dout_mat.Resize({m, n});

  auto* ddx = x_grad_grad.get_ptr();
  auto* ddy = y_grad_grad.get_ptr();

  auto* dx = x_grad;
  auto* dy = y_grad;
  auto* ddout = out_grad_grad;

  DenseTensor ddout_mat;
  if (ddout) {
    ddout->set_lod(dout->lod());
    // allocate and reshape ddout
    dev_ctx.template Alloc<T>(ddout);
    ddout_mat.ShareDataWith(*ddout);
    ddout_mat.Resize({m, n});
  }

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  // a flag to specify whether ddout value has been set, if flag
  // is false, MatMul beta should be 0 to set ddout, if flag is
  // true, MatMul beta should be 1 to add result to ddout.
  bool ddout_flag = false;
  if (ddx) {
    auto ddx_mat =
        ddx->dims().size() > 2
            ? paddle::framework::ReshapeToMatrix(*ddx, x_num_col_dims)
            : static_cast<const DenseTensor&>(*ddx);

    // dy = ddx' * dout. dy : K x M, ddx' : K x M, dout : M x N
    if (dy) {
      dy->set_lod(y.lod());
      // allocate and reshape dy
      dev_ctx.template Alloc<T>(dy);
      DenseTensor dy_mat =
          dy->dims().size() > 2
              ? paddle::framework::ReshapeToMatrix(*dy, y_num_col_dims)
              : *dy;
      blas.MatMul(ddx_mat, true, dout_mat, false, &dy_mat);
    }
    // ddout1 = ddx * y. ddx : M x K, y : K x N, ddout1 : M x N
    if (ddout) {
      blas.MatMul(ddx_mat,
                  false,
                  y_mat,
                  false,
                  static_cast<T>(1.0),
                  &ddout_mat,
                  static_cast<T>(ddout_flag));
      ddout_flag = true;
    }
  }
  if (ddy) {
    auto ddy_mat =
        ddy->dims().size() > 2
            ? paddle::framework::ReshapeToMatrix(*ddy, y_num_col_dims)
            : static_cast<const DenseTensor&>(*ddy);
    // dx = dout * ddy'. dout : M x N, ddy' : N x K, dx : M x K
    if (dx) {
      dx->set_lod(x.lod());
      // allocate and reshape dx
      dev_ctx.template Alloc<T>(dx);
      DenseTensor dx_mat =
          dx->dims().size() > 2
              ? paddle::framework::ReshapeToMatrix(*dx, x_num_col_dims)
              : *dx;
      blas.MatMul(dout_mat, false, ddy_mat, true, &dx_mat);
    }
    // ddout2 = x * ddy. x : M x K, ddy : K x N, ddout2 : M x N
    if (ddout) {
      blas.MatMul(x_mat,
                  false,
                  ddy_mat,
                  false,
                  static_cast<T>(1.0),
                  &ddout_mat,
                  static_cast<T>(ddout_flag));
    }
  }
}

}  // namespace phi
