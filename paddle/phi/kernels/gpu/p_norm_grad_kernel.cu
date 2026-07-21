// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/p_norm_grad_kernel.h"

#include <vector>

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/reduce_grad_functions.h"
#include "paddle/phi/kernels/reduce_amax_grad_kernel.h"
#include "paddle/phi/kernels/sign_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

// Helper device function to compute pow with same special cases as PowKernel
template <typename MT>
__device__ __forceinline__ MT compute_pow_like_kernel(MT val, double exponent) {
  if (exponent == 0.5) {
    return sqrt(val);
  } else if (exponent == -0.5) {
    return rsqrt(val);
  } else if (exponent == -1.0) {
    return static_cast<MT>(1) / val;
  } else if (exponent == -2.0) {
    return static_cast<MT>(1) / (val * val);
  } else if (exponent == 0.0) {
    return static_cast<MT>(1);
  } else if (exponent == 1.0) {
    return val;
  } else if (exponent == 2.0) {
    return val * val;
  } else if (exponent == 3.0) {
    return val * val * val;
  } else {
    return pow(val, static_cast<MT>(exponent));
  }
}

// Round an intermediate value (kept in compute type MT) back to the storage
// dtype T and return it as MT again. This reproduces the rounding point of a
// single torch elementwise kernel, whose output tensor is dtype T. For
// fp32/fp64 (MT == T) this is the identity; only fp16/bf16 are affected.
template <typename T, typename MT>
__device__ __forceinline__ MT RoundToStorage(MT v) {
  return static_cast<MT>(static_cast<T>(v));
}

// Torch-compatible pow: operates in storage type T (like a standalone torch
// elementwise kernel whose output tensor is dtype T). The exponent is first
// narrowed to T then promoted back to MT before calling pow(), matching the
// truncation behavior of torch's type promotion for scalar exponents.
template <typename T, typename MT>
__device__ __forceinline__ T
compute_pow_like_kernel_torch_compat(T val, double exponent) {
  MT val_MT = static_cast<MT>(val);
  if (exponent == 0.5) {
    return static_cast<T>(sqrt(val_MT));
  } else if (exponent == -0.5) {
    return static_cast<T>(rsqrt(val_MT));
  } else if (exponent == -1.0) {
    return static_cast<T>(static_cast<MT>(1) / val_MT);
  } else if (exponent == -2.0) {
    return static_cast<T>(static_cast<MT>(1) /
                          RoundToStorage<T, MT>(val_MT * val_MT));
  } else if (exponent == 0.0) {
    return static_cast<T>(1);
  } else if (exponent == 1.0) {
    return val;
  } else if (exponent == 2.0) {
    return static_cast<T>(val_MT * val_MT);
  } else if (exponent == 3.0) {
    return static_cast<T>(RoundToStorage<T, MT>(val_MT * val_MT) * val_MT);
  } else {
    MT exponent_MT = static_cast<MT>(static_cast<T>(exponent));
    return static_cast<T>(pow(val_MT, exponent_MT));
  }
}

// Fused CUDA kernel for p=2 norm gradient
// dx = grad * (x / norm).masked_fill_(norm == 0, 0)
template <typename T, bool Compat = false>
__global__ void PNormGradP2Kernel(const T* x,
                                  const T* norm,
                                  const T* grad,
                                  T* dx,
                                  int64_t pre,
                                  int64_t axis_size,
                                  int64_t post,
                                  int64_t total,
                                  bool reduce_all) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  CUDA_KERNEL_LOOP_TYPE(idx, total, int64_t) {
    int64_t norm_idx;
    if (reduce_all) {
      norm_idx = 0;
    } else {
      int64_t post_idx = idx % post;
      int64_t pre_idx = idx / (axis_size * post);
      norm_idx = pre_idx * post + post_idx;
    }

    MT norm_val = static_cast<MT>(norm[norm_idx]);
    if (norm_val == static_cast<MT>(0)) {
      dx[idx] = static_cast<T>(0);
    } else {
      MT x_val = static_cast<MT>(x[idx]);
      MT grad_val = static_cast<MT>(grad[norm_idx]);
      if constexpr (Compat) {
        // Match torch's two separate elementwise kernels (norm_backward p==2:
        // `grad * (self / norm)`): each op stores back to dtype T, so for
        // fp16/bf16 there are two rounding points. Round `self / norm` to T
        // before the multiply to stay bitwise-aligned with torch.
        MT x_div_norm = RoundToStorage<T>(x_val / norm_val);
        dx[idx] = static_cast<T>(grad_val * x_div_norm);
      } else {
        MT x_div_norm = x_val / norm_val;
        dx[idx] = static_cast<T>(x_div_norm * grad_val);
      }
    }
  }
}

// Fused CUDA kernel for p < 1 norm gradient
// dx = sign(x) * |x|^(p-1) * grad * norm^(1-p), masked_fill(x == 0, 0)
template <typename T, bool Compat = false>
__global__ void PNormGradPLessThan1Kernel(const T* x,
                                          const T* norm,
                                          const T* grad,
                                          T* dx,
                                          int64_t pre,
                                          int64_t axis_size,
                                          int64_t post,
                                          int64_t total,
                                          bool reduce_all,
                                          double porder) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  double p_minus_1 = porder - 1.0;
  double one_minus_p = 1.0 - porder;

  CUDA_KERNEL_LOOP_TYPE(idx, total, int64_t) {
    MT x_val = static_cast<MT>(x[idx]);

    // masked_fill: when x == 0, dx = 0
    if (x_val == static_cast<MT>(0)) {
      dx[idx] = static_cast<T>(0);
    } else {
      // Calculate norm/grad index
      int64_t norm_idx;
      if (reduce_all) {
        norm_idx = 0;
      } else {
        int64_t post_idx = idx % post;
        int64_t pre_idx = idx / (axis_size * post);
        norm_idx = pre_idx * post + post_idx;
      }

      MT norm_val = static_cast<MT>(norm[norm_idx]);
      MT grad_val = static_cast<MT>(grad[norm_idx]);

      // abs(x)
      MT abs_x = (x_val > static_cast<MT>(0)) ? x_val : -x_val;

      // sign(x): 1 if x > 0, -1 if x < 0 (x != 0 already checked)
      MT sign_x = (x_val > static_cast<MT>(0)) ? static_cast<MT>(1)
                                               : static_cast<MT>(-1);

      // |x|^(p-1)
      MT abs_pow;
      MT self_scaled;
      MT temp1;
      // norm^(1-p)
      MT norm_pow;
      if constexpr (Compat) {
        abs_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(abs_x), p_minus_1));
        self_scaled = RoundToStorage<T>(sign_x * abs_pow);
        temp1 = RoundToStorage<T>(self_scaled * grad_val);
        norm_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(norm_val), one_minus_p));
      } else {
        abs_pow = compute_pow_like_kernel(abs_x, p_minus_1);
        self_scaled = sign_x * abs_pow;
        temp1 = self_scaled * grad_val;
        norm_pow = compute_pow_like_kernel(norm_val, one_minus_p);
      }

      dx[idx] = static_cast<T>(temp1 * norm_pow);
    }
  }
}

// Fused CUDA kernel for p=1 norm gradient
// dx = sign(x) * grad (with broadcast)
template <typename T>
__global__ void PNormGradP1Kernel(const T* x,
                                  const T* grad,
                                  T* dx,
                                  int64_t pre,
                                  int64_t axis_size,
                                  int64_t post,
                                  int64_t total,
                                  bool reduce_all) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  CUDA_KERNEL_LOOP_TYPE(idx, total, int64_t) {
    MT x_val = static_cast<MT>(x[idx]);

    // Calculate norm/grad index for broadcasting
    int64_t grad_idx;
    if (reduce_all) {
      grad_idx = 0;
    } else {
      int64_t post_idx = idx % post;
      int64_t pre_idx = idx / (axis_size * post);
      grad_idx = pre_idx * post + post_idx;
    }

    MT grad_val = static_cast<MT>(grad[grad_idx]);

    // sign(x) * grad
    MT sign_x;
    if (x_val > static_cast<MT>(0)) {
      sign_x = static_cast<MT>(1);
    } else if (x_val < static_cast<MT>(0)) {
      sign_x = static_cast<MT>(-1);
    } else {
      sign_x = static_cast<MT>(0);
    }

    dx[idx] = static_cast<T>(sign_x * grad_val);
  }
}

// Fused CUDA kernel for 1 < p < 2 norm gradient
// dx = sign(x) * |x|^(p-1) * grad / norm^(p-1), masked_fill(norm==0, 0)
template <typename T, bool Compat = false>
__global__ void PNormGradPBetween1And2Kernel(const T* x,
                                             const T* norm,
                                             const T* grad,
                                             T* dx,
                                             int64_t pre,
                                             int64_t axis_size,
                                             int64_t post,
                                             int64_t total,
                                             bool reduce_all,
                                             double porder) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  double p_minus_1 = porder - 1.0;

  CUDA_KERNEL_LOOP_TYPE(idx, total, int64_t) {
    // Calculate norm/grad index for broadcasting
    int64_t norm_idx;
    if (reduce_all) {
      norm_idx = 0;
    } else {
      int64_t post_idx = idx % post;
      int64_t pre_idx = idx / (axis_size * post);
      norm_idx = pre_idx * post + post_idx;
    }

    MT norm_val = static_cast<MT>(norm[norm_idx]);

    // masked_fill: when norm == 0, dx = 0
    if (norm_val == static_cast<MT>(0)) {
      dx[idx] = static_cast<T>(0);
    } else {
      MT x_val = static_cast<MT>(x[idx]);
      MT grad_val = static_cast<MT>(grad[norm_idx]);

      // abs(x)
      MT abs_x = (x_val > static_cast<MT>(0)) ? x_val : -x_val;

      // sign(x)
      MT sign_x;
      if (x_val > static_cast<MT>(0)) {
        sign_x = static_cast<MT>(1);
      } else if (x_val < static_cast<MT>(0)) {
        sign_x = static_cast<MT>(-1);
      } else {
        sign_x = static_cast<MT>(0);
      }

      // |x|^(p-1)
      MT abs_pow;
      MT self_scaled;
      // norm^(p-1)
      MT norm_pow;
      // scale_v = grad / norm_pow
      MT scale_v;

      if constexpr (Compat) {
        abs_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(abs_x), p_minus_1));
        self_scaled = RoundToStorage<T>(sign_x * abs_pow);
        norm_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(norm_val), p_minus_1));
        scale_v = RoundToStorage<T>(grad_val / norm_pow);
      } else {
        abs_pow = compute_pow_like_kernel(abs_x, p_minus_1);
        self_scaled = sign_x * abs_pow;
        norm_pow = compute_pow_like_kernel(norm_val, p_minus_1);
        scale_v = grad_val / norm_pow;
      }

      dx[idx] = static_cast<T>(self_scaled * scale_v);
    }
  }
}

// Fused CUDA kernel for p > 2 norm gradient
// dx = x * |x|^(p-2) * grad / norm^(p-1), masked_fill(norm==0, 0)
template <typename T, bool Compat = false>
__global__ void PNormGradPGreaterThan2Kernel(const T* x,
                                             const T* norm,
                                             const T* grad,
                                             T* dx,
                                             int64_t pre,
                                             int64_t axis_size,
                                             int64_t post,
                                             int64_t total,
                                             bool reduce_all,
                                             double porder) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  double p_minus_2 = porder - 2.0;
  double p_minus_1 = porder - 1.0;

  CUDA_KERNEL_LOOP_TYPE(idx, total, int64_t) {
    // Calculate norm/grad index for broadcasting
    int64_t norm_idx;
    if (reduce_all) {
      norm_idx = 0;
    } else {
      int64_t post_idx = idx % post;
      int64_t pre_idx = idx / (axis_size * post);
      norm_idx = pre_idx * post + post_idx;
    }

    MT norm_val = static_cast<MT>(norm[norm_idx]);

    // masked_fill: when norm == 0, dx = 0
    if (norm_val == static_cast<MT>(0)) {
      dx[idx] = static_cast<T>(0);
    } else {
      MT x_val = static_cast<MT>(x[idx]);
      MT grad_val = static_cast<MT>(grad[norm_idx]);

      // abs(x)
      MT abs_x = (x_val > static_cast<MT>(0)) ? x_val : -x_val;

      // |x|^(p-2)
      MT abs_pow;
      // self_scaled = x * |x|^(p-2)
      MT self_scaled;
      // norm^(p-1)
      MT norm_pow;
      // scale_v = grad / norm_pow
      MT scale_v;
      if constexpr (Compat) {
        abs_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(abs_x), p_minus_2));
        self_scaled = RoundToStorage<T>(x_val * abs_pow);
        norm_pow = static_cast<MT>(compute_pow_like_kernel_torch_compat<T, MT>(
            static_cast<T>(norm_val), p_minus_1));
        scale_v = RoundToStorage<T>(grad_val / norm_pow);
      } else {
        abs_pow = compute_pow_like_kernel(abs_x, p_minus_2);
        self_scaled = x_val * abs_pow;
        norm_pow = compute_pow_like_kernel(norm_val, p_minus_1);
        scale_v = grad_val / norm_pow;
      }

      dx[idx] = static_cast<T>(self_scaled * scale_v);
    }
  }
}

// Helper function to compute pre, axis_size, post for broadcasting
inline void GetPreAxisPost(const DDim& xdim,
                           int axis,
                           bool reduce_all,
                           int64_t* pre,
                           int64_t* axis_size,
                           int64_t* post) {
  *pre = 1;
  *axis_size = 1;
  *post = 1;
  if (reduce_all) {
    *axis_size = product(xdim);
  } else {
    for (int i = 0; i < axis; ++i) {
      *pre *= xdim[i];
    }
    *axis_size = xdim[axis];
    for (int i = axis + 1; i < xdim.size(); ++i) {
      *post *= xdim[i];
    }
  }
}

template <typename T>
struct PNormGradFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  HOSTDEVICE explicit inline PNormGradFunctor(float porder, float eps) {
    this->porder = static_cast<MT>(porder - 1.0f);
    this->eps = static_cast<MT>(eps);
  }

  template <typename Context,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const Context& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    auto unstable_term =
        (*x).abs().template cast<MT>().pow(this->porder).template cast<T>();
    auto mask = (*x) == x->constant(static_cast<T>(0));
    auto stable_term =
        mask.select(x->constant(static_cast<T>(0)), unstable_term);
    auto self_scaled = (*x).sign() * stable_term;
    auto norm_term =
        (*y).template cast<MT>().pow(-this->porder).template cast<T>();
    dx->device(place) =
        self_scaled * dy->broadcast(dim) * norm_term.broadcast(dim);
  }

  MT porder;
  MT eps;
};

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& out,
                     const DenseTensor& out_grad,
                     double porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     DenseTensor* x_grad) {
  auto* in_x = &x;
  auto* in_norm = &out;
  auto* in_norm_dy = &out_grad;
  auto* out_dx = x_grad;
  dev_ctx.template Alloc<T>(out_dx);

  if (out_dx->numel() == 0) {
    return;
  }

  auto xdim = in_x->dims();
  bool reduce_all = (in_norm->numel() == 1);
  if (axis < 0) {
    axis = xdim.size() + axis;
  }
  const std::vector<int> dims = {axis};

  if (porder == 0) {
    funcs::SetConstant<Context, T> set_zero;
    set_zero(dev_ctx, out_dx, static_cast<T>(0));
  } else if (porder == INFINITY || porder == -INFINITY) {
    std::vector<int64_t> dims_for_amax;
    if (reduce_all) {
      dims_for_amax.resize(xdim.size());
      for (int i = 0; i < xdim.size(); ++i) dims_for_amax[i] = i;
    } else {
      dims_for_amax.push_back(axis);
    }

    DenseTensor x_abs;
    x_abs.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&x_abs);
    phi::AbsKernel<T, Context>(dev_ctx, *in_x, &x_abs);

    DenseTensor amax_grad_out;
    amax_grad_out.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&amax_grad_out);
    phi::ReduceAMaxGradKernel<T, Context>(dev_ctx,
                                          x_abs,
                                          *in_norm,
                                          *in_norm_dy,
                                          dims_for_amax,
                                          keepdim,
                                          reduce_all,
                                          &amax_grad_out);
    DenseTensor x_sign;
    x_sign.Resize(in_x->dims());
    dev_ctx.template Alloc<T>(&x_sign);
    phi::SignKernel<T, Context>(dev_ctx, *in_x, &x_sign);
    phi::MultiplyKernel<T, Context>(dev_ctx, amax_grad_out, x_sign, out_dx);
  } else if (porder == 1) {
    // Fused kernel: dx = sign(x) * grad (with broadcast)
    int64_t pre, axis_size, post;
    GetPreAxisPost(xdim, axis, reduce_all, &pre, &axis_size, &post);

    int64_t total = in_x->numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);

    PNormGradP1Kernel<T><<<config.block_per_grid,
                           config.thread_per_block,
                           0,
                           dev_ctx.stream()>>>(in_x->data<T>(),
                                               in_norm_dy->data<T>(),
                                               out_dx->data<T>(),
                                               pre,
                                               axis_size,
                                               post,
                                               total,
                                               reduce_all);
  } else if (porder == 2) {
    // Fused kernel: dx = grad * (x / norm).masked_fill_(norm == 0, 0)
    int64_t pre, axis_size, post;
    GetPreAxisPost(xdim, axis, reduce_all, &pre, &axis_size, &post);

    int64_t total = in_x->numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);

    if (FLAGS_use_accuracy_compatible_kernel) {
      PNormGradP2Kernel<T, true><<<config.block_per_grid,
                                   config.thread_per_block,
                                   0,
                                   dev_ctx.stream()>>>(in_x->data<T>(),
                                                       in_norm->data<T>(),
                                                       in_norm_dy->data<T>(),
                                                       out_dx->data<T>(),
                                                       pre,
                                                       axis_size,
                                                       post,
                                                       total,
                                                       reduce_all);
    } else {
      PNormGradP2Kernel<T, false><<<config.block_per_grid,
                                    config.thread_per_block,
                                    0,
                                    dev_ctx.stream()>>>(in_x->data<T>(),
                                                        in_norm->data<T>(),
                                                        in_norm_dy->data<T>(),
                                                        out_dx->data<T>(),
                                                        pre,
                                                        axis_size,
                                                        post,
                                                        total,
                                                        reduce_all);
    }
  } else if (porder < 1.0) {
    // Fused kernel: dx = sign(x) * |x|^(p-1) * grad * norm^(1-p)
    // masked_fill(x == 0, 0)
    int64_t pre, axis_size, post;
    GetPreAxisPost(xdim, axis, reduce_all, &pre, &axis_size, &post);

    int64_t total = in_x->numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);

    if (FLAGS_use_accuracy_compatible_kernel) {
      PNormGradPLessThan1Kernel<T, true>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    } else {
      PNormGradPLessThan1Kernel<T, false>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    }
  } else if (porder < 2.0) {
    // Fused kernel: dx = sign(x) * |x|^(p-1) * grad / norm^(p-1),
    // masked_fill(norm==0, 0)
    int64_t pre, axis_size, post;
    GetPreAxisPost(xdim, axis, reduce_all, &pre, &axis_size, &post);

    int64_t total = in_x->numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);

    if (FLAGS_use_accuracy_compatible_kernel) {
      PNormGradPBetween1And2Kernel<T, true>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    } else {
      PNormGradPBetween1And2Kernel<T, false>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    }
  } else {
    // Fused kernel: dx = x * |x|^(p-2) * grad / norm^(p-1),
    // masked_fill(norm==0, 0)
    int64_t pre, axis_size, post;
    GetPreAxisPost(xdim, axis, reduce_all, &pre, &axis_size, &post);

    int64_t total = in_x->numel();
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);

    if (FLAGS_use_accuracy_compatible_kernel) {
      PNormGradPGreaterThan2Kernel<T, true>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    } else {
      PNormGradPGreaterThan2Kernel<T, false>
          <<<config.block_per_grid,
             config.thread_per_block,
             0,
             dev_ctx.stream()>>>(in_x->data<T>(),
                                 in_norm->data<T>(),
                                 in_norm_dy->data<T>(),
                                 out_dx->data<T>(),
                                 pre,
                                 axis_size,
                                 post,
                                 total,
                                 reduce_all,
                                 porder);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(p_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::PNormGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
