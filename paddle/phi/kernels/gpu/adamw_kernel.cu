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

#include "paddle/phi/kernels/adamw_kernel.h"

#include <math.h>  // for sqrt in CPU and CUDA
#include <cmath>
#include <cstdlib>

#include <string>
#include <vector>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

namespace phi {

// Template accessor design
template <typename MT, bool IsCpu>
struct BetaPowAccessor;

template <typename MT>
struct BetaPowAccessor<MT, true> {  // CPU accessor
  const MT beta1;
  const MT beta2;

  BetaPowAccessor(const MT* beta1_pow, const MT* beta2_pow)
      : beta1(*beta1_pow), beta2(*beta2_pow) {}

  __device__ MT GetBeta1() const { return beta1; }
  __device__ MT GetBeta2() const { return beta2; }
};

template <typename MT>
struct BetaPowAccessor<MT, false> {  // GPU pointer
  const MT* beta1_pow;
  const MT* beta2_pow;

  BetaPowAccessor(const MT* beta1, const MT* beta2)
      : beta1_pow(beta1), beta2_pow(beta2) {}

  __device__ MT GetBeta1() const { return *beta1_pow; }
  __device__ MT GetBeta2() const { return *beta2_pow; }
};

// Unified kernel template
template <typename T,   // Parameter type
          typename TG,  // Gradient type
          typename MT,  // Multi-precision type
          typename TM,  // Moment estimation type
          typename BetaAccessor>
__global__ void AdamWKernel(MT beta1,
                            MT beta2,
                            MT epsilon,
                            MT coeff,
                            MT lr_ratio,
                            const MT* lr_,
                            const TG* grad,
                            const T* param,
                            T* param_out,
                            const MT* master_param,
                            MT* master_param_out,
                            const TM* moment1,
                            TM* moment1_out,
                            const TM* moment2,
                            TM* moment2_out,
                            const TM* moment2_max,
                            TM* moment2_max_out,
                            BetaAccessor beta_accessor,
                            int64_t ndim,
                            bool amsgrad) {
  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  MT lr = *lr_ * lr_ratio;
  // Get beta powers
  MT beta1_pow = beta_accessor.GetBeta1();
  MT beta2_pow = beta_accessor.GetBeta2();

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);

    p *= (static_cast<MT>(1.0) - lr * coeff);

    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    MT denom;
    if (amsgrad) {
      MT mom2_max = static_cast<MT>(moment2_max[id]);
      MT mom2_max_ = std::max(mom2, mom2_max);
      moment2_max_out[id] = mom2_max_;

      denom =
          (sqrt(mom2_max_) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    } else {
      denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    }

    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));

    moment1_out[id] = mom1;
    moment2_out[id] = mom2;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

// Beta power update kernel
template <typename MT>
__global__ void UpdateBetaPowKernel(MT beta1,
                                    MT beta2,
                                    const MT* beta1_pow,
                                    const MT* beta2_pow,
                                    MT* beta1_pow_out,
                                    MT* beta2_pow_out) {
  beta1_pow_out[0] = beta1 * beta1_pow[0];
  beta2_pow_out[0] = beta2 * beta2_pow[0];
}

template <typename T, typename Context>
PADDLE_API void AdamwDenseKernel_old(const Context& dev_ctx,
                                     const DenseTensor& param,
                                     const DenseTensor& grad,
                                     const DenseTensor& learning_rate,
                                     const DenseTensor& moment1,
                                     const DenseTensor& moment2,
                                     const optional<DenseTensor>& moment2_max,
                                     const DenseTensor& beta1_pow,
                                     const DenseTensor& beta2_pow,
                                     const optional<DenseTensor>& master_param,
                                     const optional<DenseTensor>& skip_update,
                                     const Scalar& beta1,
                                     const Scalar& beta2,
                                     const Scalar& epsilon,
                                     float lr_ratio,
                                     float coeff,
                                     bool with_decay,
                                     bool lazy_mode,
                                     int64_t min_row_size_to_use_multithread,
                                     bool multi_precision,
                                     bool use_global_beta_pow,
                                     bool amsgrad,
                                     DenseTensor* param_out,
                                     DenseTensor* moment1_out,
                                     DenseTensor* moment2_out,
                                     DenseTensor* moment2_max_out,
                                     DenseTensor* beta1_pow_out,
                                     DenseTensor* beta2_pow_out,
                                     DenseTensor* master_param_outs) {
  using MT = typename MPTypeTrait<T>::Type;
  MT coeff_ = static_cast<MT>(coeff);
  MT lr_ratio_ = static_cast<MT>(lr_ratio);

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  // skip_update=true, just copy input to output, and TensorCopy will call
  // mutable_data
  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (amsgrad) {
      Copy(dev_ctx,
           moment2_max.get(),
           dev_ctx.GetPlace(),
           false,
           moment2_max_out);
    }
    if (!use_global_beta_pow) {
      Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  // if with_decay = false, coeff = 0
  if (!with_decay) {
    coeff_ = static_cast<MT>(0.0);
  }

  MT beta1_ = beta1.to<MT>();
  MT beta2_ = beta2.to<MT>();
  MT epsilon_ = epsilon.to<MT>();
  VLOG(3) << "beta1_pow.numel() : " << beta1_pow.numel()
          << "beta2_pow.numel() : " << beta2_pow.numel();
  VLOG(3) << "param.numel(): " << param.numel();
  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));

  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MT>(master_param_outs) : nullptr;

  const MT* moment2_max_in_data =
      amsgrad ? moment2_max.get().data<MT>() : nullptr;
  MT* moment2_max_out_data =
      amsgrad ? dev_ctx.template Alloc<MT>(moment2_max_out) : nullptr;

  // update param and moment
  int threads = 512;
  int64_t blocks_max = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int blocks = std::min((param.numel() + threads - 1) / threads, blocks_max);

  // Determine BetaPow location
  const bool beta_pow_on_cpu =
      beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace();

  // Determine gradient type
  const bool use_bfloat32_grad = grad.dtype() == DataType::FLOAT32;
  // Determine moment type
  const bool use_bfloat16_moments = moment1.dtype() == DataType::BFLOAT16 &&
                                    moment2.dtype() == DataType::BFLOAT16;

#define LAUNCH_ADAMW_KERNEL(MOMENT_T)                                     \
  if (beta_pow_on_cpu) {                                                  \
    BetaPowAccessor<MT, true> accessor(beta1_pow.data<MT>(),              \
                                       beta2_pow.data<MT>());             \
    if (use_bfloat32_grad) {                                              \
      AdamWKernel<T, float, MT, MOMENT_T, BetaPowAccessor<MT, true>>      \
          <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
              beta1_,                                                     \
              beta2_,                                                     \
              epsilon_,                                                   \
              coeff_,                                                     \
              lr_ratio_,                                                  \
              learning_rate.data<MT>(),                                   \
              grad.data<float>(),                                         \
              param.data<T>(),                                            \
              dev_ctx.template Alloc<T>(param_out),                       \
              master_in_data,                                             \
              master_out_data,                                            \
              moment1.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
              moment2.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,      \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                      : nullptr,                                          \
              accessor,                                                   \
              param.numel(),                                              \
              amsgrad);                                                   \
    } else {                                                              \
      AdamWKernel<T, T, MT, MOMENT_T, BetaPowAccessor<MT, true>>          \
          <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
              beta1_,                                                     \
              beta2_,                                                     \
              epsilon_,                                                   \
              coeff_,                                                     \
              lr_ratio_,                                                  \
              learning_rate.data<MT>(),                                   \
              grad.data<T>(),                                             \
              param.data<T>(),                                            \
              dev_ctx.template Alloc<T>(param_out),                       \
              master_in_data,                                             \
              master_out_data,                                            \
              moment1.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
              moment2.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,      \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                      : nullptr,                                          \
              accessor,                                                   \
              param.numel(),                                              \
              amsgrad);                                                   \
    }                                                                     \
  } else {                                                                \
    BetaPowAccessor<MT, false> accessor(beta1_pow.data<MT>(),             \
                                        beta2_pow.data<MT>());            \
    if (use_bfloat32_grad) {                                              \
      AdamWKernel<T, float, MT, MOMENT_T, BetaPowAccessor<MT, false>>     \
          <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
              beta1_,                                                     \
              beta2_,                                                     \
              epsilon_,                                                   \
              coeff_,                                                     \
              lr_ratio_,                                                  \
              learning_rate.data<MT>(),                                   \
              grad.data<float>(),                                         \
              param.data<T>(),                                            \
              dev_ctx.template Alloc<T>(param_out),                       \
              master_in_data,                                             \
              master_out_data,                                            \
              moment1.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
              moment2.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,      \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                      : nullptr,                                          \
              accessor,                                                   \
              param.numel(),                                              \
              amsgrad);                                                   \
    } else {                                                              \
      AdamWKernel<T, T, MT, MOMENT_T, BetaPowAccessor<MT, false>>         \
          <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
              beta1_,                                                     \
              beta2_,                                                     \
              epsilon_,                                                   \
              coeff_,                                                     \
              lr_ratio_,                                                  \
              learning_rate.data<MT>(),                                   \
              grad.data<T>(),                                             \
              param.data<T>(),                                            \
              dev_ctx.template Alloc<T>(param_out),                       \
              master_in_data,                                             \
              master_out_data,                                            \
              moment1.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
              moment2.data<MOMENT_T>(),                                   \
              dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
              moment2_max ? moment2_max->data<MOMENT_T>() : nullptr,      \
              amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                      : nullptr,                                          \
              accessor,                                                   \
              param.numel(),                                              \
              amsgrad);                                                   \
    }                                                                     \
  }

  // Select template instantiation based on moment type
  if (use_bfloat16_moments) {
    LAUNCH_ADAMW_KERNEL(bfloat16)
  } else {
    LAUNCH_ADAMW_KERNEL(MT)
  }
#undef LAUNCH_ADAMW_KERNEL

  // Update beta_pow
  if (!use_global_beta_pow) {
    if (beta_pow_on_cpu) {
      auto* beta1_pow_out_data = dev_ctx.template HostAlloc<MT>(beta1_pow_out);
      auto* beta2_pow_out_data = dev_ctx.template HostAlloc<MT>(beta2_pow_out);
      beta1_pow_out_data[0] = beta1_ * beta1_pow.data<MT>()[0];
      beta2_pow_out_data[0] = beta2_ * beta2_pow.data<MT>()[0];
    } else {
      UpdateBetaPowKernel<MT><<<1, 1, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          dev_ctx.template Alloc<MT>(beta1_pow_out),
          dev_ctx.template Alloc<MT>(beta2_pow_out));
    }
  }
}

// =============================================================================
// Torch-style AdamW kernel: exact same math as PyTorch's adam_math() in
// fused_adam_utils.cuh. PyTorch's type promotion rules in device code:
//   - opmath_t = float (for float/fp16/bf16 inputs)
//   - lr, beta1, beta2, weight_decay, eps are double
//   - bias_correction1, bias_correction2_sqrt are passed as opmath_t (float)
//     (even though computed in double on host, truncated when passed to
//     adam_math)
//   - When double op float, float promotes to double, result is double, then
//     assigned back to opmath_t truncates to float.
// =============================================================================
// Device-side pow_ matching torch's at::native::pow_ from Pow.cuh (non-MSVC
// path):
//   template <typename Base_type, typename Exp_type>
//   static inline __host__ __device__ Base_type pow_(Base_type base, Exp_type
//   exp) {
//     return ::pow(base, exp);
//   }
// When called as pow_(double, float), C++ promotes float to double,
// so this becomes ::pow(double, double) returning double — exactly matching
// CUDA device pow.
template <typename Base_type, typename Exp_type>
static __device__ __forceinline__ Base_type torch_pow_(Base_type base,
                                                       Exp_type exp) {
  return ::pow(base, exp);
}

template <typename T,        // Parameter type (may be fp16/bf16)
          typename TG,       // Gradient type
          typename MT,       // Master precision type (= opmath_t, float for
                             // float/fp16/bf16)
          typename TM = MT>  // Moment estimation type (default = MT, can be
                             // bfloat16)
__global__ void AdamWStyleKernel(const double beta1,
                                 const double beta2,
                                 const double epsilon,
                                 const double weight_decay,
                                 const double lr_double,
                                 const TG* __restrict__ grad,
                                 const T* __restrict__ param,
                                 T* __restrict__ param_out,
                                 const MT* __restrict__ master_param,
                                 MT* __restrict__ master_param_out,
                                 const TM* __restrict__ moment1,
                                 TM* __restrict__ moment1_out,
                                 const TM* __restrict__ moment2,
                                 TM* __restrict__ moment2_out,
                                 const TM* __restrict__ moment2_max,
                                 TM* __restrict__ moment2_max_out,
                                 // step_count passed as float, matching torch's
                                 // state_steps (float tensor) bias_correction
                                 // computed ON DEVICE, exactly as torch's
                                 // FusedAdamMathFunctor does
                                 const float step_count,
                                 int64_t ndim,
                                 bool amsgrad) {
  // These are then passed to adam_math as const opmath_t& (float), truncating
  // at call boundary.
  const double bc1_dbl = 1.0 - torch_pow_(beta1, step_count);
  const double bc2_dbl = 1.0 - torch_pow_(beta2, step_count);
  const double bc2_sqrt_dbl = ::sqrt(bc2_dbl);
  // Truncate to opmath_t (float) at the "adam_math call boundary", matching
  // torch exactly
  const MT bias_correction1 = static_cast<MT>(bc1_dbl);
  const MT bias_correction2_sqrt = static_cast<MT>(bc2_sqrt_dbl);

  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);

  for (; id < ndim; id += static_cast<int64_t>(gridDim.x) *
                          static_cast<int64_t>(blockDim.x)) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT exp_avg = static_cast<MT>(moment1[id]);
    MT exp_avg_sq = static_cast<MT>(moment2[id]);

    // Weight decay: param -= lr * weight_decay * param
    if (weight_decay != 0) {
      p -= lr_double * weight_decay * p;
    }

    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    // Match torch's FMA behavior: NVCC fuses a*b + c*d into fma(a, b, c*d).
    // We explicitly use __fma_rn to match: first compute (1-beta1)*g (rounded),
    // then fma(beta1, exp_avg_double, rounded_result).
    {
      double g_d = static_cast<double>(g);
      double exp_avg_d = static_cast<double>(exp_avg);
      double one_minus_beta1_times_g = __dmul_rn(1.0 - beta1, g_d);
      exp_avg =
          static_cast<MT>(__fma_rn(beta1, exp_avg_d, one_minus_beta1_times_g));
    }

    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
    // Match torch: ((1-beta2)*g)*g is computed left-to-right, then fma'd.
    {
      double g_d = static_cast<double>(g);
      double exp_avg_sq_d = static_cast<double>(exp_avg_sq);
      double one_minus_beta2_times_g = __dmul_rn(1.0 - beta2, g_d);
      double grad_sq_term = __dmul_rn(one_minus_beta2_times_g, g_d);
      exp_avg_sq = static_cast<MT>(__fma_rn(beta2, exp_avg_sq_d, grad_sq_term));
    }

    // step_size = lr / bias_correction1
    // Match torch: lr_double (double) / bias_correction1 (float) →
    // promotes to double, truncated to float on assignment.
    const MT step_size = lr_double / bias_correction1;

    MT denom;
    if (amsgrad) {
      MT max_exp_avg_sq_val = static_cast<MT>(moment2_max[id]);
      max_exp_avg_sq_val =
          max_exp_avg_sq_val > exp_avg_sq ? max_exp_avg_sq_val : exp_avg_sq;
      moment2_max_out[id] = static_cast<TM>(max_exp_avg_sq_val);
      // Match torch: sqrt(float)/float → float, + double(eps) → double,
      // truncated to float.
      denom = (sqrt(max_exp_avg_sq_val) / bias_correction2_sqrt) + epsilon;
    } else {
      denom = (sqrt(exp_avg_sq) / bias_correction2_sqrt) + epsilon;
    }

    // param -= step_size * exp_avg / denom
    p -= step_size * exp_avg / denom;

    // Store results
    moment1_out[id] = static_cast<TM>(exp_avg);
    moment2_out[id] = static_cast<TM>(exp_avg_sq);
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename Context>
PADDLE_API void AdamwDenseKernel(const Context& dev_ctx,
                                 const DenseTensor& param,
                                 const DenseTensor& grad,
                                 const DenseTensor& learning_rate,
                                 const DenseTensor& moment1,
                                 const DenseTensor& moment2,
                                 const optional<DenseTensor>& moment2_max,
                                 const DenseTensor& beta1_pow,
                                 const DenseTensor& beta2_pow,
                                 const optional<DenseTensor>& master_param,
                                 const optional<DenseTensor>& skip_update,
                                 const Scalar& beta1,
                                 const Scalar& beta2,
                                 const Scalar& epsilon,
                                 double lr_ratio,
                                 double coeff,
                                 bool with_decay,
                                 bool lazy_mode,
                                 int64_t min_row_size_to_use_multithread,
                                 bool multi_precision,
                                 bool use_global_beta_pow,
                                 bool amsgrad,
                                 DenseTensor* param_out,
                                 DenseTensor* moment1_out,
                                 DenseTensor* moment2_out,
                                 DenseTensor* moment2_max_out,
                                 DenseTensor* beta1_pow_out,
                                 DenseTensor* beta2_pow_out,
                                 DenseTensor* master_param_outs) {
  using MT = typename MPTypeTrait<T>::Type;

  bool skip_update_ = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    std::vector<bool> skip_update_vec;
    TensorToVector(*skip_update, dev_ctx, &skip_update_vec);
    skip_update_ = skip_update_vec[0];
  }

  if (skip_update_) {
    VLOG(4) << "Adamw skip update";
    Copy(dev_ctx, param, dev_ctx.GetPlace(), false, param_out);
    Copy(dev_ctx, moment1, dev_ctx.GetPlace(), false, moment1_out);
    Copy(dev_ctx, moment2, dev_ctx.GetPlace(), false, moment2_out);
    if (amsgrad) {
      Copy(dev_ctx,
           moment2_max.get(),
           dev_ctx.GetPlace(),
           false,
           moment2_max_out);
    }
    if (!use_global_beta_pow) {
      Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
      Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    }
    return;
  }

  // weight_decay: if with_decay is false, set to 0 (matching torch behavior)
  double weight_decay = with_decay ? coeff : 0.0;

  double beta1_ = beta1.to<double>();
  double beta2_ = beta2.to<double>();
  double epsilon_ = epsilon.to<double>();

  PADDLE_ENFORCE_EQ(
      beta1_pow_out->numel(),
      1,
      errors::InvalidArgument("beta1 pow output size should be 1, but received "
                              "value is:%d.",
                              beta1_pow_out->numel()));
  PADDLE_ENFORCE_EQ(
      beta2_pow_out->numel(),
      1,
      errors::InvalidArgument("beta2 pow output size should be 1, but received "
                              "value is:%d.",
                              beta2_pow_out->numel()));

  // Compute step_count from beta_pow values, matching torch's state_steps
  // (float tensor). Torch's FusedAdamMathFunctor computes bias_correction ON
  // DEVICE using CUDA pow. We recover step from beta1_pow = beta1^step, then
  // pass step_count (float) to the kernel, which computes bias_correction on
  // device to match torch bit-for-bit.
  double beta1_pow_val;
  const bool beta_pow_on_cpu =
      beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace();
  if (beta_pow_on_cpu) {
    beta1_pow_val = static_cast<double>(beta1_pow.data<MT>()[0]);
  } else {
    MT beta1_pow_cpu;
    phi::memory_utils::Copy(CPUPlace(),
                            &beta1_pow_cpu,
                            beta1_pow.place(),
                            beta1_pow.data<MT>(),
                            sizeof(MT),
                            dev_ctx.stream());
    dev_ctx.Wait();
    beta1_pow_val = static_cast<double>(beta1_pow_cpu);
  }

  // Recover step count: step = round(log(beta_pow) / log(beta))
  double step_from_beta1 =
      std::round(std::log(beta1_pow_val) / std::log(beta1_));
  // Torch stores step as float in state_steps tensor
  float step_count = static_cast<float>(step_from_beta1);

  // Get learning rate as double.
  // The Python optimizer adjusts lr_ratio to compensate for float32 precision
  // loss when PADDLE_ADAMW_TORCH_COMPAT is enabled, so that
  // float32(lr) * lr_ratio recovers the full double-precision learning rate.
  MT lr_on_host;
  phi::memory_utils::Copy(CPUPlace(),
                          &lr_on_host,
                          learning_rate.place(),
                          learning_rate.data<MT>(),
                          sizeof(MT),
                          dev_ctx.stream());
  dev_ctx.Wait();
  const double lr_double = static_cast<double>(lr_on_host) * lr_ratio;

  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MT>(master_param_outs) : nullptr;

  // Launch kernel
  int threads = 512;
  int64_t blocks_max = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int blocks = std::min((param.numel() + threads - 1) / threads, blocks_max);

  // Determine gradient type
  const bool use_bfloat32_grad = grad.dtype() == DataType::FLOAT32;
  // Determine moment type
  const bool use_bfloat16_moments = moment1.dtype() == DataType::BFLOAT16 &&
                                    moment2.dtype() == DataType::BFLOAT16;

#define LAUNCH_ADAMW_STYLE_KERNEL(MOMENT_T)                             \
  if (use_bfloat32_grad) {                                              \
    AdamWStyleKernel<T, float, MT, MOMENT_T>                            \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
            beta1_,                                                     \
            beta2_,                                                     \
            epsilon_,                                                   \
            weight_decay,                                               \
            lr_double,                                                  \
            grad.data<float>(),                                         \
            param.data<T>(),                                            \
            dev_ctx.template Alloc<T>(param_out),                       \
            master_in_data,                                             \
            master_out_data,                                            \
            moment1.data<MOMENT_T>(),                                   \
            dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
            moment2.data<MOMENT_T>(),                                   \
            dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
            amsgrad ? moment2_max->data<MOMENT_T>() : nullptr,          \
            amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                    : nullptr,                                          \
            step_count,                                                 \
            param.numel(),                                              \
            amsgrad);                                                   \
  } else {                                                              \
    AdamWStyleKernel<T, T, MT, MOMENT_T>                                \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
            beta1_,                                                     \
            beta2_,                                                     \
            epsilon_,                                                   \
            weight_decay,                                               \
            lr_double,                                                  \
            grad.data<T>(),                                             \
            param.data<T>(),                                            \
            dev_ctx.template Alloc<T>(param_out),                       \
            master_in_data,                                             \
            master_out_data,                                            \
            moment1.data<MOMENT_T>(),                                   \
            dev_ctx.template Alloc<MOMENT_T>(moment1_out),              \
            moment2.data<MOMENT_T>(),                                   \
            dev_ctx.template Alloc<MOMENT_T>(moment2_out),              \
            amsgrad ? moment2_max->data<MOMENT_T>() : nullptr,          \
            amsgrad ? dev_ctx.template Alloc<MOMENT_T>(moment2_max_out) \
                    : nullptr,                                          \
            step_count,                                                 \
            param.numel(),                                              \
            amsgrad);                                                   \
  }

  // Select template instantiation based on moment type
  if (use_bfloat16_moments) {
    LAUNCH_ADAMW_STYLE_KERNEL(bfloat16)
  } else {
    LAUNCH_ADAMW_STYLE_KERNEL(MT)
  }
#undef LAUNCH_ADAMW_STYLE_KERNEL

  // Update beta_pow (same as original)
  if (!use_global_beta_pow) {
    if (beta_pow_on_cpu) {
      auto* beta1_pow_out_data = dev_ctx.template HostAlloc<MT>(beta1_pow_out);
      auto* beta2_pow_out_data = dev_ctx.template HostAlloc<MT>(beta2_pow_out);
      beta1_pow_out_data[0] = beta1_ * beta1_pow.data<MT>()[0];
      beta2_pow_out_data[0] = beta2_ * beta2_pow.data<MT>()[0];
    } else {
      UpdateBetaPowKernel<MT><<<1, 1, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          dev_ctx.template Alloc<MT>(beta1_pow_out),
          dev_ctx.template Alloc<MT>(beta2_pow_out));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adamw,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamwDenseKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);

  if (kernel_key.dtype() == phi::DataType::FLOAT16 ||
      kernel_key.dtype() == phi::DataType::BFLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(5).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(6).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(4).SetBackend(phi::Backend::UNDEFINED);
  kernel->OutputAt(5).SetBackend(phi::Backend::UNDEFINED);
}
