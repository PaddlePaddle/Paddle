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
#include "paddle/common/flags.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

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
                            const double* lr_,
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

// Forward declaration
template <typename T, typename Context>
PADDLE_API void AdamwDenseKernel_compatible(
    const Context& dev_ctx,
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
    DenseTensor* master_param_outs);

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
  if (FLAGS_use_accuracy_compatible_kernel) {
    AdamwDenseKernel_compatible<T, Context>(dev_ctx,
                                            param,
                                            grad,
                                            learning_rate,
                                            moment1,
                                            moment2,
                                            moment2_max,
                                            beta1_pow,
                                            beta2_pow,
                                            master_param,
                                            skip_update,
                                            beta1,
                                            beta2,
                                            epsilon,
                                            lr_ratio,
                                            coeff,
                                            with_decay,
                                            lazy_mode,
                                            min_row_size_to_use_multithread,
                                            multi_precision,
                                            use_global_beta_pow,
                                            amsgrad,
                                            param_out,
                                            moment1_out,
                                            moment2_out,
                                            moment2_max_out,
                                            beta1_pow_out,
                                            beta2_pow_out,
                                            master_param_outs);
    return;
  }
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
              learning_rate.data<double>(),                               \
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
              learning_rate.data<double>(),                               \
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
              learning_rate.data<double>(),                               \
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
              learning_rate.data<double>(),                               \
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
template <bool IsCpu>
struct AdamWLrAccessor;

// cpu
template <>
struct AdamWLrAccessor<true> {
  const double lr_double;

  explicit AdamWLrAccessor(double lr) : lr_double(lr) {}

  __device__ __forceinline__ double GetLrDouble() const { return lr_double; }
};

// gpu
template <>
struct AdamWLrAccessor<false> {
  const double* lr;
  const double lr_ratio;

  AdamWLrAccessor(const double* lr, double lr_ratio)
      : lr(lr), lr_ratio(lr_ratio) {}

  __device__ __forceinline__ double GetLrDouble() const {
    return *lr * lr_ratio;
  }
};

// Device-side pow matching torch's at::native::pow_ (promotes float exp to
// double, then calls ::pow(double, double))
template <typename Base_type, typename Exp_type>
static __device__ __forceinline__ Base_type torch_pow_(Base_type base,
                                                       Exp_type exp) {
  return ::pow(base, exp);
}

// Accuracy-compatible bias correction: computes 1-beta^step_count on device,
// matching torch's FusedAdamMathFunctor. After torch's "Use opmath_t and not
// double compute" change, the bias correction is computed in opmath_t (float
// for fp32/fp16/bf16, double for fp64), with beta and step_count both cast to
// opmath_t before pow_. We therefore compute everything in MT (= opmath_t).
template <typename MT, bool IsCpu>
struct AdamWBiasCorrAccessorCompat;

// CPU specialization: step_count pre-computed on host
template <typename MT>
struct AdamWBiasCorrAccessorCompat<MT, true> {
  const double beta1;
  const double beta2;
  const float step_count;

  AdamWBiasCorrAccessorCompat(double b1, double b2, float sc)
      : beta1(b1), beta2(b2), step_count(sc) {}

  __device__ __forceinline__ MT GetBc1() const {
    return static_cast<MT>(1) -
           torch_pow_(static_cast<MT>(beta1), static_cast<MT>(step_count));
  }
  __device__ __forceinline__ MT GetBc2() const {
    return static_cast<MT>(1) -
           torch_pow_(static_cast<MT>(beta2), static_cast<MT>(step_count));
  }
};

// GPU specialization: recover step_count from beta1_pow pointer on device
template <typename MT>
struct AdamWBiasCorrAccessorCompat<MT, false> {
  const double beta1;
  const double beta2;
  const MT* beta1_pow;

  AdamWBiasCorrAccessorCompat(double b1, double b2, const MT* bp1)
      : beta1(b1), beta2(b2), beta1_pow(bp1) {}

  __device__ __forceinline__ MT GetStepCount() const {
    return static_cast<MT>(
        ::round(::log(static_cast<double>(*beta1_pow)) / ::log(beta1)));
  }

  __device__ __forceinline__ MT GetBc1() const {
    return static_cast<MT>(1) -
           torch_pow_(static_cast<MT>(beta1), GetStepCount());
  }
  __device__ __forceinline__ MT GetBc2() const {
    return static_cast<MT>(1) -
           torch_pow_(static_cast<MT>(beta2), GetStepCount());
  }
};

template <typename T,   // Parameter type (may be fp16/bf16)
          typename TG,  // Gradient type
          typename MT,  // Master precision type (= opmath_t, float for
                        // float/fp16/bf16)
          typename TM,  // Moment estimation type (can be bfloat16)
          typename LrAccessor,
          typename BiasCorrAccessor>
__global__ void AdamWStyleKernel(const double beta1,
                                 const double beta2,
                                 const double epsilon,
                                 const double weight_decay,
                                 LrAccessor lr_accessor,
                                 BiasCorrAccessor bias_corr_accessor,
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
                                 int64_t ndim,
                                 bool amsgrad) {
  // Matches torch >= 2.12's fused Adam(W) math after PR#173224 (use fma) and
  // PR#173227 (use opmath_t, not double): every scalar lives in opmath_t
  // (== MT, float for fp32/fp16/bf16, double for fp64) and the moment updates
  // use nested fma in opmath_t.
  __shared__ MT lr_weight_decay_shared;
  __shared__ MT bias_correction2_sqrt_shared;
  __shared__ MT step_size_shared;

  const MT beta1_o = static_cast<MT>(beta1);
  const MT beta2_o = static_cast<MT>(beta2);
  const MT eps_o = static_cast<MT>(epsilon);

  if (threadIdx.x == 0) {
    // lr cast to opmath_t (matches lr_opmath = static_cast<opmath_t>(lr)).
    const MT lr_o = static_cast<MT>(lr_accessor.GetLrDouble());
    // bias_correction{1,2} are computed in opmath_t inside the accessor.
    const MT bias_correction1 = bias_corr_accessor.GetBc1();
    const MT bias_correction2 = bias_corr_accessor.GetBc2();
    bias_correction2_sqrt_shared = static_cast<MT>(sqrt(bias_correction2));
    // step_size = lr / bias_correction1 (opmath_t / opmath_t).
    step_size_shared = lr_o / bias_correction1;
    // weight decay: param -= lr * weight_decay * param (all opmath_t).
    lr_weight_decay_shared = lr_o * static_cast<MT>(weight_decay);
  }
  __syncthreads();

  const MT lr_weight_decay = lr_weight_decay_shared;
  const MT bias_correction2_sqrt = bias_correction2_sqrt_shared;
  const MT step_size = step_size_shared;

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
      p -= lr_weight_decay * p;
    }

    // exp_avg = fma(beta1, exp_avg, fma(-beta1, grad, grad))
    exp_avg = fma(beta1_o, exp_avg, fma(-beta1_o, g, g));
    // exp_avg_sq = fma(beta2, exp_avg_sq, fma(-beta2, grad*grad, grad*grad))
    const MT g_sq = g * g;
    exp_avg_sq = fma(beta2_o, exp_avg_sq, fma(-beta2_o, g_sq, g_sq));

    MT denom;
    if (amsgrad) {
      MT max_exp_avg_sq_val = static_cast<MT>(moment2_max[id]);
      max_exp_avg_sq_val =
          max_exp_avg_sq_val > exp_avg_sq ? max_exp_avg_sq_val : exp_avg_sq;
      moment2_max_out[id] = static_cast<TM>(max_exp_avg_sq_val);
      denom = (sqrt(max_exp_avg_sq_val) / bias_correction2_sqrt) + eps_o;
    } else {
      denom = (sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps_o;
    }

    // param -= step_size * exp_avg / denom
    p -= step_size * exp_avg / denom;

    moment1_out[id] = static_cast<TM>(exp_avg);
    moment2_out[id] = static_cast<TM>(exp_avg_sq);
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename Context>
PADDLE_API void AdamwDenseKernel_compatible(
    const Context& dev_ctx,
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

  const bool beta_pow_on_cpu =
      beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace();

  // Get learning rate as double. For GPU learning_rate, load it in the CUDA
  // kernel to avoid a host copy and synchronization.
  const bool lr_on_cpu = learning_rate.place() == CPUPlace();
  double lr_double = 0.0;
  if (lr_on_cpu) {
    lr_double = learning_rate.data<double>()[0] * lr_ratio;
  }

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
    AdamWStyleKernel<T,                                                 \
                     float,                                             \
                     MT,                                                \
                     MOMENT_T,                                          \
                     decltype(lr_accessor),                             \
                     decltype(bias_corr_accessor)>                      \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
            beta1_,                                                     \
            beta2_,                                                     \
            epsilon_,                                                   \
            weight_decay,                                               \
            lr_accessor,                                                \
            bias_corr_accessor,                                         \
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
            param.numel(),                                              \
            amsgrad);                                                   \
  } else {                                                              \
    AdamWStyleKernel<T,                                                 \
                     T,                                                 \
                     MT,                                                \
                     MOMENT_T,                                          \
                     decltype(lr_accessor),                             \
                     decltype(bias_corr_accessor)>                      \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(                     \
            beta1_,                                                     \
            beta2_,                                                     \
            epsilon_,                                                   \
            weight_decay,                                               \
            lr_accessor,                                                \
            bias_corr_accessor,                                         \
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
            param.numel(),                                              \
            amsgrad);                                                   \
  }

#define DISPATCH_ADAMW_STYLE_COMPAT_KERNEL(MOMENT_T)                          \
  if (lr_on_cpu) {                                                            \
    AdamWLrAccessor<true> lr_accessor(lr_double);                             \
    if (beta_pow_on_cpu) {                                                    \
      const float sc = static_cast<float>(                                    \
          std::round(std::log(static_cast<double>(beta1_pow.data<MT>()[0])) / \
                     std::log(beta1_)));                                      \
      AdamWBiasCorrAccessorCompat<MT, true> bias_corr_accessor(               \
          beta1_, beta2_, sc);                                                \
      LAUNCH_ADAMW_STYLE_KERNEL(MOMENT_T)                                     \
    } else {                                                                  \
      AdamWBiasCorrAccessorCompat<MT, false> bias_corr_accessor(              \
          beta1_, beta2_, beta1_pow.data<MT>());                              \
      LAUNCH_ADAMW_STYLE_KERNEL(MOMENT_T)                                     \
    }                                                                         \
  } else {                                                                    \
    AdamWLrAccessor<false> lr_accessor(learning_rate.data<double>(),          \
                                       lr_ratio);                             \
    if (beta_pow_on_cpu) {                                                    \
      const float sc = static_cast<float>(                                    \
          std::round(std::log(static_cast<double>(beta1_pow.data<MT>()[0])) / \
                     std::log(beta1_)));                                      \
      AdamWBiasCorrAccessorCompat<MT, true> bias_corr_accessor(               \
          beta1_, beta2_, sc);                                                \
      LAUNCH_ADAMW_STYLE_KERNEL(MOMENT_T)                                     \
    } else {                                                                  \
      AdamWBiasCorrAccessorCompat<MT, false> bias_corr_accessor(              \
          beta1_, beta2_, beta1_pow.data<MT>());                              \
      LAUNCH_ADAMW_STYLE_KERNEL(MOMENT_T)                                     \
    }                                                                         \
  }

  // This function is only reached when FLAGS_use_accuracy_compatible_kernel is
  // true (see AdamwDenseKernel), so only the torch-compatible dispatch exists.
  if (use_bfloat16_moments) {
    DISPATCH_ADAMW_STYLE_COMPAT_KERNEL(bfloat16)
  } else {
    DISPATCH_ADAMW_STYLE_COMPAT_KERNEL(MT)
  }
#undef DISPATCH_ADAMW_STYLE_COMPAT_KERNEL
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
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT64);
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
