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

#include "paddle/phi/kernels/adam_kernel.h"

#include <math.h>  // for sqrt in CPU and CUDA

#include <vector>

#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/adam_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {

template <typename T, typename TG, typename MT>
__global__ void AdamKernelREG(MT beta1,
                              MT beta2,
                              MT epsilon,
                              MT beta1_pow_,
                              MT beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* moment2_max,
                              MT* moment2_max_out,
                              const double* lr_,
                              const TG* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int64_t ndim,
                              bool amsgrad) {
  MT lr = static_cast<MT>(*lr_);
  MT beta1_pow = beta1_pow_;
  MT beta2_pow = beta2_pow_;

  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);

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

template <typename T, typename TG, typename MT>
__global__ void AdamKernelMEM(MT beta1,
                              MT beta2,
                              MT epsilon,
                              const MT* beta1_pow_,
                              const MT* beta2_pow_,
                              const MT* moment1,
                              MT* moment1_out,
                              const MT* moment2,
                              MT* moment2_out,
                              const MT* moment2_max,
                              MT* moment2_max_out,
                              const double* lr_,
                              const TG* grad,
                              const T* param,
                              T* param_out,
                              const MT* master_param,
                              MT* master_param_out,
                              int64_t ndim,
                              bool amsgrad) {
  MT lr = static_cast<MT>(*lr_);
  MT beta1_pow = *beta1_pow_;
  MT beta2_pow = *beta2_pow_;

  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);

  for (; id < ndim; id += gridDim.x * blockDim.x) {
    MT p = master_param ? master_param[id] : static_cast<MT>(param[id]);
    MT g = static_cast<MT>(grad[id]);
    MT mom1 = static_cast<MT>(moment1[id]);
    MT mom2 = static_cast<MT>(moment2[id]);

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

template <typename T>
__global__ void UpdateBetaPow(T beta1,
                              T beta2,
                              const T* beta1_pow_,
                              const T* beta2_pow_,
                              T* beta1_pow_out,
                              T* beta2_pow_out) {
  *beta1_pow_out = beta1 * beta1_pow_[0];
  *beta2_pow_out = beta2 * beta2_pow_[0];
}

// ---- Torch-compatible Adam infrastructure ----

// LrAccessor for Adam: lr tensor is MT (float), upcast to double on device.
template <typename MT, bool IsCpu>
struct AdamLrAccessor;

template <typename MT>
struct AdamLrAccessor<MT, true> {
  const double lr_double;
  explicit AdamLrAccessor(double lr) : lr_double(lr) {}
  __device__ __forceinline__ double GetLrDouble() const { return lr_double; }
};

template <typename MT>
struct AdamLrAccessor<MT, false> {
  const double* lr;
  explicit AdamLrAccessor(const double* lr) : lr(lr) {}
  __device__ __forceinline__ double GetLrDouble() const { return *lr; }
};

// Bias correction accessors matching torch's step_count-based computation.
template <typename MT, bool IsCpu>
struct AdamBiasCorrAccessorCompat;

template <typename MT>
struct AdamBiasCorrAccessorCompat<MT, true> {
  const double beta1;
  const double beta2;
  const float step_count;

  AdamBiasCorrAccessorCompat(double b1, double b2, float sc)
      : beta1(b1), beta2(b2), step_count(sc) {}

  __device__ __forceinline__ double GetBc1() const {
    return 1.0 - ::pow(beta1, step_count);
  }
  __device__ __forceinline__ double GetBc2() const {
    return 1.0 - ::pow(beta2, step_count);
  }
};

template <typename MT>
struct AdamBiasCorrAccessorCompat<MT, false> {
  const double beta1;
  const double beta2;
  const MT* beta1_pow;

  AdamBiasCorrAccessorCompat(double b1, double b2, const MT* bp1)
      : beta1(b1), beta2(b2), beta1_pow(bp1) {}

  __device__ __forceinline__ double GetBc1() const {
    const float sc = static_cast<float>(
        ::round(::log(static_cast<double>(*beta1_pow)) / ::log(beta1)));
    return 1.0 - ::pow(beta1, sc);
  }
  __device__ __forceinline__ double GetBc2() const {
    const float sc = static_cast<float>(
        ::round(::log(static_cast<double>(*beta1_pow)) / ::log(beta1)));
    return 1.0 - ::pow(beta2, sc);
  }
};

// Torch-compatible Adam kernel: no weight decay, float lr upcast to double,
// FMA-based moment updates matching torch's fused adam math.
template <typename T,
          typename TG,
          typename MT,
          typename LrAccessor,
          typename BiasCorrAccessor>
__global__ void AdamStyleKernel(const double beta1,
                                const double beta2,
                                const double epsilon,
                                LrAccessor lr_accessor,
                                BiasCorrAccessor bias_corr_accessor,
                                const TG* __restrict__ grad,
                                const T* __restrict__ param,
                                T* __restrict__ param_out,
                                const MT* __restrict__ master_param,
                                MT* __restrict__ master_param_out,
                                const MT* __restrict__ moment1,
                                MT* __restrict__ moment1_out,
                                const MT* __restrict__ moment2,
                                MT* __restrict__ moment2_out,
                                const MT* __restrict__ moment2_max,
                                MT* __restrict__ moment2_max_out,
                                int64_t ndim,
                                bool amsgrad) {
  __shared__ double one_minus_beta1_shared;
  __shared__ double one_minus_beta2_shared;
  __shared__ MT bias_correction2_sqrt_shared;
  __shared__ MT step_size_shared;

  if (threadIdx.x == 0) {
    const double lr_double = lr_accessor.GetLrDouble();
    const double bc1_dbl = bias_corr_accessor.GetBc1();
    const double bc2_dbl = bias_corr_accessor.GetBc2();
    const double bc2_sqrt_dbl = ::sqrt(bc2_dbl);

    one_minus_beta1_shared = 1.0 - beta1;
    one_minus_beta2_shared = 1.0 - beta2;
    const MT bias_correction1 = static_cast<MT>(bc1_dbl);
    bias_correction2_sqrt_shared = static_cast<MT>(bc2_sqrt_dbl);
    step_size_shared = lr_double / bias_correction1;
  }
  __syncthreads();

  const double one_minus_beta1 = one_minus_beta1_shared;
  const double one_minus_beta2 = one_minus_beta2_shared;
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
    const double g_d = static_cast<double>(g);

    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    // FMA variant A: compute (1-beta1)*grad first, then fma(beta1, exp_avg,
    // t2). This matches NVCC's fused adam: beta1*exp_avg computed exactly in
    // FMA.
    {
      const double exp_avg_d = static_cast<double>(exp_avg);
      const double t2 = __dmul_rn(one_minus_beta1, g_d);
      exp_avg = static_cast<MT>(__fma_rn(beta1, exp_avg_d, t2));
    }

    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    // Left-to-right: ((1-beta2)*g)*g, then separate add, matching PyTorch.
    {
      const double exp_avg_sq_d = static_cast<double>(exp_avg_sq);
      const double t1 = __dmul_rn(beta2, exp_avg_sq_d);
      const double t2 = __dmul_rn(one_minus_beta2, g_d);
      const double t3 = __dmul_rn(t2, g_d);
      exp_avg_sq = static_cast<MT>(__dadd_rn(t1, t3));
    }

    MT denom;
    if (amsgrad) {
      MT max_exp_avg_sq = static_cast<MT>(moment2_max[id]);
      max_exp_avg_sq =
          max_exp_avg_sq > exp_avg_sq ? max_exp_avg_sq : exp_avg_sq;
      moment2_max_out[id] = max_exp_avg_sq;
      denom = (sqrt(max_exp_avg_sq) / bias_correction2_sqrt) + epsilon;
    } else {
      denom = (sqrt(exp_avg_sq) / bias_correction2_sqrt) + epsilon;
    }

    p -= step_size * exp_avg / denom;

    moment1_out[id] = exp_avg;
    moment2_out[id] = exp_avg_sq;
    param_out[id] = static_cast<T>(p);
    if (master_param_out) {
      master_param_out[id] = p;
    }
  }
}

template <typename T, typename Context>
void AdamDenseKernel_compatible(const Context& dev_ctx,
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
    VLOG(4) << "Adam skip update";
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
  const bool lr_on_cpu = learning_rate.place() == CPUPlace();

  // Read float lr on host if available; on device it's read in kernel.
  double lr_host_double = 0.0;
  if (lr_on_cpu) {
    lr_host_double = learning_rate.data<double>()[0];
  }

  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? dev_ctx.template Alloc<MT>(master_param_outs) : nullptr;

  const MT* moment2_max_in_data =
      amsgrad ? moment2_max.get().data<MT>() : nullptr;
  MT* moment2_max_out_data =
      amsgrad ? dev_ctx.template Alloc<MT>(moment2_max_out) : nullptr;

  int threads = 512;
  int64_t blocks_max = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int blocks = std::min((param.numel() + threads - 1) / threads, blocks_max);

  const bool use_float32_grad = grad.dtype() == DataType::FLOAT32;

// Use decltype so template args are inferred from the local accessor variables.
#define LAUNCH_ADAM_STYLE_KERNEL()                   \
  if (use_float32_grad) {                            \
    AdamStyleKernel<T,                               \
                    float,                           \
                    MT,                              \
                    decltype(lr_accessor),           \
                    decltype(bias_corr_accessor)>    \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(  \
            beta1_,                                  \
            beta2_,                                  \
            epsilon_,                                \
            lr_accessor,                             \
            bias_corr_accessor,                      \
            grad.data<float>(),                      \
            param.data<T>(),                         \
            dev_ctx.template Alloc<T>(param_out),    \
            master_in_data,                          \
            master_out_data,                         \
            moment1.data<MT>(),                      \
            dev_ctx.template Alloc<MT>(moment1_out), \
            moment2.data<MT>(),                      \
            dev_ctx.template Alloc<MT>(moment2_out), \
            moment2_max_in_data,                     \
            moment2_max_out_data,                    \
            param.numel(),                           \
            amsgrad);                                \
  } else {                                           \
    AdamStyleKernel<T,                               \
                    T,                               \
                    MT,                              \
                    decltype(lr_accessor),           \
                    decltype(bias_corr_accessor)>    \
        <<<blocks, threads, 0, dev_ctx.stream()>>>(  \
            beta1_,                                  \
            beta2_,                                  \
            epsilon_,                                \
            lr_accessor,                             \
            bias_corr_accessor,                      \
            grad.data<T>(),                          \
            param.data<T>(),                         \
            dev_ctx.template Alloc<T>(param_out),    \
            master_in_data,                          \
            master_out_data,                         \
            moment1.data<MT>(),                      \
            dev_ctx.template Alloc<MT>(moment1_out), \
            moment2.data<MT>(),                      \
            dev_ctx.template Alloc<MT>(moment2_out), \
            moment2_max_in_data,                     \
            moment2_max_out_data,                    \
            param.numel(),                           \
            amsgrad);                                \
  }

  if (lr_on_cpu) {
    AdamLrAccessor<MT, true> lr_accessor(lr_host_double);
    if (beta_pow_on_cpu) {
      const float sc = static_cast<float>(
          std::round(std::log(static_cast<double>(beta1_pow.data<MT>()[0])) /
                     std::log(beta1_)));
      AdamBiasCorrAccessorCompat<MT, true> bias_corr_accessor(
          beta1_, beta2_, sc);
      LAUNCH_ADAM_STYLE_KERNEL()
    } else {
      AdamBiasCorrAccessorCompat<MT, false> bias_corr_accessor(
          beta1_, beta2_, beta1_pow.data<MT>());
      LAUNCH_ADAM_STYLE_KERNEL()
    }
  } else {
    AdamLrAccessor<MT, false> lr_accessor(learning_rate.data<double>());
    if (beta_pow_on_cpu) {
      const float sc = static_cast<float>(
          std::round(std::log(static_cast<double>(beta1_pow.data<MT>()[0])) /
                     std::log(beta1_)));
      AdamBiasCorrAccessorCompat<MT, true> bias_corr_accessor(
          beta1_, beta2_, sc);
      LAUNCH_ADAM_STYLE_KERNEL()
    } else {
      AdamBiasCorrAccessorCompat<MT, false> bias_corr_accessor(
          beta1_, beta2_, beta1_pow.data<MT>());
      LAUNCH_ADAM_STYLE_KERNEL()
    }
  }
#undef LAUNCH_ADAM_STYLE_KERNEL

  if (!use_global_beta_pow) {
    if (beta_pow_on_cpu) {
      dev_ctx.template HostAlloc<MT>(beta1_pow_out)[0] =
          static_cast<MT>(beta1_) * beta1_pow.data<MT>()[0];
      dev_ctx.template HostAlloc<MT>(beta2_pow_out)[0] =
          static_cast<MT>(beta2_) * beta2_pow.data<MT>()[0];
    } else {
      UpdateBetaPow<MT><<<1, 1, 0, dev_ctx.stream()>>>(
          static_cast<MT>(beta1_),
          static_cast<MT>(beta2_),
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          dev_ctx.template Alloc<MT>(beta1_pow_out),
          dev_ctx.template Alloc<MT>(beta2_pow_out));
    }
  }
}

template <typename T, typename Context>
PADDLE_API void AdamDenseKernel(const Context& dev_ctx,
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
    AdamDenseKernel_compatible<T, Context>(dev_ctx,
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
  const auto grad_type = grad.dtype();

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  VLOG(4) << "amsgrad: " << amsgrad;

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
    VLOG(4) << "Adam skip update";
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

  if (beta1_pow.place() == CPUPlace() && beta2_pow.place() == CPUPlace()) {
    // Compute with betapow in REG
    if (grad_type == DataType::FLOAT32) {
      AdamKernelREG<T, float, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          *beta1_pow.data<MT>(),
          *beta2_pow.data<MT>(),
          moment1.data<MT>(),
          dev_ctx.template Alloc<MT>(moment1_out),
          moment2.data<MT>(),
          dev_ctx.template Alloc<MT>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<double>(),
          grad.data<float>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    } else {
      AdamKernelREG<T, T, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          *beta1_pow.data<MT>(),
          *beta2_pow.data<MT>(),
          moment1.data<MT>(),
          dev_ctx.template Alloc<MT>(moment1_out),
          moment2.data<MT>(),
          dev_ctx.template Alloc<MT>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<double>(),
          grad.data<T>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    }
    if (!use_global_beta_pow) {
      // Cpu update
      dev_ctx.template HostAlloc<MT>(beta1_pow_out)[0] =
          beta1_ * beta1_pow.data<MT>()[0];
      dev_ctx.template HostAlloc<MT>(beta2_pow_out)[0] =
          beta2_ * beta2_pow.data<MT>()[0];
    }
  } else {
    if (grad_type == DataType::FLOAT32) {
      AdamKernelMEM<T, float, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          moment1.data<MT>(),
          dev_ctx.template Alloc<MT>(moment1_out),
          moment2.data<MT>(),
          dev_ctx.template Alloc<MT>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<double>(),
          grad.data<float>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    } else {
      AdamKernelMEM<T, T, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          epsilon_,
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          moment1.data<MT>(),
          dev_ctx.template Alloc<MT>(moment1_out),
          moment2.data<MT>(),
          dev_ctx.template Alloc<MT>(moment2_out),
          moment2_max_in_data,
          moment2_max_out_data,
          learning_rate.data<double>(),
          grad.data<T>(),
          param.data<T>(),
          dev_ctx.template Alloc<T>(param_out),
          master_in_data,
          master_out_data,
          param.numel(),
          amsgrad);
    }
    if (!use_global_beta_pow) {
      // Update with gpu
      UpdateBetaPow<MT><<<1, 1, 0, dev_ctx.stream()>>>(
          beta1_,
          beta2_,
          beta1_pow.data<MT>(),
          beta2_pow.data<MT>(),
          dev_ctx.template Alloc<MT>(beta1_pow_out),
          dev_ctx.template Alloc<MT>(beta2_pow_out));
    }
  }
}

template <typename T, typename Context>
void MergedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& param,
    const std::vector<const DenseTensor*>& grad,
    const std::vector<const DenseTensor*>& learning_rate,
    const std::vector<const DenseTensor*>& moment1,
    const std::vector<const DenseTensor*>& moment2,
    const optional<std::vector<const DenseTensor*>>& moment2_max,
    const std::vector<const DenseTensor*>& beta1_pow,
    const std::vector<const DenseTensor*>& beta2_pow,
    const optional<std::vector<const DenseTensor*>>& master_param,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    bool multi_precision,
    bool use_global_beta_pow,
    bool amsgrad,
    std::vector<DenseTensor*> param_out,
    std::vector<DenseTensor*> moment1_out,
    std::vector<DenseTensor*> moment2_out,
    std::vector<DenseTensor*> moment2_max_out,
    std::vector<DenseTensor*> beta1_pow_out,
    std::vector<DenseTensor*> beta2_pow_out,
    std::vector<DenseTensor*> master_param_out) {
  using MT = typename MPTypeTrait<T>::Type;
  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  MT beta1_ = beta1.to<MT>();
  MT beta2_ = beta2.to<MT>();
  MT epsilon_ = epsilon.to<MT>();

  size_t param_num = param.size();

  for (size_t idx = 0; idx < param_num; idx++) {
    const MT* master_in_data =
        multi_precision ? master_param.get()[idx]->data<MT>() : nullptr;
    MT* master_out_data =
        multi_precision ? dev_ctx.template Alloc<MT>(master_param_out[idx])
                        : nullptr;

    const MT* moment2_max_in_data =
        amsgrad ? moment2_max.get()[idx]->data<MT>() : nullptr;
    MT* moment2_max_out_data =
        amsgrad ? dev_ctx.template Alloc<MT>(moment2_max_out[idx]) : nullptr;

    // update param and moment
    int threads = 512;
    int64_t blocks_max = dev_ctx.GetCUDAMaxGridDimSize()[0];
    int blocks =
        std::min((param[idx]->numel() + threads - 1) / threads, blocks_max);

    const auto grad_type = grad[idx]->dtype();
    if (beta1_pow[idx]->place() == CPUPlace() &&
        beta2_pow[idx]->place() == CPUPlace()) {
      // Compute with betapow in REG
      if (grad_type == DataType::FLOAT32) {
        AdamKernelREG<T, float, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1_,
            beta2_,
            epsilon_,
            *beta1_pow[idx]->data<MT>(),
            *beta2_pow[idx]->data<MT>(),
            moment1[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment1_out[idx]),
            moment2[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment2_out[idx]),
            moment2_max_in_data,
            moment2_max_out_data,
            learning_rate[idx]->data<double>(),
            grad[idx]->data<float>(),
            param[idx]->data<T>(),
            dev_ctx.template Alloc<T>(param_out[idx]),
            master_in_data,
            master_out_data,
            param[idx]->numel(),
            amsgrad);
      } else {
        AdamKernelREG<T, T, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1_,
            beta2_,
            epsilon_,
            *beta1_pow[idx]->data<MT>(),
            *beta2_pow[idx]->data<MT>(),
            moment1[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment1_out[idx]),
            moment2[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment2_out[idx]),
            moment2_max_in_data,
            moment2_max_out_data,
            learning_rate[idx]->data<double>(),
            grad[idx]->data<T>(),
            param[idx]->data<T>(),
            dev_ctx.template Alloc<T>(param_out[idx]),
            master_in_data,
            master_out_data,
            param[idx]->numel(),
            amsgrad);
      }
      if (!use_global_beta_pow) {
        // Cpu update
        dev_ctx.template HostAlloc<MT>(beta1_pow_out[idx])[0] =
            beta1_ * beta1_pow[idx]->data<MT>()[0];
        dev_ctx.template HostAlloc<MT>(beta2_pow_out[idx])[0] =
            beta2_ * beta2_pow[idx]->data<MT>()[0];
      }
    } else {
      if (grad_type == DataType::FLOAT32) {
        AdamKernelMEM<T, float, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1_,
            beta2_,
            epsilon_,
            beta1_pow[idx]->data<MT>(),
            beta2_pow[idx]->data<MT>(),
            moment1[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment1_out[idx]),
            moment2[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment2_out[idx]),
            moment2_max_in_data,
            moment2_max_out_data,
            learning_rate[idx]->data<double>(),
            grad[idx]->data<float>(),
            param[idx]->data<T>(),
            dev_ctx.template Alloc<T>(param_out[idx]),
            master_in_data,
            master_out_data,
            param[idx]->numel(),
            amsgrad);
      } else {
        AdamKernelMEM<T, T, MT><<<blocks, threads, 0, dev_ctx.stream()>>>(
            beta1_,
            beta2_,
            epsilon_,
            beta1_pow[idx]->data<MT>(),
            beta2_pow[idx]->data<MT>(),
            moment1[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment1_out[idx]),
            moment2[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(moment2_out[idx]),
            moment2_max_in_data,
            moment2_max_out_data,
            learning_rate[idx]->data<double>(),
            grad[idx]->data<T>(),
            param[idx]->data<T>(),
            dev_ctx.template Alloc<T>(param_out[idx]),
            master_in_data,
            master_out_data,
            param[idx]->numel(),
            amsgrad);
      }
      if (!use_global_beta_pow) {
        // Update with gpu
        UpdateBetaPow<MT><<<1, 1, 0, dev_ctx.stream()>>>(
            beta1_,
            beta2_,
            beta1_pow[idx]->data<MT>(),
            beta2_pow[idx]->data<MT>(),
            dev_ctx.template Alloc<MT>(beta1_pow_out[idx]),
            dev_ctx.template Alloc<MT>(beta2_pow_out[idx]));
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::AdamDenseKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT64);
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

PD_REGISTER_KERNEL(merged_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::MergedAdamKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT64);
  // Skip beta1_pow, beta2_pow data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);

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
