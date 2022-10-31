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

#include "paddle/phi/kernels/multi_tensor_adam_kernel.h"
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/multi_tensor_apply.h"

namespace phi {

// This code is referenced from apex's multi_tensor_adam.cu.
// https://github.com/NVIDIA/apex

template <typename T, bool CPUBetaPows /*=true*/>
struct MultiTensorAdamBetaPowInfo {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  MultiTensorAdamBetaPowInfo(const MPDType* beta1pow, const MPDType* beta2pow) {
    beta1pow_ = *beta1pow;
    beta2pow_ = *beta2pow;
  }

  DEVICE MPDType GetBeta1PowValue() const { return beta1pow_; }

  DEVICE MPDType GetBeta2PowValue() const { return beta2pow_; }

 private:
  MPDType beta1pow_;
  MPDType beta2pow_;
};

template <typename T>
struct MultiTensorAdamBetaPowInfo<T, /*CPUBetaPows=*/false> {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  MultiTensorAdamBetaPowInfo(const MPDType* beta1pow, const MPDType* beta2pow) {
    beta1pow_ = beta1pow;
    beta2pow_ = beta2pow;
  }

  DEVICE MPDType GetBeta1PowValue() const { return *beta1pow_; }

  DEVICE MPDType GetBeta2PowValue() const { return *beta2pow_; }

 private:
  const MPDType* __restrict__ beta1pow_;
  const MPDType* __restrict__ beta2pow_;
};

template <typename T,
          typename MT,
          int VecSize,
          bool IsMultiPrecision,
          bool IsCPUBetaPow,
          bool UseAdamW,
          int N,
          int MaxTensorSize,
          int MaxBlockSize>
struct MultiTensorAdamFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      funcs::TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize> t_info,
      MT beta1,
      MT beta2,
      MultiTensorAdamBetaPowInfo<T, IsCPUBetaPow> beta_pow,
      MT epsilon,
      const MT* learning_rate,
      MT decay) const {
    MT lr = *learning_rate;
    MT beta1_pow = beta_pow.GetBeta1PowValue();
    MT beta2_pow = beta_pow.GetBeta2PowValue();

    int chunk_id, tensor_id;
    t_info.GetChunkIdAndTensorId(&chunk_id, &tensor_id);

    int n = t_info.sizes[tensor_id];
    int offset = chunk_id * chunk_size;
    const T* g_ptr = static_cast<const T*>(t_info.grads[tensor_id]) + offset;
    T* p_ptr = static_cast<T*>(t_info.tensor_addrs[0][tensor_id]) + offset;
    MT* mom1_ptr = static_cast<MT*>(t_info.tensor_addrs[1][tensor_id]) + offset;
    MT* mom2_ptr = static_cast<MT*>(t_info.tensor_addrs[2][tensor_id]) + offset;
    MT* mp_ptr =
        IsMultiPrecision
            ? static_cast<MT*>(t_info.tensor_addrs[3][tensor_id]) + offset
            : nullptr;

    n -= offset;
    if (n > chunk_size) {
      n = chunk_size;
    }

    int stride = blockDim.x * VecSize;
    int idx = threadIdx.x * VecSize;

#define PD_ADAM_UPDATE(                                                     \
    __p_vec, __mp_vec, __g_vec, __mom1_vec, __mom2_vec, __i)                \
  MT __p = IsMultiPrecision ? static_cast<MT>(__mp_vec[__i])                \
                            : static_cast<MT>(__p_vec[__i]);                \
  MT __g = static_cast<MT>(__g_vec[__i]);                                   \
  MT __mom1 = static_cast<MT>(__mom1_vec[__i]);                             \
  MT __mom2 = static_cast<MT>(__mom2_vec[__i]);                             \
  if (UseAdamW) {                                                           \
    __p *= (static_cast<MT>(1.0) - lr * decay);                             \
  }                                                                         \
  __mom1 = beta1 * __mom1 + (static_cast<MT>(1.0) - beta1) * __g;           \
  __mom2 = beta2 * __mom2 + (static_cast<MT>(1.0) - beta2) * __g * __g;     \
  MT __denom =                                                              \
      (sqrt(__mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;    \
  __p += (__mom1 / __denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow))); \
  __mom1_vec[__i] = __mom1;                                                 \
  __mom2_vec[__i] = __mom2;                                                 \
  __p_vec[__i] = static_cast<T>(__p);                                       \
  if (IsMultiPrecision) {                                                   \
    __mp_vec[__i] = __p;                                                    \
  }

    for (; idx + VecSize <= n; idx += stride) {
      phi::AlignedVector<T, VecSize> p_vec, g_vec;
      phi::AlignedVector<MT, VecSize> mom1_vec, mom2_vec, mp_vec;
      phi::Load(g_ptr + idx, &g_vec);
      phi::Load(mom1_ptr + idx, &mom1_vec);
      phi::Load(mom2_ptr + idx, &mom2_vec);
      if (IsMultiPrecision) {
        phi::Load(mp_ptr + idx, &mp_vec);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
          PD_ADAM_UPDATE(p_vec, mp_vec, g_vec, mom1_vec, mom2_vec, j);
        }
      } else {
        phi::Load(p_ptr + idx, &p_vec);
#pragma unroll
        for (int j = 0; j < VecSize; ++j) {
          PD_ADAM_UPDATE(p_vec, p_vec, g_vec, mom1_vec, mom2_vec, j);
        }
      }

      phi::Store(mom1_vec, mom1_ptr + idx);
      phi::Store(mom2_vec, mom2_ptr + idx);
      phi::Store(p_vec, p_ptr + idx);
      if (IsMultiPrecision) {
        phi::Store(mp_vec, mp_ptr + idx);
      }
    }

    for (; idx < n; ++idx) {
      if (IsMultiPrecision) {
        PD_ADAM_UPDATE(p_ptr, mp_ptr, g_ptr, mom1_ptr, mom2_ptr, idx);
      } else {
        PD_ADAM_UPDATE(p_ptr, p_ptr, g_ptr, mom1_ptr, mom2_ptr, idx);
      }
    }
  }
};

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

template <typename Context>
static void CopyTensorIfDifferent(const Context& dev_ctx,
                                  const DenseTensor& src,
                                  DenseTensor* dst) {
  if (&src != dst) {
    phi::Copy<Context>(dev_ctx, src, dev_ctx.GetPlace(), false, dst);
  }
}

template <typename Context>
static void CopyTensorIfDifferent(const Context& dev_ctx,
                                  const std::vector<const DenseTensor*>& src,
                                  const std::vector<DenseTensor*>& dst) {
  for (size_t i = 0; i < src.size(); ++i) {
    CopyTensorIfDifferent<Context>(dev_ctx, *(src[i]), dst[i]);
  }
}

template <typename T, typename Context>
void MultiTensorAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& params,
    const std::vector<const DenseTensor*>& grads,
    const DenseTensor& learning_rate,
    const std::vector<const DenseTensor*>& moments1,
    const std::vector<const DenseTensor*>& moments2,
    const DenseTensor& beta1_pow,
    const DenseTensor& beta2_pow,
    const paddle::optional<std::vector<const DenseTensor*>>& master_params,
    const paddle::optional<DenseTensor>& skip_update,
    const Scalar& beta1,
    const Scalar& beta2,
    const Scalar& epsilon,
    int chunk_size,
    float weight_decay,
    bool use_adamw,
    bool multi_precision,
    bool use_global_beta_pow,
    std::vector<DenseTensor*> params_out,
    std::vector<DenseTensor*> moments1_out,
    std::vector<DenseTensor*> moments2_out,
    DenseTensor* beta1_pow_out,
    DenseTensor* beta2_pow_out,
    std::vector<DenseTensor*> master_params_out) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(
      beta1_pow.place(),
      beta2_pow.place(),
      phi::errors::InvalidArgument(
          "Input(Beta1Pow) and Input(Beta2Pow) must be in the same place."));

  bool is_cpu_betapow = (beta1_pow.place() == CPUPlace());

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  MPDType beta1_tmp = beta1.to<MPDType>();
  MPDType beta2_tmp = beta2.to<MPDType>();

  bool skip_update_value = false;
  if (skip_update.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        skip_update->numel(),
        1,
        errors::InvalidArgument("Input(SkipUpdate) size must be 1, but get %d",
                                skip_update->numel()));
    DenseTensor skip_update_tensor;
    phi::Copy(
        dev_ctx, skip_update.get(), CPUPlace(), false, &skip_update_tensor);
    skip_update_value = skip_update_tensor.data<bool>()[0];
    VLOG(4) << "skip_update_value:" << skip_update_value;
  }

  // skip_update=true
  if (skip_update_value) {
    VLOG(4) << "Adam skip update";
    for (size_t i = 0; i < params.size(); i++) {
      phi::Copy(dev_ctx, *params[i], dev_ctx.GetPlace(), false, params_out[i]);
      phi::Copy(
          dev_ctx, *moments1[i], dev_ctx.GetPlace(), false, moments1_out[i]);
      phi::Copy(
          dev_ctx, *moments2[i], dev_ctx.GetPlace(), false, moments2_out[i]);
    }
    phi::Copy(dev_ctx, beta1_pow, beta1_pow.place(), false, beta1_pow_out);
    phi::Copy(dev_ctx, beta2_pow, beta2_pow.place(), false, beta2_pow_out);
    return;
  }

  CopyTensorIfDifferent(dev_ctx, params, params_out);
  CopyTensorIfDifferent(dev_ctx, moments1, moments1_out);
  CopyTensorIfDifferent(dev_ctx, moments2, moments2_out);
  if (master_params) {
    CopyTensorIfDifferent(dev_ctx, master_params.get(), master_params_out);
  }

  std::vector<std::vector<DenseTensor*>> input_vector;
  input_vector.reserve(4);

  input_vector.push_back(params_out);
  input_vector.push_back(moments1_out);
  input_vector.push_back(moments2_out);
  if (multi_precision) {
    input_vector.push_back(master_params_out);
  }

  const int kMaxTensorSizeMp = 24;
  const int kMaxBlockSizeMp = 320;

  const int kMaxTensorSize = 30;
  const int kMaxBlockSize = 320;

  VLOG(4) << "use_adamw: " << use_adamw;
  VLOG(4) << "multi_precision: " << multi_precision;

#define PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(                  \
    __multi_precision, __is_cpu_betapow, __use_adamw, __vec_size)  \
  do {                                                             \
    constexpr int kInputNum = __multi_precision ? 5 : 4;           \
    constexpr int kMaxTensorSize = __multi_precision ? 24 : 30;    \
    constexpr int kMaxBlockSize = __multi_precision ? 320 : 320;   \
    constexpr int kBlockSize = 512;                                \
    MultiTensorAdamBetaPowInfo<T, __is_cpu_betapow> beta_pow_info( \
        beta1_pow.data<MPDType>(), beta2_pow.data<MPDType>());     \
    MultiTensorAdamFunctor<T,                                      \
                           MPDType,                                \
                           __vec_size,                             \
                           __multi_precision,                      \
                           __is_cpu_betapow,                       \
                           __use_adamw,                            \
                           kInputNum,                              \
                           kMaxTensorSize,                         \
                           kMaxBlockSize>                          \
        functor;                                                   \
    funcs::LaunchMultiTensorApplyKernel<kInputNum,                 \
                                        kMaxTensorSize,            \
                                        kMaxBlockSize>(            \
        dev_ctx,                                                   \
        kBlockSize,                                                \
        ((chunk_size + __vec_size - 1) / __vec_size) * __vec_size, \
        input_vector,                                              \
        grads,                                                     \
        functor,                                                   \
        beta1_tmp,                                                 \
        beta2_tmp,                                                 \
        beta_pow_info,                                             \
        epsilon.to<MPDType>(),                                     \
        learning_rate.data<MPDType>(),                             \
        static_cast<MPDType>(weight_decay));                       \
  } while (0)

  constexpr auto kVecSize = 1;
  if (multi_precision) {
    if (is_cpu_betapow) {
      if (use_adamw) {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(true, true, true, kVecSize);
      } else {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(true, true, false, kVecSize);
      }
    } else {
      if (use_adamw) {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(true, false, true, kVecSize);
      } else {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(true, false, false, kVecSize);
      }
    }
  } else {
    if (is_cpu_betapow) {
      if (use_adamw) {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(false, true, true, kVecSize);
      } else {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(false, true, false, kVecSize);
      }
    } else {
      if (use_adamw) {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(false, false, true, kVecSize);
      } else {
        PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(false, false, false, kVecSize);
      }
    }
  }

  if (!use_global_beta_pow) {
    if (is_cpu_betapow) {
      VLOG(10) << "CPU Update BetaPow here...";
      dev_ctx.template HostAlloc<MPDType>(beta1_pow_out)[0] =
          beta1_tmp * beta1_pow.data<MPDType>()[0];
      dev_ctx.template HostAlloc<MPDType>(beta2_pow_out)[0] =
          beta2_tmp * beta2_pow.data<MPDType>()[0];
    } else {
      VLOG(10) << "GPU Update BetaPow here...";
      // Update with gpu
      UpdateBetaPow<MPDType><<<1, 1, 0, dev_ctx.stream()>>>(
          beta1_tmp,
          beta2_tmp,
          beta1_pow.data<MPDType>(),
          beta2_pow.data<MPDType>(),
          dev_ctx.template Alloc<MPDType>(beta1_pow_out),
          dev_ctx.template Alloc<MPDType>(beta2_pow_out));
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(multi_tensor_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiTensorAdamKernel,
                   phi::dtype::float16,
                   float,
                   double) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
