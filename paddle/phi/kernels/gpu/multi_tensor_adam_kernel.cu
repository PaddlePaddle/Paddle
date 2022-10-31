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
      const funcs::TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize>& t_info,
      MT beta1,
      MT beta2,
      MultiTensorAdamBetaPowInfo<T, IsCPUBetaPow> beta_pow,
      MT epsilon,
      const MT* learning_rate,
      MT decay) const {
    MT lr = *learning_rate;
    MT beta1_pow = beta_pow.GetBeta1PowValue();
    MT beta2_pow = beta_pow.GetBeta2PowValue();
    T* __restrict__ p_ptr;
    const T* __restrict__ g_ptr;
    MT* __restrict__ mom1_ptr, * __restrict__ mom2_ptr;
    MT* __restrict__ mp_ptr;
    int n;

    {
      int chunk_id, tensor_id;
      t_info.GetChunkIdAndTensorId(&chunk_id, &tensor_id);

      n = t_info.sizes[tensor_id];
      int offset = chunk_id * chunk_size;
      g_ptr = static_cast<const T*>(t_info.grads[tensor_id]) + offset;
      p_ptr = static_cast<T*>(t_info.tensor_addrs[0][tensor_id]) + offset;
      mom1_ptr = static_cast<MT*>(t_info.tensor_addrs[1][tensor_id]) + offset;
      mom2_ptr = static_cast<MT*>(t_info.tensor_addrs[2][tensor_id]) + offset;
      mp_ptr =
          IsMultiPrecision
              ? static_cast<MT*>(t_info.tensor_addrs[3][tensor_id]) + offset
              : nullptr;

      n -= offset;
      if (n > chunk_size) {
        n = chunk_size;
      }
    }

    int stride = blockDim.x * VecSize;
    int idx = threadIdx.x * VecSize;

    for (; idx < n; idx += stride) {
      phi::AlignedVector<T, VecSize> g_vec;
      phi::AlignedVector<T, VecSize> p_vec;
      phi::AlignedVector<MT, VecSize> mp_vec;
      phi::AlignedVector<MT, VecSize> mom1_vec;
      phi::AlignedVector<MT, VecSize> mom2_vec;
      if (idx < n - VecSize && idx < chunk_size - VecSize) {
        if (IsMultiPrecision) {
          phi::Load<MT, VecSize>(mp_ptr + idx, &mp_vec);
        } else {
          phi::Load<T, VecSize>(p_ptr + idx, &p_vec);
        }
        phi::Load<T, VecSize>(g_ptr + idx, &g_vec);
        phi::Load<MT, VecSize>(mom1_ptr + idx, &mom1_vec);
        phi::Load<MT, VecSize>(mom2_ptr + idx, &mom2_vec);
      } else if (idx < n && idx < chunk_size) {
        int size = n - idx;
#pragma unroll
        for (int j = 0; j < size; j++) {
          if (IsMultiPrecision) {
            mp_vec[j] = mp_ptr[idx + j];
          } else {
            p_vec[j] = p_ptr[idx + j];
          }
          g_vec[j] = g_ptr[idx + j];
          mom1_vec[j] = static_cast<MT>(mom1_ptr[idx + j]);
          mom2_vec[j] = static_cast<MT>(mom2_ptr[idx + j]);
        }
#pragma unroll
        for (int j = size; j < VecSize; j++) {
          g_vec[j] = T(0);
          p_vec[j] = T(0);
          mp_vec[j] = MT(0);
          mom1_vec[j] = MT(0);
          mom2_vec[j] = MT(0);
        }
      } else {
        g_vec[0] = T(0);
        g_vec[1] = T(0);
        g_vec[2] = T(0);
        g_vec[3] = T(0);
        p_vec[0] = T(0);
        p_vec[1] = T(0);
        p_vec[2] = T(0);
        p_vec[3] = T(0);
        mp_vec[0] = MT(0);
        mp_vec[1] = MT(0);
        mp_vec[2] = MT(0);
        mp_vec[3] = MT(0);
        mom1_vec[0] = MT(0);
        mom1_vec[1] = MT(0);
        mom1_vec[2] = MT(0);
        mom1_vec[3] = MT(0);
        mom2_vec[0] = MT(0);
        mom2_vec[1] = MT(0);
        mom2_vec[2] = MT(0);
        mom2_vec[3] = MT(0);
      }

#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        MT p = IsMultiPrecision ? mp_vec[j] : static_cast<MT>(p_vec[j]);
        UpdateMoments(&mom1_vec[j],
                      &mom2_vec[j],
                      static_cast<MT>(g_vec[j]),
                      beta1,
                      beta2);
        p = UpdateParameter(p,
                            mom1_vec[j],
                            mom2_vec[j],
                            beta1_pow,
                            beta2_pow,
                            lr,
                            epsilon,
                            decay);
        mp_vec[j] = p;
      }

      if (idx < n && idx < chunk_size) {
        phi::Store<MT, VecSize>(mom1_vec, mom1_ptr + idx);
        phi::Store<MT, VecSize>(mom2_vec, mom2_ptr + idx);
        if (IsMultiPrecision) {
          phi::Store<MT, VecSize>(mp_vec, mp_ptr + idx);
        }
        for (int j = 0; j < VecSize; j++) {
          p_ptr[idx + j] = static_cast<T>(mp_vec[j]);
        }
      } else if (idx < n && idx < chunk_size) {
        int size = n > chunk_size ? chunk_size - idx : n - idx;
#pragma unroll
        for (int j = 0; j < size; j++) {
          if (IsMultiPrecision) {
            mp_ptr[idx + j] = mp_vec[j];
          }
          p_ptr[idx + j] = static_cast<T>(mp_vec[j]);
          mom1_ptr[idx + j] = mom1_vec[j];
          mom2_ptr[idx + j] = mom2_vec[j];
        }
      }
    }
  }

 private:
  static __device__ __forceinline__ void UpdateMoments(
      MT* __restrict__ mom1_ptr,
      MT* __restrict__ mom2_ptr,
      MT g,
      MT beta1,
      MT beta2) {
    MT mom1 = static_cast<MT>(mom1_ptr[0]);
    MT mom2 = static_cast<MT>(mom2_ptr[0]);
    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    mom1_ptr[0] = mom1;
    mom2_ptr[0] = mom2;
  }

  static __device__ __forceinline__ MT UpdateParameter(MT p,
                                                       MT mom1,
                                                       MT mom2,
                                                       MT beta1_pow,
                                                       MT beta2_pow,
                                                       MT lr,
                                                       MT epsilon,
                                                       MT decay) {
    if (UseAdamW) {
      p *= (static_cast<MT>(1.0) - lr * decay);
    }
    MT denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
    return p;
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
                                  const std::vector<const DenseTensor*>& src,
                                  const std::vector<DenseTensor*>& dst) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] != dst[i]) {
      VLOG(10) << "Copy Tensor " << i;
      phi::Copy<Context>(dev_ctx, *(src[i]), dev_ctx.GetPlace(), false, dst[i]);
    }
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

  constexpr auto kVecSize = 4;
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
