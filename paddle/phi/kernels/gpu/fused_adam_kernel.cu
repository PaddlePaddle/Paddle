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

#include "paddle/phi/kernels/fused_adam_kernel.h"
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
struct FusedAdamBetaPowInfo {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  FusedAdamBetaPowInfo(const MPDType* beta1pow, const MPDType* beta2pow) {
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
struct FusedAdamBetaPowInfo<T, /*CPUBetaPows=*/false> {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  FusedAdamBetaPowInfo(const MPDType* beta1pow, const MPDType* beta2pow) {
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
          bool AMSGrad,
          int N,
          int MaxTensorSize,
          int MaxBlockSize>
struct FusedAdamFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      const funcs::TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize>& t_info,
      MT beta1,
      MT beta2,
      FusedAdamBetaPowInfo<T, IsCPUBetaPow> beta_pow,
      MT epsilon,
      const MT* learning_rate,
      MT decay) const {
    MT lr = *learning_rate;
    MT beta1_pow = beta_pow.GetBeta1PowValue();
    MT beta2_pow = beta_pow.GetBeta2PowValue();
    T* __restrict__ p_ptr;
    const T* __restrict__ g_ptr;
    MT* __restrict__ mom1_ptr;
    MT* __restrict__ mom2_ptr;
    MT* __restrict__ mom2_max_ptr;
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
      mom2_max_ptr =
          AMSGrad ? static_cast<MT*>(t_info.tensor_addrs[3][tensor_id]) + offset
                  : nullptr;
      mp_ptr =
          IsMultiPrecision
              ? static_cast<MT*>(
                    t_info.tensor_addrs[3 + (AMSGrad ? 1 : 0)][tensor_id]) +
                    offset
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
      phi::AlignedVector<MT, VecSize> mom2_max_vec;
      if (idx <= n - VecSize) {
        if (IsMultiPrecision) {
          phi::Load<MT, VecSize>(mp_ptr + idx, &mp_vec);
        } else {
          phi::Load<T, VecSize>(p_ptr + idx, &p_vec);
        }
        phi::Load<T, VecSize>(g_ptr + idx, &g_vec);
        phi::Load<MT, VecSize>(mom1_ptr + idx, &mom1_vec);
        phi::Load<MT, VecSize>(mom2_ptr + idx, &mom2_vec);
        if (AMSGrad) {
          phi::Load<MT, VecSize>(mom2_max_ptr + idx, &mom2_max_vec);
        }
      } else {
        int size = n - idx;
        for (int j = 0; j < size; j++) {
          if (IsMultiPrecision) {
            mp_vec[j] = mp_ptr[idx + j];
          } else {
            p_vec[j] = p_ptr[idx + j];
          }
          g_vec[j] = g_ptr[idx + j];
          mom1_vec[j] = static_cast<MT>(mom1_ptr[idx + j]);
          mom2_vec[j] = static_cast<MT>(mom2_ptr[idx + j]);
          if (AMSGrad) {
            mom2_max_vec[j] = static_cast<MT>(mom2_max_ptr[idx + j]);
          }
        }
#pragma unroll
        for (int j = size; j < VecSize; j++) {
          g_vec[j] = T(0);
          p_vec[j] = T(0);
          mp_vec[j] = MT(0);
          mom1_vec[j] = MT(0);
          mom2_vec[j] = MT(0);
          if (AMSGrad) {
            mom2_max_vec[j] = MT(0);
          }
        }
      }

#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        MT p = IsMultiPrecision ? mp_vec[j] : static_cast<MT>(p_vec[j]);
        UpdateMoments(&mom1_vec[j],
                      &mom2_vec[j],
                      AMSGrad ? &mom2_max_vec[j] : nullptr,
                      static_cast<MT>(g_vec[j]),
                      beta1,
                      beta2);
        mp_vec[j] = UpdateParameter(p,
                                    mom1_vec[j],
                                    mom2_vec[j],
                                    AMSGrad ? mom2_max_vec[j] : MT(0),
                                    beta1_pow,
                                    beta2_pow,
                                    lr,
                                    epsilon,
                                    decay);
      }

      if (idx <= n - VecSize) {
        phi::Store<MT, VecSize>(mom1_vec, mom1_ptr + idx);
        phi::Store<MT, VecSize>(mom2_vec, mom2_ptr + idx);
        if (AMSGrad) {
          phi::Store<MT, VecSize>(mom2_max_vec, mom2_max_ptr + idx);
        }
        if (IsMultiPrecision) {
          phi::Store<MT, VecSize>(mp_vec, mp_ptr + idx);
        }
        for (int j = 0; j < VecSize; j++) {
          p_ptr[idx + j] = static_cast<T>(mp_vec[j]);
        }
      } else {
        int size = n - idx;
        for (int j = 0; j < size; j++) {
          if (IsMultiPrecision) {
            mp_ptr[idx + j] = mp_vec[j];
          }
          p_ptr[idx + j] = static_cast<T>(mp_vec[j]);
          mom1_ptr[idx + j] = mom1_vec[j];
          mom2_ptr[idx + j] = mom2_vec[j];
          if (AMSGrad) {
            mom2_max_ptr[idx + j] = mom2_max_vec[j];
          }
        }
      }
    }
  }

 private:
  static __device__ __forceinline__ void UpdateMoments(
      MT* __restrict__ mom1_ptr,
      MT* __restrict__ mom2_ptr,
      MT* __restrict__ mom2_max_ptr,
      MT g,
      MT beta1,
      MT beta2) {
    MT mom1 = static_cast<MT>(mom1_ptr[0]);
    MT mom2 = static_cast<MT>(mom2_ptr[0]);

    mom1 = beta1 * mom1 + (static_cast<MT>(1.0) - beta1) * g;
    mom2 = beta2 * mom2 + (static_cast<MT>(1.0) - beta2) * g * g;

    mom1_ptr[0] = mom1;
    mom2_ptr[0] = mom2;

    if (AMSGrad) {
      MT mom2_max = static_cast<MT>(mom2_max_ptr[0]);
      mom2_max_ptr[0] = std::max(mom2, mom2_max);
    }
  }

  static __device__ __forceinline__ MT UpdateParameter(MT p,
                                                       MT mom1,
                                                       MT mom2,
                                                       MT mom2_max,
                                                       MT beta1_pow,
                                                       MT beta2_pow,
                                                       MT lr,
                                                       MT epsilon,
                                                       MT decay) {
    if (UseAdamW) {
      p *= (static_cast<MT>(1.0) - lr * decay);
    }

    MT denom;
    if (AMSGrad) {
      denom =
          (sqrt(mom2_max) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    } else {
      denom = (sqrt(mom2) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
    }

    p += (mom1 / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
    return p;
  }
};

template <typename T, int N>
__global__ void UpdateBetaPowGroup(
    Array<T*, N> beta1_pow, Array<T*, N> beta2_pow, T beta1, T beta2, int n) {
  auto idx = threadIdx.x;
  if (idx < n) {
    beta1_pow[idx][0] *= beta1;
    beta2_pow[idx][0] *= beta2;
  }
}

template <typename Context>
static void CopyTensorIfDifferent(const Context& dev_ctx,
                                  const std::vector<const DenseTensor*>& src,
                                  const std::vector<DenseTensor*>& dst,
                                  bool use_src_place = false) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] != dst[i]) {
      VLOG(10) << "Copy Tensor " << i;
      phi::Place place = (use_src_place ? src[i]->place() : dev_ctx.GetPlace());
      phi::Copy<Context>(dev_ctx, *(src[i]), place, false, dst[i]);
    }
  }
}

template <typename T, typename TensorT>
static int GetVecSizeFromTensors(const std::vector<TensorT*>& tensors,
                                 int vec_size = 4) {
  for (const auto* t : tensors) {
    vec_size = min(vec_size, GetVectorizedSize(t->template data<T>()));
  }
  return vec_size;
}

template <typename T, typename Context>
void FusedAdamKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& params,
    const std::vector<const DenseTensor*>& grads,
    const DenseTensor& learning_rate,
    const std::vector<const DenseTensor*>& moments1,
    const std::vector<const DenseTensor*>& moments2,
    const paddle::optional<std::vector<const DenseTensor*>>& moments2_max,
    const std::vector<const DenseTensor*>& beta1_pows,
    const std::vector<const DenseTensor*>& beta2_pows,
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
    bool amsgrad,
    std::vector<DenseTensor*> params_out,
    std::vector<DenseTensor*> moments1_out,
    std::vector<DenseTensor*> moments2_out,
    std::vector<DenseTensor*> moments2_max_out,
    std::vector<DenseTensor*> beta1_pows_out,
    std::vector<DenseTensor*> beta2_pows_out,
    std::vector<DenseTensor*> master_params_out) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  auto n = params.size();
  auto beta1_pow_first = beta1_pows[0];
  auto beta2_pow_first = beta2_pows[0];

  for (int i = 1; i < beta1_pows.size(); i++) {
    PADDLE_ENFORCE_EQ(beta1_pow_first->place(),
                      beta1_pows[i]->place(),
                      common::errors::InvalidArgument(
                          "All Beta1Pow must be in the same place."));
    PADDLE_ENFORCE_EQ(beta2_pow_first->place(),
                      beta2_pows[i]->place(),
                      common::errors::InvalidArgument(
                          "All Beta2Pow must be in the same place."));
  }

  PADDLE_ENFORCE_EQ(
      beta1_pow_first->place(),
      beta2_pow_first->place(),
      common::errors::InvalidArgument(
          "Input(Beta1Pows) and Input(Beta2Pows) must be in the same place."));

  bool is_cpu_betapow = (beta1_pow_first->place() == CPUPlace());

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

  CopyTensorIfDifferent(dev_ctx, params, params_out);
  CopyTensorIfDifferent(dev_ctx, moments1, moments1_out);
  CopyTensorIfDifferent(dev_ctx, moments2, moments2_out);
  if (amsgrad) {
    CopyTensorIfDifferent(dev_ctx, moments2_max.get(), moments2_max_out);
  }
  CopyTensorIfDifferent(dev_ctx, beta1_pows, beta1_pows_out, true);
  CopyTensorIfDifferent(dev_ctx, beta2_pows, beta2_pows_out, true);
  if (master_params) {
    CopyTensorIfDifferent(dev_ctx, master_params.get(), master_params_out);
  }

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
    return;
  }

  MPDType beta1_tmp = beta1.to<MPDType>();
  MPDType beta2_tmp = beta2.to<MPDType>();

  std::vector<std::vector<DenseTensor*>> input_vector;
  input_vector.reserve(5);

  input_vector.push_back(params_out);
  input_vector.push_back(moments1_out);
  input_vector.push_back(moments2_out);
  if (amsgrad) {
    input_vector.push_back(moments2_max_out);
  }
  if (multi_precision) {
    input_vector.push_back(master_params_out);
  }

  VLOG(4) << "use_adamw: " << use_adamw;
  VLOG(4) << "multi_precision: " << multi_precision;

#define PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(                       \
    __multi_precision, __is_cpu_betapow, __use_adamw, __amsgrad, __vec_size) \
  do {                                                                       \
    constexpr int kInputNum =                                                \
        (__multi_precision ? 5 : 4) + (__amsgrad ? 1 : 0);                   \
    constexpr int kMaxTensorSize = __multi_precision ? 48 : 60;              \
    constexpr int kMaxBlockSize = __multi_precision ? 320 : 320;             \
    constexpr int kBlockSize = 512;                                          \
    FusedAdamBetaPowInfo<T, __is_cpu_betapow> beta_pow_info(                 \
        beta1_pow_first->data<MPDType>(), beta2_pow_first->data<MPDType>()); \
    FusedAdamFunctor<T,                                                      \
                     MPDType,                                                \
                     __vec_size,                                             \
                     __multi_precision,                                      \
                     __is_cpu_betapow,                                       \
                     __use_adamw,                                            \
                     __amsgrad,                                              \
                     kInputNum,                                              \
                     kMaxTensorSize,                                         \
                     kMaxBlockSize>                                          \
        functor;                                                             \
    funcs::LaunchMultiTensorApplyKernel<kInputNum,                           \
                                        kMaxTensorSize,                      \
                                        kMaxBlockSize>(                      \
        dev_ctx,                                                             \
        kBlockSize,                                                          \
        ((chunk_size + __vec_size - 1) / __vec_size) * __vec_size,           \
        input_vector,                                                        \
        grads,                                                               \
        functor,                                                             \
        beta1_tmp,                                                           \
        beta2_tmp,                                                           \
        beta_pow_info,                                                       \
        epsilon.to<MPDType>(),                                               \
        learning_rate.data<MPDType>(),                                       \
        static_cast<MPDType>(weight_decay));                                 \
  } while (0)

#define PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(__vec_size) \
  case __vec_size: {                                         \
    if (multi_precision) {                                   \
      if (is_cpu_betapow) {                                  \
        if (use_adamw) {                                     \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, true, true, true, __vec_size);         \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, true, true, false, __vec_size);        \
          }                                                  \
        } else {                                             \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, true, false, true, __vec_size);        \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, true, false, false, __vec_size);       \
          }                                                  \
        }                                                    \
      } else {                                               \
        if (use_adamw) {                                     \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, false, true, true, __vec_size);        \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, false, true, false, __vec_size);       \
          }                                                  \
        } else {                                             \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, false, false, true, __vec_size);       \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                true, false, false, false, __vec_size);      \
          }                                                  \
        }                                                    \
      }                                                      \
    } else {                                                 \
      if (is_cpu_betapow) {                                  \
        if (use_adamw) {                                     \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, true, true, true, __vec_size);        \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, true, true, false, __vec_size);       \
          }                                                  \
        } else {                                             \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, true, false, true, __vec_size);       \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, true, false, false, __vec_size);      \
          }                                                  \
        }                                                    \
      } else {                                               \
        if (use_adamw) {                                     \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, false, true, true, __vec_size);       \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, false, true, false, __vec_size);      \
          }                                                  \
        } else {                                             \
          if (amsgrad) {                                     \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, false, false, true, __vec_size);      \
          } else {                                           \
            PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL_BASE(   \
                false, false, false, false, __vec_size);     \
          }                                                  \
        }                                                    \
      }                                                      \
    }                                                        \
  } break

  int vec_size = GetVecSizeFromTensors<T>(params_out);
  vec_size = GetVecSizeFromTensors<MPDType>(moments1_out, vec_size);
  vec_size = GetVecSizeFromTensors<MPDType>(moments2_out, vec_size);
  if (amsgrad) {
    vec_size = GetVecSizeFromTensors<MPDType>(moments2_max_out, vec_size);
  }
  if (master_params) {
    vec_size = GetVecSizeFromTensors<MPDType>(master_params_out, vec_size);
  }

  switch (vec_size) {
    PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(4);
    PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(2);
    PD_LAUNCH_MULTI_TENSOR_APPLY_ADAM_KERNEL(1);
    default:
      PADDLE_THROW(
          errors::InvalidArgument("Unsupported vectorized size %d", vec_size));
      break;
  }

  if (!use_global_beta_pow) {
    if (is_cpu_betapow) {
      for (size_t i = 0; i < n; i++) {
        VLOG(10) << "CPU Update BetaPow here...";
        auto* beta1_ptr =
            dev_ctx.template HostAlloc<MPDType>(beta1_pows_out[i]);
        (*beta1_ptr) *= beta1_tmp;

        auto* beta2_ptr =
            dev_ctx.template HostAlloc<MPDType>(beta2_pows_out[i]);
        (*beta2_ptr) *= beta2_tmp;
      }
    } else {
      constexpr size_t kGroupSize = 32;
      auto group_num = (n + kGroupSize - 1) / kGroupSize;
      VLOG(10) << "GPU Update BetaPow here...";
      for (size_t i = 0; i < group_num; ++i) {
        size_t start = i * kGroupSize;
        size_t end = std::min((i + 1) * kGroupSize, n);
        Array<MPDType*, kGroupSize> beta1_ptrs, beta2_ptrs;
        for (size_t j = start; j < end; ++j) {
          size_t idx = j - start;
          beta1_ptrs[idx] = dev_ctx.template Alloc<MPDType>(beta1_pows_out[j]);
          beta2_ptrs[idx] = dev_ctx.template Alloc<MPDType>(beta2_pows_out[j]);
        }
        UpdateBetaPowGroup<MPDType, kGroupSize>
            <<<1, kGroupSize, 0, dev_ctx.stream()>>>(
                beta1_ptrs, beta2_ptrs, beta1_tmp, beta2_tmp, end - start);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::FusedAdamKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {
  // Skip beta1_pow, beta2_pow, skip_update data transform
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(7).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(9).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(4).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(5).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(6).SetDataType(phi::DataType::UNDEFINED);
}
