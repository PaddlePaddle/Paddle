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
#include "paddle/phi/kernels/multi_tensor_adam_utility_kernel.h"

namespace phi {

const int kBlockSize = 512;
const int kILP = 4;

// This code is referenced from apex's multi_tensor_adam.cu.
// https://github.com/NVIDIA/apex

template <typename T, typename MT, int N, int MaxTensorSize, int MaxBlockSize>
struct MultiTensorAdamFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      TensorAndBlockInfo<N, MaxTensorSize, MaxBlockSize> t_info,
      MT beta1,
      MT beta2,
      const MT* beta1_pow_,
      const MT* beta2_pow_,
      MT epsilon,
      const MT* learning_rate,
      bool use_adamw,
      bool multi_precision,
      MT decay) {
    MT lr = *learning_rate;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;

    int tensor_id = t_info.tenosr_for_this_block[blockIdx.x];

    int chunk_idx = t_info.chunk_for_this_block[blockIdx.x] +
                    t_info.start_chunk_this_tensor;

    int n = t_info.sizes[tensor_id];
    const T* g = static_cast<const T*>(t_info.grads[tensor_id]);
    g += chunk_idx * chunk_size;
    MT* mp;
    T* p;
    p = static_cast<T*>(t_info.tensor_addrs[0][tensor_id]);
    p += chunk_idx * chunk_size;
    MT* m = static_cast<MT*>(t_info.tensor_addrs[1][tensor_id]);
    m += chunk_idx * chunk_size;
    MT* v = static_cast<MT*>(t_info.tensor_addrs[2][tensor_id]);
    v += chunk_idx * chunk_size;

    if (multi_precision) {
      mp = static_cast<MT*>(t_info.tensor_addrs[3][tensor_id]);
      mp += chunk_idx * chunk_size;
    }

    n -= chunk_idx * chunk_size;

    for (int i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * kILP) {
      phi::AlignedVector<T, kILP> g_v;
      phi::AlignedVector<T, kILP> p_v;
      phi::AlignedVector<MT, kILP> mp_v;
      phi::AlignedVector<MT, kILP> m_v;
      phi::AlignedVector<MT, kILP> v_v;
      int i = i_start + threadIdx.x * kILP;
      if (i < n - kILP && i < chunk_size - kILP) {
        if (multi_precision) {
          phi::Load<MT, kILP>(mp + i, &mp_v);
        } else {
          phi::Load<T, kILP>(p + i, &p_v);
        }
        phi::Load<T, kILP>(g + i, &g_v);
        phi::Load<MT, kILP>(m + i, &m_v);
        phi::Load<MT, kILP>(v + i, &v_v);
      } else if (i < n && i < chunk_size) {
        int size = n > chunk_size ? chunk_size - i : n - i;
#pragma unroll
        for (int ii = 0; ii < size; ii++) {
          if (multi_precision) {
            mp_v[ii] = mp[i + ii];
          } else {
            p_v[ii] = p[i + ii];
          }
          g_v[ii] = g[i + ii];
          m_v[ii] = static_cast<MT>(m[i + ii]);
          v_v[ii] = static_cast<MT>(v[i + ii]);
        }
#pragma unroll
        for (int ii = size; ii < kILP; ii++) {
          g_v[ii] = T(0);
          p_v[ii] = T(0);
          mp_v[ii] = MT(0);
          m_v[ii] = MT(0);
          v_v[ii] = MT(0);
        }
      } else {
        g_v[0] = T(0);
        g_v[1] = T(0);
        g_v[2] = T(0);
        g_v[3] = T(0);
        p_v[0] = T(0);
        p_v[1] = T(0);
        p_v[2] = T(0);
        p_v[3] = T(0);
        mp_v[0] = MT(0);
        mp_v[1] = MT(0);
        mp_v[2] = MT(0);
        mp_v[3] = MT(0);
        m_v[0] = MT(0);
        m_v[1] = MT(0);
        m_v[2] = MT(0);
        m_v[3] = MT(0);
        v_v[0] = MT(0);
        v_v[1] = MT(0);
        v_v[2] = MT(0);
        v_v[3] = MT(0);
      }

#pragma unroll
      for (int ii = 0; ii < kILP; ii++) {
        MT p = multi_precision ? mp_v[ii] : static_cast<MT>(p_v[ii]);
        MT g = static_cast<MT>(g_v[ii]);
        MT m = static_cast<MT>(m_v[ii]);
        MT v = static_cast<MT>(v_v[ii]);
        if (!use_adamw) {
          m = beta1 * m + (static_cast<MT>(1.0) - beta1) * g;
          v = beta2 * v + (static_cast<MT>(1.0) - beta2) * g * g;
          m_v[ii] = m;
          v_v[ii] = v;
          MT denom =
              (sqrt(v) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
          p += (m / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
          mp_v[ii] = p;
        } else {  // weight decay
          p *= (static_cast<MT>(1.0) - lr * decay);
          m = beta1 * m + (static_cast<MT>(1.0) - beta1) * g;
          v = beta2 * v + (static_cast<MT>(1.0) - beta2) * g * g;
          m_v[ii] = m;
          v_v[ii] = v;
          MT denom =
              (sqrt(v) / sqrt(static_cast<MT>(1.0) - beta2_pow)) + epsilon;
          p += (m / denom) * (-(lr / (static_cast<MT>(1.0) - beta1_pow)));
          mp_v[ii] = p;
        }
      }
      if (i < n && i < chunk_size) {
        phi::Store<MT, kILP>(m_v, m + i);
        phi::Store<MT, kILP>(v_v, v + i);
        if (multi_precision) {
          phi::Store<MT, kILP>(mp_v, mp + i);
        }
        for (int ii = 0; ii < kILP; ii++) {
          p[i + ii] = static_cast<T>(mp_v[ii]);
        }
      } else if (i < n && i < chunk_size) {
        int size = n > chunk_size ? chunk_size - i : n - i;
#pragma unroll
        for (int ii = 0; ii < size; ii++) {
          if (multi_precision) {
            mp[i + ii] = mp_v[ii];
          }
          p[i + ii] = static_cast<T>(mp_v[ii]);
          m[i + ii] = m_v[ii];
          v[i + ii] = v_v[ii];
        }
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

  VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;
  MPDType beta1_tmp = beta1.to<MPDType>();
  MPDType beta2_tmp = beta2.to<MPDType>();
  MPDType weight_decay_ = static_cast<MPDType>(weight_decay);
  MPDType epsilon_tmp = epsilon.to<MPDType>();

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

  std::vector<std::vector<DenseTensor*>> input_vector;

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

  if (multi_precision) {
    MultiTensorAdamUtilityKernel<5, kMaxTensorSizeMp, kMaxBlockSizeMp, MPDType>(
        dev_ctx,
        kBlockSize,
        chunk_size,
        input_vector,
        grads,
        MultiTensorAdamFunctor<T,
                               MPDType,
                               5,
                               kMaxTensorSizeMp,
                               kMaxBlockSizeMp>(),
        beta1_tmp,
        beta2_tmp,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        epsilon_tmp,
        learning_rate.data<MPDType>(),
        use_adamw,
        multi_precision,
        weight_decay_);
  } else {
    MultiTensorAdamUtilityKernel<4, kMaxTensorSize, kMaxBlockSize, MPDType>(
        dev_ctx,
        kBlockSize,
        chunk_size,
        input_vector,
        grads,
        MultiTensorAdamFunctor<T, MPDType, 4, kMaxTensorSize, kMaxBlockSize>(),
        beta1_tmp,
        beta2_tmp,
        beta1_pow.data<MPDType>(),
        beta2_pow.data<MPDType>(),
        epsilon_tmp,
        learning_rate.data<MPDType>(),
        use_adamw,
        multi_precision,
        weight_decay_);
  }

  if (!use_global_beta_pow) {
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

}  // namespace phi

PD_REGISTER_KERNEL(multi_tensor_adam,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiTensorAdamKernel,
                   phi::dtype::float16,
                   float,
                   double) {}
