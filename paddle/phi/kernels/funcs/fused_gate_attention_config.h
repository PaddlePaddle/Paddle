// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <sstream>
#include <string>
#include <type_traits>

#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi::funcs {

inline std::string MemoryDebugString(const DenseTensor& tensor) {
  int device_id = phi::backends::gpu::GetCurrentDeviceId();
  int64_t allocated =
      phi::memory_utils::DeviceMemoryStatCurrentValue("Allocated", device_id);
  int64_t reserved =
      phi::memory_utils::DeviceMemoryStatCurrentValue("Reserved", device_id);

  std::stringstream stream;
  stream << "shape=[" << tensor.dims()
         << "], size=" << static_cast<float>(tensor.memory_size()) / (1 << 20)
         << " MB, ptr=" << tensor.data()
         << "; [MEMORY] allocated=" << static_cast<float>(allocated) / (1 << 20)
         << " MB"
         << ", reserved=" << static_cast<float>(reserved) / (1 << 20) << " MB";
  return stream.str();
}

template <typename T>
void AllocWithDebugInfo(const GPUContext& dev_ctx,
                        const std::string& info,
                        DenseTensor* tensor) {
  dev_ctx.Alloc<T>(tensor, tensor->numel() * sizeof(T));
  if (VLOG_IS_ON(4)) {
    VLOG(4) << info << ": " << MemoryDebugString(*tensor);
  }
}

inline std::string TensorDebugString(const DenseTensor* tensor,
                                     const std::string& info) {
  std::stringstream stream;
  stream << info << ": ";
  if (tensor) {
    if (tensor->initialized()) {
      stream << "shape=[" << tensor->dims() << "], ptr=" << tensor->data();
    } else {
      stream << "not initialized";
    }
  } else {
    stream << "nullptr";
  }
  return stream.str();
}

inline void WaitWithDebugInfo(const GPUContext& dev_ctx) {
  if (VLOG_IS_ON(5)) {
    dev_ctx.Wait();
    VLOG(5) << "[Flash attn Synchronize] ";
  }
}

template <typename T>
inline void TypeDebugInfo() {
  if (VLOG_IS_ON(4)) {
    if (std::is_same<T, phi::float16>::value) {
      VLOG(4) << "[Grad]: T is phi::float16.";
    } else if (std::is_same<T, phi::bfloat16>::value) {
      VLOG(4) << "[Grad]: T is phi::bfloat16.";
    } else if (std::is_same<T, float>::value) {
      VLOG(4) << "[Grad]: T is float.";
    }
  }
}

template <typename T>
struct GateAttentionConfig {
 public:
  const GPUContext& dev_ctx;

  bool merge_qkv;
  bool has_gating;
  bool use_flash_attn;

  int64_t batch_size;
  int64_t seq_len_m;
  int64_t seq_len_r;
  int64_t q_dim;
  int64_t kv_dim;
  int64_t head_dim;
  int64_t m_size;
  int64_t num_heads;

  DDim qkv_out_dims;
  DDim qkv_transpose_out_dims;

  DDim q_out_dims;
  DDim kv_out_dims;
  DDim q_transpose_out_dims;
  DDim kv_transpose_out_dims;

  DDim qk_out_dims;
  DDim softmax_out_dims;
  DDim qktv_out_dims;
  DDim gate_out_dims;

  GateAttentionConfig(const GPUContext& dev_ctx,
                      const DenseTensor* query,
                      const DenseTensor* key,
                      const DenseTensor* query_weight,
                      const DenseTensor* qkv_weight,
                      bool merge_qkv,
                      bool has_gating,
                      bool use_flash_attn)
      : dev_ctx(dev_ctx),
        merge_qkv(merge_qkv),
        has_gating(has_gating),
        use_flash_attn(use_flash_attn) {
    batch_size = query->dims()[0];
    seq_len_m = query->dims()[1];
    seq_len_r = query->dims()[2];
    q_dim = query->dims()[3];

    if (merge_qkv) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_weight,
          common::errors::NotFound("The input qkv_weight can not be nullptr "
                                   "when merge_qkv is true."));

      num_heads = qkv_weight->dims()[1];
      head_dim = qkv_weight->dims()[2];
      m_size = seq_len_r;
      kv_dim = q_dim;

      qkv_out_dims = {batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim};
      qkv_transpose_out_dims = {
          3, batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
    } else {
      PADDLE_ENFORCE_NOT_NULL(
          key,
          common::errors::NotFound(
              "The input key can not be nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          query_weight,
          common::errors::NotFound("The input query_weight can not be "
                                   "nullptr when merge_qkv is false."));

      num_heads = query_weight->dims()[1];
      head_dim = query_weight->dims()[2];
      m_size = key->dims()[2];
      kv_dim = key->dims()[3];

      q_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, head_dim};
      kv_out_dims = {batch_size, seq_len_m, m_size, num_heads, head_dim};
      q_transpose_out_dims = {
          batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
      kv_transpose_out_dims = {
          batch_size, seq_len_m, num_heads, m_size, head_dim};
    }

    qk_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    softmax_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    qktv_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, head_dim};
    gate_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, head_dim};
  }

  bool CanUseFlashAttn() const {
#if defined(PADDLE_WITH_FLASHATTN) && !defined(PADDLE_WITH_HIP)
    if (!std::is_same<T, phi::bfloat16>::value &&
        !std::is_same<T, phi::float16>::value) {
      return false;
    }

    if (merge_qkv && batch_size == 1 &&
        (head_dim == 32 || head_dim == 64 || head_dim == 128)) {
      return use_flash_attn;
    }
#endif
    return false;
  }

  int64_t GetQuerySize() const {
    return batch_size * seq_len_m * seq_len_r * num_heads * head_dim;
  }

  DenseTensor* GetQKVOut() {
    if (!qkv_out.IsInitialized()) {
      qkv_out.Resize(qkv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "qkv_out", &qkv_out);
    }
    return &qkv_out;
  }

  DenseTensor* GetQueryOut() {
    if (!query_out.IsInitialized()) {
      query_out.Resize(q_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "query_out", &query_out);
    }
    return &query_out;
  }

  DenseTensor* GetKeyOut() {
    if (!key_out.IsInitialized()) {
      key_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "key_out", &key_out);
    }
    return &key_out;
  }

  DenseTensor* GetValueOut() {
    if (!value_out.IsInitialized()) {
      value_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "value_out", &value_out);
    }
    return &value_out;
  }

  DenseTensor* GetQKOut(DenseTensor* softmax_out);

  DenseTensor* GetQKTVOut(DenseTensor* gate_out) {
    if (has_gating && gate_out) {
      gate_out->Resize(qktv_out_dims);
      return gate_out;
    }
    if (!qktv_out.IsInitialized()) {
      qktv_out.Resize(qktv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "qktv_out", &qktv_out);
    }
    return &qktv_out;
  }

  void ClearQKVOut() {
    if (qkv_out.IsInitialized()) qkv_out.clear();
  }

  void ClearQKOut() {
    if (qk_out.IsInitialized()) qk_out.clear();
  }

  void ClearQKTVOut() {
    if (qktv_out.IsInitialized()) qktv_out.clear();
  }

 protected:
  DenseTensor qkv_out;
  DenseTensor query_out;
  DenseTensor key_out;
  DenseTensor value_out;
  DenseTensor qk_out;
  DenseTensor qktv_out;
};

template <typename T>
struct GateAttentionGradConfig : public GateAttentionConfig<T> {
 public:
  GateAttentionGradConfig(const GPUContext& dev_ctx,
                          const DenseTensor* query,
                          const DenseTensor* key,
                          const DenseTensor* query_weight,
                          const DenseTensor* qkv_weight,
                          bool merge_qkv,
                          bool has_gating,
                          bool use_flash_attn)
      : GateAttentionConfig<T>(dev_ctx,
                               query,
                               key,
                               query_weight,
                               qkv_weight,
                               merge_qkv,
                               has_gating,
                               use_flash_attn) {}

  DenseTensor* GetQKVOutGrad() {
    if (!qkv_out_grad.IsInitialized()) {
      qkv_out_grad.Resize(this->qkv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "qkv_out_grad", &qkv_out_grad);
    }
    return &qkv_out_grad;
  }

  DenseTensor* GetQueryOutGrad() {
    if (!query_out_grad.IsInitialized()) {
      query_out_grad.Resize(this->q_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "query_out_grad", &query_out_grad);
    }
    return &query_out_grad;
  }

  DenseTensor* GetKeyOutGrad() {
    if (!key_out_grad.IsInitialized()) {
      key_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "key_out_grad", &key_out_grad);
    }
    return &key_out_grad;
  }

  DenseTensor* GetValueOutGrad() {
    if (!value_out_grad.IsInitialized()) {
      value_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "value_out_grad", &value_out_grad);
    }
    return &value_out_grad;
  }

  DenseTensor* GetQKOutGrad(DenseTensor* softmax_out_grad);

 protected:
  DenseTensor qkv_out_grad;
  DenseTensor query_out_grad;
  DenseTensor key_out_grad;
  DenseTensor value_out_grad;
  DenseTensor qk_out_grad;
};

}  // namespace phi::funcs
