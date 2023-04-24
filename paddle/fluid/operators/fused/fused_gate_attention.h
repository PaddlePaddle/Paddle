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

#include "paddle/phi/kernels/arange_kernel.h"
// #include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#include "paddle/phi/backends/dynload/flashattn.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/range_function.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

#ifdef PADDLE_WITH_FLASHATTN
#include "paddle/phi/kernels/flash_attn_kernel.h"
#endif

namespace paddle {
namespace operators {

inline std::string MemoryDebugString(const phi::DenseTensor& t) {
  int device_id = platform::GetCurrentDeviceId();
  int64_t allocated =
      memory::DeviceMemoryStatCurrentValue("Allocated", device_id);
  int64_t reserved =
      memory::DeviceMemoryStatCurrentValue("Reserved", device_id);

  std::stringstream ss;
  ss << "shape=[" << t.dims()
     << "], size=" << static_cast<float>(t.memory_size()) / (1 << 20)
     << " MB, ptr=" << t.data()
     << "; [MEMORY] allocated=" << static_cast<float>(allocated) / (1 << 20)
     << " MB"
     << ", reserved=" << static_cast<float>(reserved) / (1 << 20) << " MB";
  return ss.str();
}

template <typename T>
void AllocWithDebugInfo(const phi::GPUContext& dev_ctx,
                        const std::string& info,
                        phi::DenseTensor* t) {
  dev_ctx.Alloc<T>(t, t->numel() * sizeof(T));
  VLOG(4) << info << ": " << MemoryDebugString(*t);
}

template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct GateAttentionConfig {
 public:
  const phi::GPUContext& dev_ctx;

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

  phi::DDim qkv_out_dims;
  phi::DDim qkv_transpose_out_dims;

  phi::DDim q_out_dims;
  phi::DDim kv_out_dims;
  phi::DDim q_transpose_out_dims;
  phi::DDim kv_transpose_out_dims;

  phi::DDim qk_out_dims;
  phi::DDim softmax_out_dims;
  phi::DDim qktv_out_dims;
  phi::DDim gate_out_dims;

  GateAttentionConfig(const phi::GPUContext& dev_ctx,
                      const phi::DenseTensor* query,
                      const phi::DenseTensor* key,
                      const phi::DenseTensor* query_weight,
                      const phi::DenseTensor* qkv_weight,
                      bool merge_qkv,
                      bool has_gating,
                      bool use_flash_attn = false)
      : dev_ctx(dev_ctx),
        merge_qkv(merge_qkv),
        has_gating(has_gating),
        use_flash_attn(use_flash_attn) {
    // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
    batch_size = query->dims()[0];
    seq_len_m = query->dims()[1];
    seq_len_r = query->dims()[2];
    q_dim = query->dims()[3];

    if (merge_qkv) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_weight,
          platform::errors::NotFound("The input qkv_weight can not be nullptr "
                                     "when merge_qkv is true."));

      // When q_dim == kv_dim, QKV matmul can be computed merged.
      // qkv_weight: shape=[3, num_heads, head_dim, q_dim]
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
          platform::errors::NotFound(
              "The input key can not be nullptr when merge_qkv is false."));
      PADDLE_ENFORCE_NOT_NULL(
          query_weight,
          platform::errors::NotFound("The input query_weight can not be "
                                     "nullptr when merge_qkv is false."));

      // When q_dim != kv_dim, QKV matmul must be computed saparately.
      // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
      // query_w: shape=[q_dim, num_heads, head_dim]
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

  bool UseFlashAttn(const bool merge_qkv, const bool run_flash_attn) {
    if (!run_flash_attn) {
      return false;
    }

    if (merge_qkv) {
      switch (head_dim) {
        case 16:
        case 32:
        case 64:
        case 128:
          return true;
        default:
          return false;
      }
    } else {
      return false;
    }
  }

  int64_t GetQuerySize() const {
    return batch_size * seq_len_m * seq_len_r * num_heads * head_dim;
  }

  phi::DenseTensor* GetQKVOut() {
    if (!qkv_out.IsInitialized()) {
      qkv_out.Resize(qkv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "qkv_out", &qkv_out);
    }
    return &qkv_out;
  }

  phi::DenseTensor* GetQueryOut() {
    if (!query_out.IsInitialized()) {
      query_out.Resize(q_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "query_out", &query_out);
    }
    return &query_out;
  }

  phi::DenseTensor* GetKeyOut() {
    if (!key_out.IsInitialized()) {
      key_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "key_out", &key_out);
    }
    return &key_out;
  }

  phi::DenseTensor* GetValueOut() {
    if (!value_out.IsInitialized()) {
      value_out.Resize(kv_out_dims);
      AllocWithDebugInfo<T>(dev_ctx, "value_out", &value_out);
    }
    return &value_out;
  }

  phi::DenseTensor* GetQKOut(phi::DenseTensor* softmax_out) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = m_size;
    if (!softmax_out || phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      // Not sure whether cudnn softmax can execute inplace.
      if (!qkv_out.IsInitialized()) {
        qk_out.Resize(qk_out_dims);
        AllocWithDebugInfo<T>(dev_ctx, "qk_out", &qk_out);
      }
      return &qk_out;
    } else {
      // Enable inplace softmax.
      return softmax_out;
    }
  }

  phi::DenseTensor* GetQKTVOut(phi::DenseTensor* gate_out) {
    if (has_gating && gate_out) {
      // Reuse gate_out.
      gate_out->Resize(qktv_out_dims);
      return gate_out;
    } else {
      if (!qktv_out.IsInitialized()) {
        qktv_out.Resize(qktv_out_dims);
        AllocWithDebugInfo<T>(dev_ctx, "qktv_out", &qktv_out);
      }
      return &qktv_out;
    }
  }

  void ClearQKVOut() {
    if (qkv_out.IsInitialized()) {
      qkv_out.clear();
    }
  }

  void ClearQKOut() {
    if (qk_out.IsInitialized()) {
      qk_out.clear();
    }
  }

  void ClearQKTVOut() {
    if (qktv_out.IsInitialized()) {
      qktv_out.clear();
    }
  }

 protected:
  phi::DenseTensor qkv_out;
  phi::DenseTensor query_out;
  phi::DenseTensor key_out;
  phi::DenseTensor value_out;
  // qk_out = BatchedGEMM(Q, K^T)
  // qk_out: shape=[batch_size, seq_len_m, num_heads, seq_len_r, m_size]
  // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
  // The shape of qk_out, softmax_out is the same, thus can be called inplace.
  phi::DenseTensor qk_out;
  // qktv_out may reuse gate_out.
  phi::DenseTensor qktv_out;
};

template <typename T>
struct GateAttentionGradConfig : public GateAttentionConfig<T> {
 public:
  GateAttentionGradConfig(const phi::GPUContext& dev_ctx,
                          const phi::DenseTensor* query,
                          const phi::DenseTensor* key,
                          const phi::DenseTensor* query_weight,
                          const phi::DenseTensor* qkv_weight,
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

  phi::DenseTensor* GetQKVOutGrad() {
    if (!qkv_out_grad.IsInitialized()) {
      qkv_out_grad.Resize(this->qkv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "qkv_out_grad", &qkv_out_grad);
    }
    return &qkv_out_grad;
  }

  phi::DenseTensor* GetQueryOutGrad() {
    if (!query_out_grad.IsInitialized()) {
      query_out_grad.Resize(this->q_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "query_out_grad", &query_out_grad);
    }
    return &query_out_grad;
  }

  phi::DenseTensor* GetKeyOutGrad() {
    if (!key_out_grad.IsInitialized()) {
      key_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "key_out_grad", &key_out_grad);
    }
    return &key_out_grad;
  }

  phi::DenseTensor* GetValueOutGrad() {
    if (!value_out_grad.IsInitialized()) {
      value_out_grad.Resize(this->kv_out_dims);
      AllocWithDebugInfo<T>(this->dev_ctx, "value_out_grad", &value_out_grad);
    }
    return &value_out_grad;
  }

  phi::DenseTensor* GetQKOutGrad(phi::DenseTensor* softmax_out_grad) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = this->m_size;
    if (!softmax_out_grad ||
        phi::UseCudnnSoftmax<T>(this->dev_ctx, softmax_dim, true)) {
      if (!qk_out_grad.IsInitialized()) {
        qk_out_grad.Resize(this->qk_out_dims);
        AllocWithDebugInfo<T>(this->dev_ctx, "qk_out_grad", &qk_out_grad);
      }
      return &qk_out_grad;
    } else {
      return softmax_out_grad;
    }
  }

 protected:
  phi::DenseTensor qkv_out_grad;
  phi::DenseTensor query_out_grad;
  phi::DenseTensor key_out_grad;
  phi::DenseTensor value_out_grad;
  phi::DenseTensor qk_out_grad;
};

#define DEBUG_HERE printf("[%s, %d]: Run here!\n", __func__, __LINE__);
#define DEBUG_DATA_INT(name, x)                                                \
  do {                                                                         \
    printf(                                                                    \
        "[%s, %d]: %s = %d\n", __func__, __LINE__, name, static_cast<int>(x)); \
  }                                                                            \
  whilie(0);

#define DEBUG_DATA_FlOAT(name, x)  \
  do {                             \
    printf("[%s, %d]: %s = %f\n",  \
           __func__,               \
           __LINE__,               \
           std::string(name),      \
           static_cast<float>(x)); \
  }                                \
  whilie(0);

#define DEBUG_DIMS(x)                                    \
  do {                                                   \
    printf("[%s, %d]: dims is : [", __func__, __LINE__); \
    for (int i = 0; i < x.size(); ++i) {                 \
      printf("%d, ", x[i]);                              \
    }                                                    \
    printf(" ]\n");                                      \
  }                                                      \
  whilie(0);

template <typename T>
__global__ void FlashAttRange(int start, int step, int size, T* out1, T* out2) {
  CUDA_KERNEL_LOOP(index, size) {
    out1[index] = static_cast<T>(start + step * index);
    out2[index] = static_cast<T>(start + step * index);
  }
}

static void GetFlashAttnDimsString(const std::string& prefix,
                                   const phi::DDim dim_val) {
  // if (VLOG_IS_ON(4)) {
  std::ostringstream out_string;
  out_string << "FlashAttn - " << prefix << ".dims() is [ ";
  for (int i = 0; i < dim_val.size(); ++i) {
    out_string << dim_val[i] << ", ";
  }
  out_string << "]\n";
  VLOG(4) << out_string.str();
  std::cout << out_string.str();
  // }
}

#define DBGPTR(ptr, prefix)                                              \
  do {                                                                   \
    std::ostringstream out_string;                                       \
    void* data = static_cast<void*>(ptr);                                \
    out_string << "[" << __func__ << ", " << __LINE__ << "]: " << prefix \
               << "`s addr is ";                                         \
    out_string << ptr << std::endl;                                      \
    std::cout << out_string.str();                                       \
  } while (0);

#define DBG_WAIT                                        \
  do {                                                  \
    printf("[%s, %d] Run here.\n", __func__, __LINE__); \
    dev_ctx_.Wait();                                    \
  } while (0);

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const phi::GPUContext& dev_ctx, bool merge_qkv)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv) {}

  void ComputeForward(const phi::DenseTensor* nonbatched_bias,
                      const phi::DenseTensor* src_mask,
                      phi::DenseTensor* q_transpose_out,
                      phi::DenseTensor* k_transpose_out,
                      phi::DenseTensor* v_transpose_out,
                      phi::DenseTensor* qkv_transpose_out,
                      phi::DenseTensor* softmax_out,
                      phi::DenseTensor* fmha_out,
                      phi::DenseTensor* gate_out,
                      GateAttentionConfig<T>* config) {
    T* q_ptr = nullptr;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;
    bool is_bf16 =
        qkv_transpose_out->dtype() == DataType::BFLOAT16 ? true : false;

    if (std::is_same<T, phi::dtype::float16>::value) {
      std::cout << "T is phi::dtype::float16. \n";
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      std::cout << "T is phi::dtype::bfloat16. \n";
    } else if (std::is_same<T, float>::value) {
      std::cout << "T is float. \n";
    }

    if (config->UseFlashAttn(merge_qkv_, config->use_flash_attn && is_bf16)) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_transpose_out,
          platform::errors::NotFound("The input qkv_transpose_out can not be "
                                     "nullptr when merge_qkv is true."));

      // 1. Dealing with qkv_out for flash_attn.
      phi::DenseTensor* qkv_out = config->GetQKVOut();
      ComputeQKVTransposeForwardForFlashAttn(*qkv_out, qkv_transpose_out);
      config->ClearQKVOut();

      int seq_batch_size = static_cast<int>(config->batch_size) *
                           static_cast<int>(config->seq_len_m);
      qkv_transpose_out->Resize(
          {3,
           seq_batch_size * static_cast<int>(config->seq_len_r),
           static_cast<int>(config->num_heads),
           static_cast<int>(config->head_dim)});
      DBG_WAIT;

      // q_size == k_size
      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;

      // 2. Dealing with cu_seq_q and cu_seq_k for flash_attn.
      phi::DenseTensor cu_seq_q, cu_seq_k;
      int64_t end_size = (seq_batch_size + 1);
      int64_t seq_size = 0;
      int64_t start = 0, end = end_size,
              step = static_cast<int>(config->seq_len_r);
      phi::funcs::GetSize<int64_t>(start, end, step, &seq_size);
      cu_seq_q.Resize({end_size});
      cu_seq_k.Resize({end_size});
      AllocWithDebugInfo<int32_t>(dev_ctx_, "cu_seq_q", &cu_seq_q);
      AllocWithDebugInfo<int32_t>(dev_ctx_, "cu_seq_k", &cu_seq_k);
      int64_t block = std::min(seq_size, static_cast<int64_t>(256));
      int64_t grid = (seq_size + block - 1) / block;
      FlashAttRange<int32_t><<<grid, block, 0, dev_ctx_.stream()>>>(
          start, step, end, cu_seq_q.data<int32_t>(), cu_seq_k.data<int32_t>());
      VLOG(4) << "[Flash_attn] cu_seq_len : start = " << start
              << ", step = " << step << ", end = " << end;
      DBG_WAIT;

      // 3. Dealing with mask and bias for flash_attn.
      phi::DenseTensor temp_mask, temp_bias;
      auto dims_merge_func = [&](const phi::DenseTensor* src_tensor,
                                 phi::DenseTensor* dst_tensor,
                                 const std::string& prefix) {
        if (src_tensor) {
          int64_t first_dim = 1;
          dst_tensor->ShareDataWith(*src_tensor);
          auto dims_ = src_tensor->dims();
          for (int i = 0; i < dims_.size() - 3; ++i) {
            first_dim *= dims_[i];
          }
          auto dims_rank = dims_.size();
          dst_tensor->Resize({first_dim,
                              dims_[dims_rank - 3],
                              dims_[dims_rank - 2],
                              dims_[dims_rank - 1]});
          GetFlashAttnDimsString(prefix, temp_mask.dims());
        }
      };
      auto& qkv_dims = qkv_transpose_out->dims();
      dims_merge_func(src_mask, &temp_mask, "mask_dim");
      dims_merge_func(nonbatched_bias, &temp_bias, "bias_dim");
      GetFlashAttnDimsString("qkv_transpose_out", qkv_dims);
      DBG_WAIT;
      // 4. flash_attn parameter setting.

      int batch_size_ = seq_batch_size;
      int total_q_ = qkv_dims[1];    // q.dims()[0]
      int total_k_ = qkv_dims[1];    // q.dims()[0]
      int num_heads_ = qkv_dims[2];  // q.dims()[1]
      int head_size_ = qkv_dims[3];  // q.dims()[2]
      int max_seqlen_q_ = batch_size_;
      int max_seqlen_k_ = batch_size_;
      int num_splits = 0;  // 0 for an internal heuristic, which is optimal
      VLOG(6) << "[Flash_attn Fwd] batch_size : " << batch_size_;
      VLOG(6) << "[Flash_attn Fwd] total_q   : " << total_q_;
      VLOG(6) << "[Flash_attn Fwd] total_k   : " << total_k_;
      VLOG(6) << "[Flash_attn Fwd] num_heads : " << num_heads_;
      VLOG(6) << "[Flash_attn Fwd] head_size : " << head_size_;
      VLOG(6) << "[Flash_attn Fwd] max_seqlen_q : " << max_seqlen_q_;
      VLOG(6) << "[Flash_attn Fwd] max_seqlen_k : " << max_seqlen_k_;

      // 5. construct softmax_lse
      phi::DenseTensor softmax_lse;
      int softmax_lse_last_dim = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
      softmax_lse.Resize({batch_size_, num_heads_, softmax_lse_last_dim});
      AllocWithDebugInfo<float>(
          dev_ctx_, "flash_attn: softmax_lse", &softmax_lse);

      DBG_WAIT;
      // 6. construct random seed
      auto gen = dev_ctx_.GetGenerator();
      uint64_t inc = batch_size_ * num_heads_ * 32;
      auto seed_offset_pair = gen->IncrementOffset(inc);
      uint64_t seed = seed_offset_pair.first;
      uint64_t offset = seed_offset_pair.second;

      GetFlashAttnDimsString("softmax_out", softmax_out->dims());
      GetFlashAttnDimsString("softmax_lse", softmax_lse.dims());
      GetFlashAttnDimsString("cu_seq_q", cu_seq_q.dims());
      GetFlashAttnDimsString("cu_seq_k", cu_seq_k.dims());
      DBG_WAIT;

      // 7. flas_attn part one, get temp worksapce size.
      float p_dropout = 0.f;
      float softmax_scale = static_cast<float>(1);
      cudaStream_t stream = dev_ctx_.stream();
      uint64_t workspace_size;
      bool succ = phi::dynload::flash_attn_fwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          nullptr,  // for calculation workspace size
          cu_seq_q.data<int32_t>(),
          cu_seq_k.data<int32_t>(),
          total_q_,
          total_k_,
          batch_size_,
          num_heads_,
          head_size_,
          max_seqlen_q_,
          max_seqlen_k_,
          p_dropout,
          softmax_scale,
          /*zero_tensors=*/false,
          /*is_causal=*/false,
          is_bf16,
          num_splits,
          softmax_lse.data(),
          softmax_out->data(),
          nullptr,
          &workspace_size,
          stream,
          seed,
          offset,
          src_mask ? temp_mask.data() : nullptr,
          nonbatched_bias ? temp_bias.data() : nullptr,
          temp_mask.dims().Get(),
          temp_bias.dims().Get());
      if (!succ) {
        PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
      }
      DBG_WAIT;

      phi::DenseTensor workspace;
      printf("workspace_size = %d\n", workspace_size);
      if (workspace_size > 0) {
        workspace = phi::Empty<float, phi::GPUContext>(
            dev_ctx_, {int64_t(workspace_size / sizeof(float))});
        DBGPTR(workspace.data(), "workspace");
      }
      DBG_WAIT;

#define DBG_INIT(prefix, x)                                       \
  do {                                                            \
    printf("[%s, %d] ", __func__, __LINE__);                      \
    if (x->initialized()) {                                       \
      std::cout << prefix << " is initialized." << std::endl;     \
    } else {                                                      \
      std::cout << prefix << " is not initialized." << std::endl; \
    }                                                             \
  } while (0);
      DBG_INIT("qkv_transpose_out", qkv_transpose_out);
      DBG_INIT("softmax_out", softmax_out);
      DBG_INIT("src_mask", src_mask);
      DBG_INIT("fmha_out", fmha_out);
      DBG_INIT("gate_out", gate_out);

      // 8. flas_attn part two, run impl.
      succ = phi::dynload::flash_attn_fwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          static_cast<void*>(
              fmha_out->data()),  // for calculation workspace size
          cu_seq_q.data<int32_t>(),
          cu_seq_k.data<int32_t>(),
          total_q_,
          total_k_,
          batch_size_,
          num_heads_,
          head_size_,
          max_seqlen_q_,
          max_seqlen_k_,
          p_dropout,
          softmax_scale,
          /*zero_tensors=*/false,
          /*is_causal=*/false,
          is_bf16,
          num_splits,
          softmax_lse.data(),
          softmax_out->data(),
          (workspace_size > 0) ? static_cast<void*>(workspace.data()) : nullptr,
          &workspace_size,
          stream,
          seed,
          offset,
          src_mask ? temp_mask.data() : nullptr,
          nonbatched_bias ? temp_bias.data() : nullptr,
          temp_mask.dims().Get(),
          temp_bias.dims().Get());
      DBG_WAIT;
      if (!succ) {
        PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
      }
      DBG_WAIT;
    } else {
      if (merge_qkv_) {
        // qkv_transpose_out = transpose(qkv_out)
        PADDLE_ENFORCE_NOT_NULL(
            qkv_transpose_out,
            platform::errors::NotFound("The input qkv_transpose_out can not be "
                                       "nullptr when merge_qkv is true."));

        phi::DenseTensor* qkv_out = config->GetQKVOut();
        ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);
        config->ClearQKVOut();

        // q_size == k_size
        int64_t q_size = config->GetQuerySize();
        q_ptr = qkv_transpose_out->data<T>();
        k_ptr = q_ptr + q_size;
        v_ptr = k_ptr + q_size;
      } else {
        PADDLE_ENFORCE_NOT_NULL(
            q_transpose_out,
            platform::errors::NotFound("The input q_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));
        PADDLE_ENFORCE_NOT_NULL(
            k_transpose_out,
            platform::errors::NotFound("The input k_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));
        PADDLE_ENFORCE_NOT_NULL(
            v_transpose_out,
            platform::errors::NotFound("The input v_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));

        phi::DenseTensor* query_out = config->GetQueryOut();
        phi::DenseTensor* key_out = config->GetKeyOut();
        phi::DenseTensor* value_out = config->GetValueOut();
        ComputeQKVTransposeForward(*query_out,
                                   *key_out,
                                   *value_out,
                                   q_transpose_out,
                                   k_transpose_out,
                                   v_transpose_out);

        // q_size != k_size
        q_ptr = q_transpose_out->data<T>();
        k_ptr = k_transpose_out->data<T>();
        v_ptr = v_transpose_out->data<T>();
      }
      // qk_out = BatchedGEMM(Q, K^T)
      // [batch_size, seq_len_m, num_heads, seq_len_r, head_dim] *
      //                [batch_size, seq_len_m, num_heads, m_size, head_dim]
      // -> [batch_size, seq_len_m, num_heads, seq_len_r, m_size]
      phi::DenseTensor* qk_out = config->GetQKOut(softmax_out);
      T* qk_out_ptr = qk_out->data<T>();

      int64_t gemm_batch_size =
          config->batch_size * config->seq_len_m * config->num_heads;
      int64_t gemm_m = config->seq_len_r;
      int64_t gemm_n = config->m_size;
      int64_t gemm_k = config->head_dim;
      // attn = torch.matmul(q, k.transpose(-1, -2))
      T alpha = static_cast<T>(1.0 / sqrt(config->head_dim));
      // ComputeBatchedGEMM(merge_qkv_ ?
      //                         phi::slice_ddim(qkv_transpose_out->dims(),
      //                                         1,
      //                                         qkv_transpose_out->dims().size()
      //                                         - 1) : q_transpose_out->dims(),
      //                    merge_qkv_ ?
      //                         phi::slice_ddim(qkv_transpose_out->dims(),
      //                                         1,
      //                                         qkv_transpose_out->dims().size()
      //                                         - 1) : k_transpose_out->dims(),
      //                    q_ptr,
      //                    k_ptr,
      //                    qk_out_ptr,
      //                    false,
      //                    true,
      //                    gemm_m,
      //                    gemm_n,
      //                    gemm_k,
      //                    gemm_batch_size,
      //                    alpha);

      ComputeBatchedGEMM(q_ptr,
                         k_ptr,
                         qk_out_ptr,
                         false,
                         true,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size,
                         alpha);
      // attn = softmax_dropout(attn, 0, self.training, mask=mask, bias=bias)
      // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
      ComputeBiasMaskSoftmaxForward(
          nonbatched_bias, src_mask, qk_out, softmax_out);
      config->ClearQKOut();

      // qktv_out = BatchedGEMM(softmax_out, V)
      // [batch_size, seq_len_m, num_heads, seq_len_r, m_size] *
      // [batch_size, seq_len_m, num_heads, m_size,    head_dim]
      // -> [batch_size, seq_len_m, num_heads, seq_len_r, head_dim]
      phi::DenseTensor* qktv_out = config->GetQKTVOut(gate_out);
      T* qktv_out_ptr = qktv_out->data<T>();

      gemm_m = config->seq_len_r;
      gemm_n = config->head_dim;
      gemm_k = config->m_size;

      // o = torch.matmul(attn, v)
      T* softmax_out_ptr = softmax_out->data<T>();
      // ComputeBatchedGEMM(softmax_out->dims(),
      //                    merge_qkv_ ?
      //                         phi::slice_ddim(qkv_transpose_out->dims(),
      //                                         1,
      //                                         qkv_transpose_out->dims().size()
      //                                         - 1) : v_transpose_out->dims(),
      //                    softmax_out_ptr,
      //                    v_ptr,
      //                    qktv_out_ptr,
      //                    false,
      //                    false,
      //                    gemm_m,
      //                    gemm_n,
      //                    gemm_k,
      //                    gemm_batch_size);

      ComputeBatchedGEMM(softmax_out_ptr,
                         v_ptr,
                         qktv_out_ptr,
                         false,
                         false,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size);

      // fmha_out = transpose(qktv_out)
      // o = o.transpose(-2, -3).contiguous()
      ComputeQKTVTransposeForward(*qktv_out, fmha_out);
    }

    config->ClearQKTVOut();
    if (config->has_gating) {
      gate_out->Resize(config->gate_out_dims);
    }
  }

  void ComputeBackward(const phi::DenseTensor* q_transpose_out,
                       const phi::DenseTensor* k_transpose_out,
                       const phi::DenseTensor* v_transpose_out,
                       const phi::DenseTensor* qkv_transpose_out,
                       const phi::DenseTensor* softmax_out,
                       const phi::DenseTensor* fmha_out_grad,
                       phi::DenseTensor* src_mask_grad,
                       phi::DenseTensor* nonbatched_bias_grad,
                       GateAttentionGradConfig<T>* config,
                       const phi::DenseTensor* fmha_out = nullptr,
                       const phi::DenseTensor* softmax_lse = nullptr,
                       const phi::DenseTensor* nonbatched_bias = nullptr,
                       const phi::DenseTensor* src_mask = nullptr) {
    const T* q_ptr = nullptr;
    const T* k_ptr = nullptr;
    const T* v_ptr = nullptr;

    T* q_grad_ptr = nullptr;
    T* k_grad_ptr = nullptr;
    T* v_grad_ptr = nullptr;

    phi::DenseTensor q_transpose_out_grad;
    phi::DenseTensor k_transpose_out_grad;
    phi::DenseTensor v_transpose_out_grad;
    phi::DenseTensor qkv_transpose_out_grad;

    bool is_bf16 =
        qkv_transpose_out->dtype() == DataType::BFLOAT16 ? true : false;

    if (std::is_same<T, phi::dtype::float16>::value) {
      std::cout << "[Grad]: T is phi::dtype::float16. \n";
    } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
      std::cout << "[Grad]: T is phi::dtype::bfloat16. \n";
    } else if (std::is_same<T, float>::value) {
      std::cout << "[Grad]: T is float. \n";
    }

    if (config->UseFlashAttn(merge_qkv_, config->use_flash_attn && is_bf16)) {
      PADDLE_ENFORCE_NOT_NULL(
          qkv_transpose_out,
          platform::errors::NotFound("The input qkv_transpose_out can not be"
                                     "nullptr when merge_qkv is true."));
      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;

      qkv_transpose_out_grad.Resize(config->qkv_transpose_out_dims);
      AllocWithDebugInfo<T>(
          dev_ctx_, "qkv_transpose_out_grad", &qkv_transpose_out_grad);

      int seq_batch_size = static_cast<int>(config->batch_size) *
                           static_cast<int>(config->seq_len_m);
      DBG_WAIT;

      // 2. Dealing with cu_seq_q and cu_seq_k for flash_attn.
      phi::DenseTensor cu_seq_q, cu_seq_k;
      int64_t start = 0;
      int64_t step = static_cast<int>(config->seq_len_r);
      int64_t end_size = (seq_batch_size + 1);
      int64_t end = end_size;
      int64_t seq_size = 0;
      phi::funcs::GetSize<int64_t>(start, end, step, &seq_size);
      cu_seq_q.Resize({end_size});
      cu_seq_k.Resize({end_size});
      AllocWithDebugInfo<int32_t>(dev_ctx_, "Grad: cu_seq_q", &cu_seq_q);
      AllocWithDebugInfo<int32_t>(dev_ctx_, "Grad: cu_seq_k", &cu_seq_k);
      int64_t block = std::min(seq_size, static_cast<int64_t>(256));
      int64_t grid = (seq_size + block - 1) / block;
      FlashAttRange<int32_t><<<grid, block, 0, dev_ctx_.stream()>>>(
          start, step, end, cu_seq_q.data<int32_t>(), cu_seq_k.data<int32_t>());
      VLOG(4) << "[Flash_attn] cu_seq_len : start = " << start
              << ", step = " << step << ", end = " << end;
      DBG_WAIT;

      // 3. Dealing with mask and bias for flash_attn.
      phi::DenseTensor temp_mask, temp_bias;
      auto dims_merge_func = [&](const phi::DenseTensor* src_tensor,
                                 phi::DenseTensor* dst_tensor,
                                 const std::string& prefix) {
        if (src_tensor) {
          int64_t first_dim = 1;
          dst_tensor->ShareDataWith(*src_tensor);
          auto dims_ = src_tensor->dims();
          for (int i = 0; i < dims_.size() - 3; ++i) {
            first_dim *= dims_[i];
          }
          auto dims_rank = dims_.size();
          dst_tensor->Resize({first_dim,
                              dims_[dims_rank - 3],
                              dims_[dims_rank - 2],
                              dims_[dims_rank - 1]});
          GetFlashAttnDimsString(prefix, temp_mask.dims());
        }
      };
      dims_merge_func(src_mask, &temp_mask, "[Grad] mask_dim");
      dims_merge_func(nonbatched_bias, &temp_bias, "[Grad] bias_dim");

      phi::DDim qkv_dims({3,
                          seq_batch_size * static_cast<int>(config->seq_len_r),
                          static_cast<int>(config->num_heads),
                          static_cast<int>(config->head_dim)});
      int batch_size_ = seq_batch_size;
      int total_q_ = qkv_dims[1];    // q.dims()[0]
      int total_k_ = qkv_dims[1];    // q.dims()[0]
      int num_heads_ = qkv_dims[2];  // q.dims()[1]
      int head_size_ = qkv_dims[3];  // q.dims()[2]
      int max_seqlen_q_ = batch_size_;
      int max_seqlen_k_ = batch_size_;
      VLOG(6) << "[Flash_attn Grad] batch_size : " << batch_size_;
      VLOG(6) << "[Flash_attn Grad] total_q   : " << total_q_;
      VLOG(6) << "[Flash_attn Grad] total_k   : " << total_k_;
      VLOG(6) << "[Flash_attn Grad] num_heads : " << num_heads_;
      VLOG(6) << "[Flash_attn Grad] head_size : " << head_size_;
      VLOG(6) << "[Flash_attn Grad] max_seqlen_q : " << max_seqlen_q_;
      VLOG(6) << "[Flash_attn Grad] max_seqlen_k : " << max_seqlen_k_;

      // 5. construct softmax_lse
      int last_q_dim = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
      // softmax_lse->Resize({batch_size_, num_heads_, last_q_dim});
      // AllocWithDebugInfo<float>(
      //     dev_ctx_, "flash_attn: softmax_lse", softmax_lse);
      DBG_WAIT;

      phi::DenseTensor softmax_d = phi::Empty<float, phi::GPUContext>(
          dev_ctx_, {batch_size_, num_heads_, last_q_dim});
      DBG_WAIT;

      phi::DenseTensor bias_d;
      if (nonbatched_bias) {
        bias_d = phi::Empty<T, phi::GPUContext>(
            dev_ctx_, {batch_size_, num_heads_, max_seqlen_q_, max_seqlen_k_});
      }
      DBG_WAIT;

      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();
      q_transpose_out_grad.Resize(config->q_transpose_out_dims);
      k_transpose_out_grad.Resize(config->kv_transpose_out_dims);
      v_transpose_out_grad.Resize(config->kv_transpose_out_dims);

      q_grad_ptr = dev_ctx_.Alloc<T>(&q_transpose_out_grad,
                                     q_transpose_out_grad.numel() * sizeof(T));
      k_grad_ptr = dev_ctx_.Alloc<T>(&k_transpose_out_grad,
                                     k_transpose_out_grad.numel() * sizeof(T));
      v_grad_ptr = dev_ctx_.Alloc<T>(&v_transpose_out_grad,
                                     v_transpose_out_grad.numel() * sizeof(T));

      // 6. construct random seed
      auto gen = dev_ctx_.GetGenerator();
      uint64_t inc = batch_size_ * num_heads_ * 32;
      auto seed_offset_pair = gen->IncrementOffset(inc);
      uint64_t seed = seed_offset_pair.first;
      uint64_t offset = seed_offset_pair.second;

      // 7. flas_attn part one, get temp worksapce size.
      uint64_t workspace_size;
      float p_dropout = 0.f;
      float softmax_scale = static_cast<float>(1);
      cudaStream_t stream = dev_ctx_.stream();
      int num_splits = 0;  // 0 for an internal heuristic, which is optimal
      bool succ = phi::dynload::flash_attn_bwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          static_cast<void*>(q_grad_ptr),
          static_cast<void*>(k_grad_ptr),
          static_cast<void*>(v_grad_ptr),
          static_cast<const void*>(fmha_out->data()),
          static_cast<const void*>(fmha_out_grad->data()),
          cu_seq_q.data<int32_t>(),
          cu_seq_k.data<int32_t>(),
          total_q_,
          total_k_,
          batch_size_,
          num_heads_,
          head_size_,
          max_seqlen_q_,
          max_seqlen_k_,
          p_dropout,
          softmax_scale,
          /*zero_tensors=*/false,
          /*is_causal=*/false,
          is_bf16,
          num_splits,
          softmax_lse->data(),
          softmax_d.data(),
          bias_d.data(),
          nullptr,
          &workspace_size,
          stream,
          seed,
          offset,
          src_mask ? temp_mask.data() : nullptr,
          nonbatched_bias ? temp_bias.data() : nullptr,
          temp_mask.dims().Get(),
          temp_bias.dims().Get());
      if (!succ) {
        PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
      }
      DBG_WAIT;

      phi::DenseTensor workspace;
      printf("workspace_size = %d\n", workspace_size);
      if (workspace_size > 0) {
        workspace = phi::Empty<float, phi::GPUContext>(
            dev_ctx_, {int64_t(workspace_size / sizeof(float))});
        DBGPTR(workspace.data(), "workspace");
      }
      DBG_WAIT;

      succ = phi::dynload::flash_attn_bwd_with_bias_and_mask(
          static_cast<const void*>(q_ptr),
          static_cast<const void*>(k_ptr),
          static_cast<const void*>(v_ptr),
          static_cast<void*>(q_grad_ptr),
          static_cast<void*>(k_grad_ptr),
          static_cast<void*>(v_grad_ptr),
          static_cast<const void*>(fmha_out->data()),
          static_cast<const void*>(fmha_out_grad->data()),
          cu_seq_q.data<int32_t>(),
          cu_seq_k.data<int32_t>(),
          total_q_,
          total_k_,
          batch_size_,
          num_heads_,
          head_size_,
          max_seqlen_q_,
          max_seqlen_k_,
          p_dropout,
          softmax_scale,
          /*zero_tensors=*/false,
          /*is_causal=*/false,
          is_bf16,
          num_splits,
          softmax_lse->data(),
          softmax_d.data(),
          bias_d.data(),
          workspace.data(),
          &workspace_size,
          stream,
          seed,
          offset,
          src_mask ? temp_mask.data() : nullptr,
          nonbatched_bias ? temp_bias.data() : nullptr,
          temp_mask.dims().Get(),
          temp_bias.dims().Get());
      if (!succ) {
        PADDLE_THROW(phi::errors::External(phi::dynload::flash_attn_error()));
      }
      DBG_WAIT;

      if (nonbatched_bias) {
        // compare block reduce
        // auto size = attn_bias->sizes();
        // dbias = ds.reshape({ -1, size[0], size[1], size[2], size[3] }).sum({
        // 0 }); result.push_back( dbias );
        const auto temp_bias_num = temp_bias.numel();
        const auto bias_d_num = bias_d.numel();
        auto dbias_first_dim = bias_d_num / temp_bias_num;
        bias_d.Resize({dbias_first_dim,
                       temp_bias.dims()[0],
                       temp_bias.dims()[1],
                       temp_bias.dims()[2],
                       temp_bias.dims()[3]});
        phi::funcs::
            ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
                dev_ctx_,
                bias_d,
                nonbatched_bias_grad,
                kps::IdentityFunctor<T>(),
                {0});
      }
    } else {
      if (merge_qkv_) {
        PADDLE_ENFORCE_NOT_NULL(
            qkv_transpose_out,
            platform::errors::NotFound("The input qkv_transpose_out can not be "
                                       "nullptr when merge_qkv is true."));

        int64_t q_size = config->GetQuerySize();
        q_ptr = qkv_transpose_out->data<T>();
        k_ptr = q_ptr + q_size;
        v_ptr = k_ptr + q_size;

        qkv_transpose_out_grad.Resize(config->qkv_transpose_out_dims);
        AllocWithDebugInfo<T>(
            dev_ctx_, "qkv_transpose_out_grad", &qkv_transpose_out_grad);

        q_grad_ptr = qkv_transpose_out_grad.data<T>();
        k_grad_ptr = q_grad_ptr + q_size;
        v_grad_ptr = k_grad_ptr + q_size;
      } else {
        PADDLE_ENFORCE_NOT_NULL(
            q_transpose_out,
            platform::errors::NotFound("The input q_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));
        PADDLE_ENFORCE_NOT_NULL(
            k_transpose_out,
            platform::errors::NotFound("The input k_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));
        PADDLE_ENFORCE_NOT_NULL(
            v_transpose_out,
            platform::errors::NotFound("The input v_transpose_out can not be "
                                       "nullptr when merge_qkv is false."));

        q_ptr = q_transpose_out->data<T>();
        k_ptr = k_transpose_out->data<T>();
        v_ptr = v_transpose_out->data<T>();

        q_transpose_out_grad.Resize(config->q_transpose_out_dims);
        k_transpose_out_grad.Resize(config->kv_transpose_out_dims);
        v_transpose_out_grad.Resize(config->kv_transpose_out_dims);

        q_grad_ptr = dev_ctx_.Alloc<T>(
            &q_transpose_out_grad, q_transpose_out_grad.numel() * sizeof(T));
        k_grad_ptr = dev_ctx_.Alloc<T>(
            &k_transpose_out_grad, k_transpose_out_grad.numel() * sizeof(T));
        v_grad_ptr = dev_ctx_.Alloc<T>(
            &v_transpose_out_grad, v_transpose_out_grad.numel() * sizeof(T));
      }

      phi::DenseTensor softmax_out_grad;
      softmax_out_grad.Resize(config->softmax_out_dims);
      AllocWithDebugInfo<T>(dev_ctx_, "softmax_out_grad", &softmax_out_grad);

      int64_t gemm_batch_size =
          config->batch_size * config->seq_len_m * config->num_heads;
      {
        // Forward: fmha_out = transpose(qktv_out)
        phi::DenseTensor qktv_out_grad;
        qktv_out_grad.Resize(config->qktv_out_dims);
        AllocWithDebugInfo<T>(dev_ctx_, "qktv_out_grad", &qktv_out_grad);
        ComputeQKTVTransposeBackward(*fmha_out_grad, &qktv_out_grad);

        // Forward: qktv_out = BatchedGEMM(softmax_out, V)
        // Backward:
        //  V_grad = BatchedGEMM(softmax_out^T, qktv_out_grad) (dy = x^T * dout)
        int64_t gemm_m = config->m_size;
        int64_t gemm_n = config->head_dim;
        int64_t gemm_k = config->seq_len_r;

        const T* softmax_out_ptr = softmax_out->data<T>();
        const T* qktv_out_grad_ptr = qktv_out_grad.data<T>();
        ComputeBatchedGEMM(softmax_out_ptr,
                           qktv_out_grad_ptr,
                           v_grad_ptr,
                           true,
                           false,
                           gemm_m,
                           gemm_n,
                           gemm_k,
                           gemm_batch_size);

        // Backward: softmax_out_grad = qktv_out_grad * V^T (dx = dout * y^T)
        gemm_m = config->seq_len_r;
        gemm_n = config->m_size;
        gemm_k = config->head_dim;

        T* softmax_out_grad_ptr = softmax_out_grad.data<T>();
        ComputeBatchedGEMM(qktv_out_grad_ptr,
                           v_ptr,
                           softmax_out_grad_ptr,
                           false,
                           true,
                           gemm_m,
                           gemm_n,
                           gemm_k,
                           gemm_batch_size);
      }

      phi::DenseTensor* qk_out_grad = config->GetQKOutGrad(&softmax_out_grad);
      ComputeBiasMaskSoftmaxBackward(&softmax_out_grad,
                                     softmax_out,
                                     src_mask_grad,
                                     qk_out_grad,
                                     nonbatched_bias_grad);

      // Forward: qk_out = BatchedGEMM(Q, K^T)
      // Backward: k_grad = BatchedGEMM(qk_out_grad^T, Q) (dy = dout^t * x)
      int64_t gemm_m = config->m_size;
      int64_t gemm_n = config->head_dim;
      int64_t gemm_k = config->seq_len_r;
      T alpha = static_cast<T>(1.0 / sqrt(config->head_dim));

      T* qk_out_grad_ptr = qk_out_grad->data<T>();
      ComputeBatchedGEMM(qk_out_grad_ptr,
                         q_ptr,
                         k_grad_ptr,
                         true,
                         false,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size,
                         alpha);

      // Backward: q_grad = BatchedGEMM(qk_out_grad, K) (dx = dout * y)
      gemm_m = config->seq_len_r;
      gemm_n = config->head_dim;
      gemm_k = config->m_size;
      ComputeBatchedGEMM(qk_out_grad_ptr,
                         k_ptr,
                         q_grad_ptr,
                         false,
                         false,
                         gemm_m,
                         gemm_n,
                         gemm_k,
                         gemm_batch_size,
                         alpha);
    }

    if (merge_qkv_ || config->use_flash_attn) {
      phi::DenseTensor* qkv_out_grad = config->GetQKVOutGrad();
      ComputeQKVTransposeBackward(qkv_transpose_out_grad, qkv_out_grad);
    } else {
      phi::DenseTensor* q_out_grad = config->GetQueryOutGrad();
      phi::DenseTensor* k_out_grad = config->GetKeyOutGrad();
      phi::DenseTensor* v_out_grad = config->GetValueOutGrad();
      ComputeQKVTransposeBackward(q_transpose_out_grad,
                                  k_transpose_out_grad,
                                  v_transpose_out_grad,
                                  q_out_grad,
                                  k_out_grad,
                                  v_out_grad);
    }
  }

  void ComputeQKVTransposeForward(const phi::DenseTensor& q_out,
                                  const phi::DenseTensor& k_out,
                                  const phi::DenseTensor& v_out,
                                  phi::DenseTensor* q_transpose_out,
                                  phi::DenseTensor* k_transpose_out,
                                  phi::DenseTensor* v_transpose_out) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, q_out, perm, q_transpose_out);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, k_out, perm, k_transpose_out);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, v_out, perm, v_transpose_out);
  }

  void ComputeQKVTransposeBackward(const phi::DenseTensor& q_transpose_out_grad,
                                   const phi::DenseTensor& k_transpose_out_grad,
                                   const phi::DenseTensor& v_transpose_out_grad,
                                   phi::DenseTensor* q_out_grad,
                                   phi::DenseTensor* k_out_grad,
                                   phi::DenseTensor* v_out_grad) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, q_transpose_out_grad, perm, q_out_grad);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, k_transpose_out_grad, perm, k_out_grad);
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, v_transpose_out_grad, perm, v_out_grad);
  }

  // [batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim] ->
  //         [3, batch_size, seq_len_m, num_heads, seq_len_r, head_dim]
  void ComputeQKVTransposeForward(const phi::DenseTensor& qkv_out,
                                  phi::DenseTensor* qkv_transpose_out) {
    std::vector<int> perm = {3, 0, 1, 4, 2, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_out, perm, qkv_transpose_out);
  }

  // [batch_size, seq_len_m, seq_len_r, 3, num_heads, head_dim] ->
  //         [3, batch_size, seq_len_m, seq_len_r, num_heads, head_dim]
  void ComputeQKVTransposeForwardForFlashAttn(
      const phi::DenseTensor& qkv_out, phi::DenseTensor* qkv_transpose_out) {
    std::vector<int> perm = {3, 0, 1, 2, 4, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_out, perm, qkv_transpose_out);
  }

  void ComputeQKVTransposeBackward(
      const phi::DenseTensor& qkv_transpose_out_grad,
      phi::DenseTensor* qkv_out_grad) {
    std::vector<int> perm = {1, 2, 4, 0, 3, 5};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_transpose_out_grad, perm, qkv_out_grad);
  }

  // [batch_size, seq_len_m, num_head, seq_len_r, c] ->
  //         [batch_size, seq_len_m, seq_len_r, num_head, c]
  void ComputeQKTVTransposeForward(const phi::DenseTensor& qktv_out,
                                   phi::DenseTensor* fmha_out) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(dev_ctx_, qktv_out, perm, fmha_out);
  }

  void ComputeQKTVTransposeBackward(const phi::DenseTensor& fmha_out_grad,
                                    phi::DenseTensor* qktv_out_grad) {
    std::vector<int> perm = {0, 1, 3, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, fmha_out_grad, perm, qktv_out_grad);
  }

  // qk_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxForward(const phi::DenseTensor* nonbatched_bias,
                                     const phi::DenseTensor* src_mask,
                                     phi::DenseTensor* qk_out,
                                     phi::DenseTensor* softmax_out) {
    if (nonbatched_bias) {
      std::vector<const phi::DenseTensor*> ins = {
          qk_out, src_mask, nonbatched_bias};
      std::vector<phi::DenseTensor*> outs = {qk_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kTernary, T, T>(
          dev_ctx_, ins, &outs, -1, TernaryAddFunctor<T>());
    } else {
      std::vector<const phi::DenseTensor*> ins = {qk_out, src_mask};
      std::vector<phi::DenseTensor*> outs = {qk_out};
      phi::funcs::BroadcastKernel<phi::ElementwiseType::kBinary, T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    }
    phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *qk_out, -1, softmax_out);
  }

  // src_mask_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxBackward(const phi::DenseTensor* softmax_out_grad,
                                      const phi::DenseTensor* softmax_out,
                                      phi::DenseTensor* src_mask_grad,
                                      phi::DenseTensor* qk_out_grad,
                                      phi::DenseTensor* nonbatched_bias_grad) {
    PADDLE_ENFORCE_NOT_NULL(
        qk_out_grad,
        platform::errors::NotFound("The qk_out_grad can not be nullptr."));

    PADDLE_ENFORCE_EQ(qk_out_grad->dims(),
                      softmax_out->dims(),
                      platform::errors::InvalidArgument(
                          "The shape of qk_out_grad and softmax_out is "
                          "expected to be the same. But received qk_out_grad's "
                          "shape = %s, softmax_out's shape = %s.",
                          qk_out_grad->dims(),
                          softmax_out->dims()));

    PADDLE_ENFORCE_EQ(src_mask_grad,
                      nullptr,
                      platform::errors::InvalidArgument(
                          "src_mask_grad is expected to be nullptr."));

    phi::SoftmaxBackwardCUDAKernelDriver<T>(
        dev_ctx_, *softmax_out, *softmax_out_grad, -1, qk_out_grad);

    if (nonbatched_bias_grad) {
      // [batch_size, seq_len_m, num_heads, seq_len_r, m_size] ->
      //      [batch_size, 1, num_heads, seq_len_r, m_size]
      phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx_,
          *qk_out_grad,
          nonbatched_bias_grad,
          kps::IdentityFunctor<T>(),
          {1});
    }
  }

 private:
  void ComputeBatchedGEMM(const phi::DDim& a_dims,
                          const phi::DDim& b_dims,
                          const T* a_ptr,
                          const T* b_ptr,
                          T* c_ptr,
                          bool trans_a,
                          bool trans_b,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          int64_t batch_size,
                          T alpha = static_cast<T>(1.0),
                          T beta = static_cast<T>(0.0)) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;
    int64_t stride_out = m * n;

    phi::funcs::MatmulPlanner matmul_planner(
        vectorize(a_dims),
        vectorize(b_dims),
        trans_a,
        trans_b,
        phi::CppTypeToDataType<T>::Type(),
        phi::funcs::MatmulFusedType::kMatmul);

    using blaslt = phi::funcs::MatmulWithCublasLt<T>;
    blaslt::RunWithBatch(dev_ctx_,
                         a_ptr,
                         b_ptr,
                         c_ptr,
                         m,
                         n,
                         k,
                         trans_a,
                         trans_b,
                         batch_size,
                         stride_a,
                         stride_b,
                         stride_out,
                         &matmul_planner,
                         alpha,
                         beta);
  }

  void ComputeBatchedGEMM(const T* a_ptr,
                          const T* b_ptr,
                          T* c_ptr,
                          bool trans_a,
                          bool trans_b,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          int64_t batch_size,
                          T alpha = static_cast<T>(1.0),
                          T beta = static_cast<T>(0.0)) {
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;

    CBLAS_TRANSPOSE cblas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE cblas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    blas.BatchedGEMM(cblas_trans_a,
                     cblas_trans_b,
                     m,
                     n,
                     k,
                     alpha,
                     a_ptr,
                     b_ptr,
                     beta,
                     c_ptr,
                     batch_size,
                     stride_a,
                     stride_b);
  }

  const phi::GPUContext& dev_ctx_;
  bool merge_qkv_;
};

}  // namespace operators
}  // namespace paddle
