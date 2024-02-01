// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

namespace phi {
namespace fusion {

__global__ void set_rng_state(std::pair<uint64_t, uint64_t> seed_offset,
                              int64_t *rng_state_ptr) {
  rng_state_ptr[0] = static_cast<int64_t>(seed_offset.first);
  rng_state_ptr[1] = static_cast<int64_t>(seed_offset.second);
}

template <typename T, typename Context>
void FusedDotProductAttentionKernel(const Context &dev_ctx,
                                    const DenseTensor &q,
                                    const DenseTensor &k,
                                    const DenseTensor &v,
                                    const DenseTensor &mask,
                                    float scaling_factor,
                                    float dropout_probability,
                                    bool is_training,
                                    bool is_causal_masking,
                                    DenseTensor *out,
                                    DenseTensor *softmax_out,
                                    DenseTensor *rng_state) {
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    phi::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8906,
                    phi::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8906, "
                        "but got %d.",
                        cudnn_version));

  // allocate output variables
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(softmax_out);
  dev_ctx.template Alloc<int64_t>(rng_state);

  // get handles
  auto handle = dev_ctx.cudnn_handle();

  auto tensor_dtype = phi::backends::gpu::ToCudnnDataType(q.dtype());
  bool is_type_supported =
      (tensor_dtype == CUDNN_DATA_HALF || tensor_dtype == CUDNN_DATA_BFLOAT16);
  PADDLE_ENFORCE_EQ(
      is_type_supported,
      true,
      phi::errors::InvalidArgument(
          "cuDNN fused attention Only supports FP16/BF16 currently"));
  auto mha_layout = MHA_Layout::NOT_INTERLEAVED;
  auto bias_type = MHA_Bias_Type::NO_BIAS;
  auto mask_type = is_causal_masking ? MHA_Mask_Type::CAUSAL_MASK
                                     : MHA_Mask_Type::PADDING_MASK;
  std::vector<cudnn_frontend::Operation const *> all_ops;
  std::vector<cudnn_frontend::Operation> ops;
  std::set<std::pair<uint64_t, void *>> data_ptrs;

  // q dim: {b, s_q, h, d};
  // k,v dim: {b, s_kv, h, d};
  auto batch_size = q.dims()[0];
  auto q_seq_len = q.dims()[1];
  auto num_heads = q.dims()[2];
  auto head_size = q.dims()[3];
  auto kv_seq_len = k.dims()[1];

  // only support seqlen >= 64 and seqlen <= 512 and seqlen % 64 == 0
  // currently
  bool can_divide_by_64 = (q_seq_len % 64 == 0 && kv_seq_len % 64 == 0);
  PADDLE_ENFORCE_EQ(can_divide_by_64,
                    true,
                    phi::errors::InvalidArgument(
                        "cuDNN FMHA only supports sequence length >= 64,"
                        "and sequence length % 64 == 0, "
                        "but got sequence length: %d and %d.",
                        q_seq_len,
                        kv_seq_len));

  auto gen_cuda = dev_ctx.GetGenerator();
  // threads per CTA = 128
  auto rng_elts_per_thread = (q_seq_len * kv_seq_len + 128 - 1) / 128;
  auto seed_offset = gen_cuda->IncrementOffset(rng_elts_per_thread);
  set_rng_state<<<1, 1, 0, dev_ctx.stream()>>>(
      seed_offset, static_cast<int64_t *>(rng_state->data<int64_t>()));

  void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q.data<T>()));
  void *k_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(k.data<T>()));
  void *v_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(v.data<T>()));
  void *out_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(out->data<T>()));
  void *softmax_out_dev_ptr =
      reinterpret_cast<void *>(const_cast<float *>(softmax_out->data<float>()));
  void *bias_dev_ptr = nullptr;
  void *mask_dev_ptr =
      reinterpret_cast<void *>(const_cast<int32_t *>(mask.data<int32_t>()));
  // rng_state: {seed, offset}
  void *seed_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state->data<int64_t>()));
  void *offset_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state->data<int64_t>()) + 1);

  fused_attn_arbitrary_seqlen_fwd(batch_size,
                                  num_heads,
                                  q_seq_len,
                                  kv_seq_len,
                                  head_size,
                                  is_training,
                                  scaling_factor,
                                  dropout_probability,
                                  mha_layout,
                                  mask_type,
                                  q_dev_ptr,
                                  k_dev_ptr,
                                  v_dev_ptr,
                                  softmax_out_dev_ptr,
                                  out_dev_ptr,
                                  mask_dev_ptr,
                                  seed_dev_ptr,
                                  offset_dev_ptr,
                                  tensor_dtype,
                                  dev_ctx.stream(),
                                  handle);
}

template <typename T, typename Context>
void FusedDotProductAttentionGradKernel(const Context &dev_ctx,
                                        const DenseTensor &q,
                                        const DenseTensor &k,
                                        const DenseTensor &v,
                                        const DenseTensor &O,
                                        const DenseTensor &softmax_out,
                                        const DenseTensor &rng_state,
                                        const DenseTensor &mask,
                                        const DenseTensor &dO,
                                        float scaling_factor,
                                        float dropout_probability,
                                        bool is_causal_masking,
                                        DenseTensor *q_grad,
                                        DenseTensor *k_grad,
                                        DenseTensor *v_grad) {
  PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                    80,
                    phi::errors::PreconditionNotMet(
                        "This op only supports Ampere and later devices, "
                        "but got compute capability: %d.",
                        dev_ctx.GetComputeCapability()));
  auto cudnn_version = phi::backends::gpu::DnnVersion();
  PADDLE_ENFORCE_GE(cudnn_version,
                    8906,
                    phi::errors::PreconditionNotMet(
                        "This op only supports CUDNN version >= 8906, "
                        "but got %d.",
                        cudnn_version));

  // allocate output variables
  dev_ctx.template Alloc<T>(q_grad);
  dev_ctx.template Alloc<T>(k_grad);
  dev_ctx.template Alloc<T>(v_grad);

  // get handles
  auto handle = dev_ctx.cudnn_handle();

  auto tensor_dtype = phi::backends::gpu::ToCudnnDataType(q.dtype());
  bool support_type =
      (tensor_dtype == CUDNN_DATA_HALF || tensor_dtype == CUDNN_DATA_BFLOAT16);
  PADDLE_ENFORCE_EQ(support_type,
                    true,
                    phi::errors::InvalidArgument(
                        "cuDNN FMHA Only supports FP16/BF16 currently"));
  auto mha_layout = MHA_Layout::NOT_INTERLEAVED;
  auto mask_type = is_causal_masking ? MHA_Mask_Type::CAUSAL_MASK
                                     : MHA_Mask_Type::PADDING_MASK;
  std::vector<cudnn_frontend::Operation const *> all_ops;
  std::vector<cudnn_frontend::Operation> ops;
  std::set<std::pair<uint64_t, void *>> data_ptrs;

  // q dim: {b, s_q, h, d};
  // k, v dim: {b, s_kv, h, d};
  auto batch_size = q.dims()[0];
  auto q_seq_len = q.dims()[1];
  auto num_heads = q.dims()[2];
  auto head_size = q.dims()[3];
  auto kv_seq_len = k.dims()[1];

  void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q.data<T>()));
  void *k_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(k.data<T>()));
  void *v_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(v.data<T>()));
  void *dq_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(q_grad->data<T>()));
  void *dk_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(k_grad->data<T>()));
  void *dv_dev_ptr =
      reinterpret_cast<void *>(const_cast<T *>(v_grad->data<T>()));
  void *o_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(O.data<T>()));
  void *do_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dO.data<T>()));
  void *softmax_out_dev_ptr =
      reinterpret_cast<void *>(const_cast<float *>(softmax_out.data<float>()));
  void *mask_dev_ptr =
      reinterpret_cast<void *>(const_cast<int32_t *>(mask.data<int32_t>()));
  void *seed_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state.data<int64_t>()));
  void *offset_dev_ptr = reinterpret_cast<void *>(
      const_cast<int64_t *>(rng_state.data<int64_t>()) + 1);

  fused_attn_arbitrary_seqlen_bwd(batch_size,
                                  num_heads,
                                  q_seq_len,
                                  kv_seq_len,
                                  head_size,
                                  scaling_factor,
                                  dropout_probability,
                                  mha_layout,
                                  mask_type,
                                  q_dev_ptr,
                                  k_dev_ptr,
                                  v_dev_ptr,
                                  o_dev_ptr,
                                  softmax_out_dev_ptr,
                                  dq_dev_ptr,
                                  dk_dev_ptr,
                                  dv_dev_ptr,
                                  do_dev_ptr,
                                  mask_dev_ptr,
                                  seed_dev_ptr,
                                  offset_dev_ptr,
                                  tensor_dtype,
                                  dev_ctx.stream(),
                                  handle,
                                  false);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dot_product_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDotProductAttentionKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);  // mask
}

PD_REGISTER_KERNEL(fused_dot_product_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDotProductAttentionGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(6).SetDataType(phi::DataType::INT32);  // mask
}
