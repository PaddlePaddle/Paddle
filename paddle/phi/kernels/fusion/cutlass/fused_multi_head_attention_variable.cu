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

#include "paddle/phi/kernels/fusion/cutlass/fused_multi_head_attention/autogen/cutlass_forward.h"
#include "paddle/phi/kernels/fusion/fused_multihead_attention_variable_kernel.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const DenseTensor& seq_lens,
    const paddle::optional<DenseTensor>& mask,
    const float scale,
    const bool causal,
    DenseTensor* output) {
  ctx.template Alloc<T>(output);
  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens.data<int>();

  params.num_batches = query.dims()[0];
  params.num_heads = query.dims()[1];
  params.query_seq_len = query.dims()[2];
  params.head_size = query.dims()[3];
  params.key_value_seq_len = key.dims()[2];
  params.value_head_size = value.dims()[3];

  params.datatype = query.dtype();
  params.query_ptr = query.data();
  params.key_ptr = key.data();
  params.value_ptr = value.data();

  params.output_ptr = output->data();

  params.ldq = params.head_size;
  params.ldk = params.head_size;
  params.ldv = params.value_head_size;
  params.ldo = params.value_head_size;

  params.ElementQ = params.query_seq_len * params.head_size;
  params.ElementK = params.key_value_seq_len * params.head_size;
  params.ElementV = params.key_value_seq_len * params.value_head_size;
  params.ElementO = params.query_seq_len * params.value_head_size;

  params.scale = scale;
  params.causal = causal;

  if (mask) {
    // [B, 1, S, D]
    auto mask_tensor = mask.get();
    params.ldm = mask_tensor.dims()[3];
    params.ElementM = mask_tensor.dims()[2] * mask_tensor.dims()[3];
    params.mask_ptr = mask_tensor.data();
    params.mask_broadcast_row = false;
  }

  bool kernel_launched = false;

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    using KernelType = decltype(k_);
    if (kernel_launched) {
      return;
    }
    if (mask && !KernelType::kAddMask) {
      return;
    }
    if (!mask && KernelType::kAddMask) {
      return;
    }
    if (KernelType::kMaskBroadcastRow) {
      // not support mask_broad_cast
      return;
    }
    if (params.mask_ptr &&
        reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
        params.ldm % (16 / sizeof(T)) == 0 && !KernelType::kMaskIsAligned) {
      return;
    }
    if (params.mask_ptr &&
        !(reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
          params.ldm % (16 / sizeof(T)) == 0) &&
        KernelType::kMaskIsAligned) {
      return;
    }
    if (KernelType::kSingleValueIteration &&
        KernelType::kKeysPerBlock < params.value_head_size) {
      return;
    }
    if (KernelType::kKeysPerBlock == 64 && params.value_head_size > 64) {
      return;
    }
    if (params.head_size % KernelType::MM0::kAlignmentA) {
      return;
    }
    kernel_launched = true;
    kernel_fn(k_, params, ctx);
  };
  dispatch_cutlass_forward<T, decltype(launchKernel)>(ctx, launchKernel);
  PADDLE_ENFORCE_EQ(
      kernel_launched,
      true,
      phi::errors::InvalidArgument("the kernel should not be launched"));
}

template <typename T, typename Context>
void MultiHeadAttentionVariableWrapper(const Context& ctx,
                                       T* query,
                                       T* key,
                                       T* value,
                                       const int* seq_lens,
                                       const phi::DenseTensor* mask_tensor,
                                       const float scale,
                                       const bool causal,
                                       const int64_t batch_size,
                                       const int64_t num_heads,
                                       const int64_t seq_len,
                                       const int64_t out_seq_len,
                                       const int64_t head_size,
                                       const int64_t value_head_size,
                                       const int prompt_num,
                                       T* output) {
  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens;

  params.num_batches = batch_size;
  params.num_heads = num_heads;
  params.query_seq_len = seq_len;
  params.head_size = head_size;
  params.key_value_seq_len = out_seq_len;
  params.value_head_size = value_head_size;

  if (std::is_same<T, phi::dtype::float16>::value) {
    params.datatype = DataType::FLOAT16;
  } else if (std::is_same<T, phi::dtype::bfloat16>::value) {
    params.datatype = DataType::BFLOAT16;
  } else {
    params.datatype = DataType::FLOAT32;
  }
  params.query_ptr = query;
  params.key_ptr = key;
  params.value_ptr = value;

  params.prompt_num = prompt_num;

  params.output_ptr = output;

  params.ldq = params.head_size;
  params.ldk = params.head_size;
  params.ldv = params.value_head_size;
  params.ldo = params.value_head_size;

  params.ElementQ = params.query_seq_len * params.head_size;
  params.ElementK = params.key_value_seq_len * params.head_size;
  params.ElementV = params.key_value_seq_len * params.value_head_size;
  params.ElementO = params.query_seq_len * params.value_head_size;

  params.scale = scale;
  params.causal = causal;

  if (mask_tensor) {
    // [B, 1, S, D]
    params.mask_broadcast_row = false;
    params.mask_ptr = mask_tensor->data<T>();
    params.ldm = mask_tensor->dims()[3];
    params.ElementM = mask_tensor->dims()[2] * mask_tensor->dims()[3];
  } else {
    params.mask_broadcast_row = false;
    params.mask_ptr = nullptr;
  }

  bool kernel_launched = false;

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    using KernelType = decltype(k_);
    if (kernel_launched) {
      return;
    }
    if (mask_tensor && !KernelType::kAddMask) {
      return;
    }
    if (!mask_tensor && KernelType::kAddMask) {
      return;
    }
    if (KernelType::kMaskBroadcastRow) {
      // not support mask_broad_cast
      return;
    }
    if (params.mask_ptr &&
        reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
        params.ldm % (16 / sizeof(T)) == 0 && !KernelType::kMaskIsAligned) {
      return;
    }
    if (params.mask_ptr &&
        !(reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
          params.ldm % (16 / sizeof(T)) == 0) &&
        KernelType::kMaskIsAligned) {
      return;
    }
    if (KernelType::kSingleValueIteration &&
        KernelType::kKeysPerBlock < params.value_head_size) {
      return;
    }
    if (KernelType::kKeysPerBlock == 64 && params.value_head_size > 64) {
      return;
    }
    if (params.head_size % KernelType::MM0::kAlignmentA) {
      return;
    }
    kernel_launched = true;
    kernel_fn(k_, params, ctx);
  };
  dispatch_cutlass_forward<T, decltype(launchKernel)>(ctx, launchKernel);
  PADDLE_ENFORCE_EQ(
      kernel_launched,
      true,
      phi::errors::InvalidArgument("the kernel should not be launched"));
}

template void MultiHeadAttentionVariableWrapper(const phi::GPUContext& ctx,
                                                phi::dtype::bfloat16* query,
                                                phi::dtype::bfloat16* key,
                                                phi::dtype::bfloat16* value,
                                                const int* seq_lens,
                                                const phi::DenseTensor* mask,
                                                const float scale,
                                                const bool causal,
                                                const int64_t batch_size,
                                                const int64_t num_heads,
                                                const int64_t seq_len,
                                                const int64_t out_seq_len,
                                                const int64_t head_size,
                                                const int64_t value_head_size,
                                                const int prompt_num,
                                                phi::dtype::bfloat16* output);

template void MultiHeadAttentionVariableWrapper(const phi::GPUContext& ctx,
                                                phi::dtype::float16* query,
                                                phi::dtype::float16* key,
                                                phi::dtype::float16* value,
                                                const int* seq_lens,
                                                const phi::DenseTensor* mask,
                                                const float scale,
                                                const bool causal,
                                                const int64_t batch_size,
                                                const int64_t num_heads,
                                                const int64_t seq_len,
                                                const int64_t out_seq_len,
                                                const int64_t head_size,
                                                const int64_t value_head_size,
                                                const int prompt_num,
                                                phi::dtype::float16* output);

template void MultiHeadAttentionVariableWrapper(const phi::GPUContext& ctx,
                                                float* query,
                                                float* key,
                                                float* value,
                                                const int* seq_lens,
                                                const phi::DenseTensor* mask,
                                                const float scale,
                                                const bool causal,
                                                const int64_t batch_size,
                                                const int64_t num_heads,
                                                const int64_t seq_len,
                                                const int64_t out_seq_len,
                                                const int64_t head_size,
                                                const int64_t value_head_size,
                                                const int prompt_num,
                                                float* output);

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_multihead_attention_variable,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiHeadAttentionVariableForwardKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}
