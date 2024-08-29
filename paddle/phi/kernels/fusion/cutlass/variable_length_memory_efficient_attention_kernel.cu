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

#include "paddle/phi/kernels/fusion/cutlass/variable_length_memory_efficient_attention.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void MultiHeadAttentionVariableForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const DenseTensor& seq_lens,
    const DenseTensor& kv_seq_lens,
    const paddle::optional<DenseTensor>& mask,
    const float scale,
    const bool causal,
    const int pre_cache_length,
    DenseTensor* output) {
  ctx.template Alloc<T>(output);
  Params params{};
  // [B, N, S, H]
  params.seq_lens = seq_lens.data<int>();
  params.kv_seq_lens = kv_seq_lens.data<int>();

  params.num_batches = query.dims()[0];
  params.num_heads = query.dims()[1];
  params.kv_num_heads = key.dims()[1];
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
  params.pre_cache_length = pre_cache_length;

  if (mask) {
    // [B, 1, S, D]
    auto mask_tensor = mask.get();
    int64_t mask_num_heads = mask_tensor.dims()[1];
    params.ldm = mask_tensor.dims()[3];
    params.ElementM = mask_tensor.dims()[2] * mask_tensor.dims()[3];
    params.mask_ptr = mask_tensor.data();
    params.mask_broadcast_head = mask_num_heads == 1 ? true : false;
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
    if (mask && reinterpret_cast<uintptr_t>(params.mask_ptr) % 16 == 0 &&
        params.ldm % (16 / sizeof(T)) == 0 && !KernelType::kMaskIsAligned) {
      return;
    }
    if (mask &&
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
      common::errors::InvalidArgument("the kernel should not be launched"));
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(variable_length_memory_efficient_attention,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::MultiHeadAttentionVariableForwardKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}
