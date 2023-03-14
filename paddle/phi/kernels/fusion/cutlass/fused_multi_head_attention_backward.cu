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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/api/include/tensor_operants.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen/memory_efficient_attention.h"

#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename T, typename Context>
void MultiHeadAttentionBackwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& cu_seqlens_q,
    const paddle::optional<DenseTensor>& cu_seqlens_k,
    const DenseTensor& output,
    const paddle::optional<DenseTensor>& logsumexp,
    const DenseTensor& seed_and_offset,
    const DenseTensor& output_grad,
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
    const bool causal,
    const double dropout_p,
    const float scale,
    DenseTensor* query_grad,
    DenseTensor* key_grad,
    DenseTensor* value_grad,
    DenseTensor* bias_grad) {
  bool kernel_launched = false;

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    // ndim
    PADDLE_ENFORCE_EQ(query.dims().size(), output_grad.dims().size());
    PADDLE_ENFORCE_EQ(query.dims().size(), key.dims().size());
    PADDLE_ENFORCE_EQ(query.dims().size(), value.dims().size());
    PADDLE_ENFORCE_EQ(query.dims().size(), 4);

    // batch size
    PADDLE_ENFORCE_EQ(query.dims()[0], output_grad.dims()[0]);
    PADDLE_ENFORCE_EQ(query.dims()[0], key.dims()[0]);
    PADDLE_ENFORCE_EQ(query.dims()[0], value.dims()[0]);

    // seqlen
    PADDLE_ENFORCE_EQ(key.dims()[1], value.dims()[1]);
    PADDLE_ENFORCE_EQ(query.dims()[1], output_grad.dims()[1]);

    // Num heads
    PADDLE_ENFORCE_EQ(query.dims()[2], key.dims()[2]);
    PADDLE_ENFORCE_EQ(query.dims()[2], value.dims()[2]);
    PADDLE_ENFORCE_EQ(query.dims()[2], output_grad.dims()[2]);

    // Embedding per head
    PADDLE_ENFORCE_EQ(query.dims()[3], key.dims()[3]);
    PADDLE_ENFORCE_EQ(value.dims()[3], output_grad.dims()[3]);

    PADDLE_ENFORCE_EQ(
        ((cu_seqlens_q && cu_seqlens_k) || (!cu_seqlens_q && !cu_seqlens_k)),
        true);
    PADDLE_ENFORCE_EQ((!(cu_seqlens_q && bias)), true);

    const auto& k_dims = key.dims();
    const auto& q_dims = query.dims();
    const auto& v_dims = value.dims();

    int64_t max_seqlen_q_tmp, max_seqlen_k_tmp;
    if (cu_seqlens_q) {
      PADDLE_ENFORCE_EQ(cu_seqlens_q.get().dtype(), DataType::INT32);
      PADDLE_ENFORCE_EQ(cu_seqlens_k.get().dtype(), DataType::INT32);
      PADDLE_ENFORCE_EQ((cu_seqlens_q.get().dims().size() == 1 &&
                         cu_seqlens_k.get().dims().size() == 1),
                        true);
      max_seqlen_q_tmp = max_seqlen_q.to<int64_t>();
      max_seqlen_k_tmp = max_seqlen_k.to<int64_t>();
      VLOG(3) << "max_seqlen_q_tmp" << max_seqlen_q_tmp;
      VLOG(3) << "max_seqlen_k_tmp" << max_seqlen_k_tmp;
      PADDLE_ENFORCE_EQ(cu_seqlens_q.get().dims()[0],
                        cu_seqlens_k.get().dims()[0]);
      PADDLE_ENFORCE_EQ(q_dims[0], 1);
      PADDLE_ENFORCE_LT(0, max_seqlen_q_tmp);
      PADDLE_ENFORCE_LT(0, max_seqlen_k_tmp);
      PADDLE_ENFORCE_LE(max_seqlen_q_tmp, q_dims[1]);
      PADDLE_ENFORCE_LE(max_seqlen_k_tmp, k_dims[1]);
    } else {
      max_seqlen_q_tmp = q_dims[1];
      max_seqlen_k_tmp = k_dims[1];
    }
    VLOG(3) << "max_seqlen_q_tmp has been set " << max_seqlen_q_tmp
            << " max_seqlen_k_tmp " << max_seqlen_k_tmp;

    ctx.template Alloc<T>(query_grad);
    ctx.template Alloc<T>(key_grad);
    ctx.template Alloc<T>(value_grad);
    if (bias && bias_grad) {
      ctx.template Alloc<T>(bias_grad);
    }

    auto use_dropout = dropout_p != 0.0;
    const auto maxK = std::max(q_dims[3], v_dims[3]);
    int compute_capacity = ctx.GetComputeCapability();
    const auto max_shmem =
        getMaximumSharedMemoryPerBlockKb(compute_capacity) * 1024;

    using KernelType = decltype(k_);
    using scalar_t = typename KernelType::scalar_t;
    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (KernelType::kMaxK < maxK) {
      return;
    }
    // Dropout must be supported if we need it
    if (use_dropout && !KernelType::kApplyDropout) {
      return;
    }
    // Alignment
    if ((q_dims[3] % KernelType::kMinimumAlignment) ||
        (k_dims[3] % KernelType::kMinimumAlignment) ||
        (v_dims[3] % KernelType::kMinimumAlignment)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename KernelType::SharedStorage);
    if (smem_bytes > max_shmem) {
      return;
    }

    VLOG(3) << "smem has been set" << smem_bytes << " " << max_shmem;

    kernel_launched = true;

    DenseTensor delta;
    if (KernelType::kKernelComputesDelta) {
      phi::EmptyKernel<float, Context>(
          ctx,
          {output.dims()[0], output.dims()[2], output.dims()[1]},
          output.dtype(),
          &delta);
    } else {
      DenseTensor output_grad_tmp =
          output_grad.dtype() == DataType::FLOAT32
              ? output_grad
              : phi::Cast<T, Context>(ctx, output_grad, DataType::FLOAT32);
      DenseTensor output_tmp =
          output.dtype() == DataType::FLOAT32
              ? output
              : phi::Cast<T, Context>(ctx, output, DataType::FLOAT32);
      DenseTensor delta_mul =
          phi::Multiply<float, Context>(ctx, output_grad_tmp, output_tmp);

      DenseTensor delta_sum;
      phi::EmptyKernel<float, Context>(
          ctx,
          {delta_mul.dims()[0], delta_mul.dims()[1], delta_mul.dims()[2]},
          DataType::FLOAT32,
          &delta_sum);
      phi::SumKernel<float, Context>(
          ctx, delta_mul, {-1}, delta_mul.dtype(), false, &delta_sum);
      phi::EmptyKernel<float, Context>(
          ctx,
          {delta_mul.dims()[0], delta_mul.dims()[2], delta_mul.dims()[1]},
          DataType::FLOAT32,
          &delta);
      phi::TransposeKernel<float, Context>(ctx, delta_sum, {0, 2, 1}, &delta);
    }

    PADDLE_ENFORCE_EQ(delta.dims()[0], query.dims()[0]);
    PADDLE_ENFORCE_EQ(delta.dims()[1], query.dims()[2]);
    PADDLE_ENFORCE_EQ(delta.dims()[2], query.dims()[1]);

    VLOG(3) << "delta has been set" << delta.data();

    typename KernelType::Params p;
    p.query_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(query.data()));
    p.key_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(key.data()));
    p.value_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(value.data()));
    if (logsumexp && logsumexp.get().data() != 0) {
      p.logsumexp_ptr = const_cast<float*>(
          reinterpret_cast<const float*>(logsumexp.get().data()));
      VLOG(3) << "logsumexp_ptr" << p.logsumexp_ptr;
    }
    p.output_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(output.data()));
    p.grad_output_ptr = const_cast<scalar_t*>(
        reinterpret_cast<const scalar_t*>(output_grad.data()));
    p.grad_query_ptr = const_cast<scalar_t*>(
        reinterpret_cast<const scalar_t*>(query_grad->data()));
    p.grad_key_ptr = const_cast<scalar_t*>(
        reinterpret_cast<const scalar_t*>(key_grad->data()));
    p.grad_value_ptr = const_cast<scalar_t*>(
        reinterpret_cast<const scalar_t*>(value_grad->data()));
    p.delta_ptr =
        const_cast<float*>(reinterpret_cast<const float*>(delta.data()));
    p.head_dim = q_dims[3];
    p.head_dim_value = v_dims[3];

    p.num_queries = max_seqlen_q_tmp;
    p.num_keys = max_seqlen_k_tmp;
    p.num_batches = cu_seqlens_q ? cu_seqlens_q.get().dims()[0] - 1 : q_dims[0];
    p.num_heads = q_dims[2];
    p.causal = causal;

    if (scale < 0) {
      p.scale = static_cast<float>(1.0 / std::sqrt(p.head_dim));
    } else {
      p.scale = scale;
    }
    VLOG(3) << "p.scale" << p.scale;

    if (cu_seqlens_q) {
      p.cu_seqlens_q_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(cu_seqlens_q->data()));
      p.cu_seqlens_k_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(cu_seqlens_k->data()));
      VLOG(3) << "p.cu_seqlens_q_ptr" << p.cu_seqlens_q_ptr;
    }

    if (logsumexp && logsumexp.get().data() != 0) {
      p.lse_strideH = logsumexp.get().dims()[2];
      p.lse_strideB = logsumexp.get().dims()[1] * logsumexp.get().dims()[2];
      VLOG(3) << "p.lse_strideH " << p.lse_strideH;
    }

    p.gO_strideH = output_grad.dims()[3];
    p.gO_strideM = output_grad.dims()[3] * output_grad.dims()[2];
    p.gO_strideB =
        output_grad.dims()[3] * output_grad.dims()[2] * output_grad.dims()[1];

    p.o_strideH = output.dims()[3];
    p.o_strideB = output.dims()[3] * output.dims()[2] * output.dims()[1];

    p.gQ_strideH = query_grad->dims()[3];
    p.gK_strideH = key_grad->dims()[3];
    p.gV_strideH = value_grad->dims()[3];
    p.gQ_strideB =
        query_grad->dims()[3] * query_grad->dims()[2] * query_grad->dims()[1];
    p.gK_strideB =
        key_grad->dims()[3] * key_grad->dims()[2] * key_grad->dims()[1];
    p.gV_strideB =
        value_grad->dims()[3] * value_grad->dims()[2] * value_grad->dims()[1];
    p.gQKV_strideM_multiplier = 1;
    PADDLE_ENFORCE_EQ(p.gQ_strideM(),
                      query_grad->dims()[3] * query_grad->dims()[2]);
    PADDLE_ENFORCE_EQ(p.gK_strideM(),
                      key_grad->dims()[3] * key_grad->dims()[2]);
    PADDLE_ENFORCE_EQ(p.gV_strideM(),
                      value_grad->dims()[3] * value_grad->dims()[2]);

    p.q_strideB = query.dims()[3] * query.dims()[2] * query.dims()[1];
    p.k_strideB = key.dims()[3] * key.dims()[2] * key.dims()[1];
    p.v_strideB = value.dims()[3] * value.dims()[2] * value.dims()[1];
    p.q_strideM = query.dims()[3] * query.dims()[2];
    p.k_strideM = key.dims()[3] * key.dims()[2];
    p.v_strideM = value.dims()[3] * value.dims()[2];
    p.q_strideH = query.dims()[3];
    p.k_strideH = key.dims()[3];
    p.v_strideH = value.dims()[3];

    p.delta_strideH = delta.dims()[2];
    p.delta_strideB = delta.dims()[2] * delta.dims()[1];

    if (bias) {
      p.bias_ptr = const_cast<scalar_t*>(
          reinterpret_cast<const scalar_t*>(bias.get().data()));
      p.bias_strideB = q_dims[2] * q_dims[1] * k_dims[1];
      p.bias_strideH = q_dims[1] * k_dims[1];
      p.bias_strideM = k_dims[1];
      VLOG(3) << "p.bias_ptr" << p.bias_ptr;
      if (bias_grad) {
        p.grad_bias_ptr = const_cast<scalar_t*>(
            reinterpret_cast<const scalar_t*>(bias_grad->data()));
        p.gB_strideB = q_dims[2] * q_dims[1] * k_dims[1];
        p.gB_strideH = q_dims[1] * k_dims[1];
        p.gB_strideM = k_dims[1];
        VLOG(3) << "p.grad_bias_ptr" << p.grad_bias_ptr;
      } else {
        p.grad_bias_ptr = nullptr;
      }
    } else {
      p.bias_ptr = nullptr;
      p.grad_bias_ptr = nullptr;
    }
    if (dropout_p != 0) {
      int64_t* seed_and_offset_ptr = const_cast<int64_t*>(
          reinterpret_cast<const int64_t*>(seed_and_offset.data<int64_t>()));
      p.seed = seed_and_offset_ptr[0];
      p.offset = seed_and_offset_ptr[1];
      p.dropout_prob = dropout_p;
      VLOG(3) << "seed_and_offset_ptr " << seed_and_offset_ptr;
      VLOG(3) << "p.seed " << p.seed << " " << p.offset;
      VLOG(3) << "p.dropout_prob " << p.dropout_prob;
    }

    int64_t size_bytes = p.workspace_size();
    paddle::memory::AllocationPtr temp_workspace{nullptr};
    VLOG(3) << "size_bytes " << size_bytes;
    temp_workspace = paddle::memory::Alloc(
        ctx.GetPlace(),
        size_bytes,
        phi::Stream(reinterpret_cast<phi::StreamId>(ctx.stream())));
    if (size_bytes) {
      p.workspace = reinterpret_cast<typename KernelType::output_accum_t*>(
          temp_workspace->ptr());
      VLOG(3) << "p.workspace" << p.workspace;
    }
    VLOG(3) << "temp_workspace has been set";

    if (smem_bytes > 0xc000) {
      PADDLE_ENFORCE_GPU_SUCCESS(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    KernelType::check_supported(p);
    VLOG(3) << "Kernel launched with func : " << typeid(kernel_fn).name()
            << " block dim " << p.getBlocksGrid() << " thread dim "
            << p.getThreadsGrid();
    kernel_fn<<<p.getBlocksGrid(),
                p.getThreadsGrid(),
                smem_bytes,
                ctx.stream()>>>(p);
  };
  dispatch_cutlass_backward<T>(ctx, launchKernel);
  PADDLE_ENFORCE_EQ(kernel_launched, true);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    fused_multihead_attention_grad,
    GPU,
    ALL_LAYOUT,
    phi::fusion::cutlass_internal::MultiHeadAttentionBackwardKernel,
    float) {
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
