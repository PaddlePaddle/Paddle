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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen/memory_efficient_attention.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

template <typename T, typename Context>
void MultiHeadAttentionForwardKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& cu_seqlens_q,
    const paddle::optional<DenseTensor>& cu_seqlens_k,
    const paddle::optional<DenseTensor>& seqstart_q,
    const paddle::optional<DenseTensor>& seqstart_k,
    const paddle::optional<DenseTensor>& causal_diagonal,
    const paddle::optional<DenseTensor>& seqlen_k,
    const Scalar& max_seqlen_q,
    const Scalar& max_seqlen_k,
    const bool causal,
    const double dropout_p,
    const float scale,
    const bool is_test,
    DenseTensor* output,
    DenseTensor* logsumexp,
    DenseTensor* seed_and_offset) {
  phi::Dim<1> seed_dims;
  seed_dims[0] = 2;
  seed_and_offset->Resize(seed_dims);
  ctx.template HostAlloc<int64_t>(seed_and_offset);

  VLOG(3) << seed_and_offset->data();

  int compute_capacity = ctx.GetComputeCapability();
  const auto max_shmem =
      getMaximumSharedMemoryPerBlockKb(compute_capacity) * 1024;
  bool kernel_launched = false;

  auto max_seqlen_q_num = max_seqlen_q.to<uint64_t>();
  auto max_seqlen_k_num = max_seqlen_k.to<uint64_t>();

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    using KernelType = decltype(k_);
    bool is_launched = kernel_launched;
    if (is_launched) {
      return;
    }

    using scalar_t = typename KernelType::scalar_t;
    bool use_dropout = (dropout_p != 0);
    if (!KernelType::kSupportsDropout && use_dropout) {
      VLOG(3) << "run in to use dropout" << use_dropout;
      return;
    }
    if (!KernelType::kSupportsBias && bias) {
      VLOG(3) << "run in to bias";
      return;
    }

    const auto& v_dims = value.dims();
    if (KernelType::kSingleValueIteration &&
        KernelType::kKeysPerBlock < v_dims[3]) {
      VLOG(3) << "run in to value dim" << v_dims;
      return;
    }

    const auto& k_dims = key.dims();
    const auto& q_dims = query.dims();

    int64_t max_seqlen_q_tmp, max_seqlen_k_tmp;

    if (seqstart_q) {
      max_seqlen_q_tmp = max_seqlen_q_num;
      max_seqlen_k_tmp = 0;  // Will be set inside the kernel
    } else {
      max_seqlen_q_tmp = q_dims[1];
      max_seqlen_k_tmp = k_dims[1];
    }
    VLOG(3) << "max_seqlen_q_tmp " << max_seqlen_q_tmp;

    if ((q_dims[3] % KernelType::kAlignmentQ) ||
        (k_dims[3] % KernelType::kAlignmentK) ||
        (v_dims[3] % KernelType::kAlignmentV)) {
      VLOG(3) << "run in to query dim" << q_dims;
      VLOG(3) << "run in to key dim" << k_dims;
      return;
    }

    size_t smem_bytes = sizeof(typename KernelType::SharedStorage);
    if (smem_bytes > max_shmem) {
      VLOG(3) << "run in to shmem" << smem_bytes << " " << max_shmem;
      return;
    }

    kernel_launched = true;
    VLOG(3) << "launching";

    output->Resize({q_dims[0], q_dims[1], q_dims[2], v_dims[3]});

    constexpr int64_t kAlignLSE = KernelType::kAlignLSE;
    phi::Dim<3> logsumexp_dims;
    logsumexp_dims[0] = seqstart_q ? seqstart_q.get().dims()[0] - 1 : q_dims[0];
    logsumexp_dims[1] = q_dims[2];
    logsumexp_dims[2] =
        is_test ? 0 : (max_seqlen_q_tmp + kAlignLSE - 1) / kAlignLSE;
    logsumexp_dims[2] *= kAlignLSE;
    logsumexp->Resize(logsumexp_dims);
    ctx.template Alloc<float>(logsumexp);
    VLOG(3) << "logsumexp dims" << logsumexp_dims;
    VLOG(3) << "logsumexp" << logsumexp;
    VLOG(3) << "kAlignLSE" << kAlignLSE;

    typename KernelType::Params p;
    p.query_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(query.data()));
    p.key_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(key.data()));
    p.value_ptr =
        const_cast<scalar_t*>(reinterpret_cast<const scalar_t*>(value.data()));
    p.logsumexp_ptr = is_test ? nullptr : logsumexp->data<float>();
    VLOG(3) << "logsumexp_ptr" << p.logsumexp_ptr;

    DenseTensor out_accum;
    if (KernelType::kNeedsOutputAccumulatorBuffer) {
      out_accum.Resize(output->dims());
      p.output_accum_ptr =
          ctx.template Alloc<typename KernelType::output_accum_t>(&out_accum);
      VLOG(3) << "output_accum_ptr " << p.output_accum_ptr;
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = ctx.template Alloc<typename KernelType::output_t>(output);
    VLOG(3) << "output_ptr " << p.output_ptr;

    if (seqstart_q) {
      p.seqstart_q_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(seqstart_q.get().data()));
      p.seqstart_k_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(seqstart_k.get().data()));
      VLOG(3) << "seqstart_q_ptr " << p.seqstart_q_ptr;
    } else {
      p.seqstart_q_ptr = nullptr;
      p.seqstart_k_ptr = nullptr;
    }

    if (cu_seqlens_q) {
      int32_t* cu_seqstart_q_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(cu_seqlens_q.get().data()));
      int32_t* cu_seqstart_k_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(cu_seqlens_k.get().data()));
      VLOG(3) << "cu_seqstart_k_ptr " << cu_seqstart_k_ptr;
    } else {
      int32_t* cu_seqstart_q_ptr = nullptr;
      int32_t* cu_seqstart_k_ptr = nullptr;
    }

    p.num_heads = q_dims[2];
    p.head_dim = q_dims[3];
    p.head_dim_value = v_dims[3];

    p.num_queries = max_seqlen_q_tmp;
    p.num_keys = max_seqlen_k_tmp;
    p.num_batches = seqstart_q ? seqstart_q.get().dims()[0] - 1 : q_dims[0];
    p.causal = causal;
    if (causal_diagonal) {
      p.causal_diagonal_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(causal_diagonal.get().data()));
    } else {
      p.causal_diagonal_ptr = nullptr;
    }
    VLOG(3) << "causal_diagonal_ptr " << p.causal_diagonal_ptr;

    p.seqlen_k_ptr = nullptr;
    if (seqlen_k) {
      p.seqlen_k_ptr = const_cast<int32_t*>(
          reinterpret_cast<const int32_t*>(seqlen_k.get().data()));
    } else {
      p.seqlen_k_ptr = nullptr;
    }
    VLOG(3) << "seqlen_k_ptr " << p.seqlen_k_ptr;

    if (scale < 0) {
      p.scale = static_cast<float>(1.0 / std::sqrt(p.head_dim));
    } else {
      p.scale = scale;
    }
    VLOG(3) << "scale " << p.scale;

    p.q_strideB = q_dims[1] * q_dims[2] * q_dims[3];
    p.k_strideB = k_dims[1] * k_dims[2] * k_dims[3];
    p.v_strideB = v_dims[1] * v_dims[2] * v_dims[3];
    p.q_strideM = q_dims[2] * q_dims[3];
    p.k_strideM = k_dims[2] * k_dims[3];
    p.v_strideM = v_dims[2] * v_dims[3];
    p.q_strideH = q_dims[3];
    p.k_strideH = k_dims[3];
    p.v_strideH = v_dims[3];
    p.o_strideM = output->dims()[2] * output->dims()[3];

    if (bias) {
      p.attn_bias_ptr = const_cast<scalar_t*>(
          reinterpret_cast<const scalar_t*>(bias.get().data()));
      p.bias_strideB = q_dims[2] * q_dims[1] * k_dims[1];
      p.bias_strideH = q_dims[1] * k_dims[1];
      p.bias_strideM = k_dims[1];
    } else {
      p.attn_bias_ptr = nullptr;
    }
    VLOG(3) << "attn_bias_ptr " << p.attn_bias_ptr;
    VLOG(3) << "bias_strideB " << p.bias_strideB;
    VLOG(3) << "bias_strideH " << p.bias_strideH;
    VLOG(3) << "bias_strideM " << p.bias_strideM;

    uint64_t* seed_and_offset_ptr = const_cast<uint64_t*>(
        reinterpret_cast<const uint64_t*>(seed_and_offset->data()));
    auto gen = ctx.GetGenerator();
    uint64_t inc = query.dims()[0] * query.dims()[2] * 32;
    auto seed_offset_pair = gen->IncrementOffset(inc);
    auto seed = (uint64_t)(seed_offset_pair.first);
    auto offset = (uint64_t)(seed_offset_pair.second);
    seed_and_offset_ptr[0] = seed;
    seed_and_offset_ptr[1] = offset;
    VLOG(3) << "seed and offset: " << seed << " " << offset << " "
            << seed_and_offset_ptr;

    p.use_dropout = use_dropout;
    if (use_dropout) {
      p.seed = seed;
      p.offset = offset;
      p.dropout_prob = dropout_p;
    } else {
      p.dropout_prob = 0.0;
    }

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
  dispatch_cutlass_forward<T>(ctx, launchKernel);
  PADDLE_ENFORCE_EQ(kernel_launched, true);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    fused_multihead_attention,
    GPU,
    ALL_LAYOUT,
    phi::fusion::cutlass_internal::MultiHeadAttentionForwardKernel,
    float) {}
