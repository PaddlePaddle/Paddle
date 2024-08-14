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

#include "paddle/phi/api/include/tensor_operants.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/get_pad_lse.cu.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention/autogen/memory_efficient_attention.h"
#include "paddle/phi/kernels/fusion/cutlass/memory_efficient_attention_utils.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

using gemm_kernel_utils::getMaximumSharedMemoryPerBlockKb;

template <typename T, typename Context>
void MemoryEfficientAttentionGradKernel(
    const Context& ctx,
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const paddle::optional<DenseTensor>& bias,
    const paddle::optional<DenseTensor>& cu_seqlens_q,
    const paddle::optional<DenseTensor>& cu_seqlens_k,
    const DenseTensor& output,
    const DenseTensor& logsumexp,
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

  DenseTensor dq_tmp;
  DenseTensor dk_tmp;
  DenseTensor dv_tmp;
  bool has_query_grad = (query_grad != nullptr);
  bool has_key_grad = (key_grad != nullptr);
  bool has_value_grad = (value_grad != nullptr);

  auto launchKernel = [&](auto k_, auto kernel_fn) {
    PADDLE_ENFORCE_EQ(
        query.dims().size(),
        output_grad.dims().size(),
        common::errors::InvalidArgument(
            "The size of query's dimensions "
            "should be equal to output grad. But received query's "
            "dimensions = %d, output grad's dimensions = %d.",
            query.dims().size(),
            output_grad.dims().size()));
    PADDLE_ENFORCE_EQ(query.dims().size(),
                      key.dims().size(),
                      common::errors::InvalidArgument(
                          "The size of query's dimensions "
                          "should be equal to key. But received query's "
                          "dimensions = %d, key's dimensions = %d.",
                          query.dims().size(),
                          key.dims().size()));
    PADDLE_ENFORCE_EQ(query.dims().size(),
                      value.dims().size(),
                      common::errors::InvalidArgument(
                          "The size of query's dimensions "
                          "should be equal to value. But received query's "
                          "dimensions = %d, value's dimensions = %d.",
                          query.dims().size(),
                          key.dims().size()));
    PADDLE_ENFORCE_EQ(query.dims().size(),
                      4,
                      common::errors::InvalidArgument(
                          "The size of query's dimensions "
                          "dim size of query is illegal. Expected dimension "
                          "size=4. Received %d.",
                          query.dims().size()));

    // batch size
    PADDLE_ENFORCE_EQ(
        query.dims()[0],
        output_grad.dims()[0],
        common::errors::InvalidArgument(
            "The batch size of query's dimensions "
            "should be equal to output grad. But received query's "
            "batch size = %d, output grad's batch size = %d.",
            query.dims()[0],
            output_grad.dims()[0]));
    PADDLE_ENFORCE_EQ(query.dims()[0],
                      key.dims()[0],
                      common::errors::InvalidArgument(
                          "The batch size of query's dimensions "
                          "should be equal to key. But received query's "
                          "batch size = %d, key's batch size = %d.",
                          query.dims()[0],
                          key.dims()[0]));
    PADDLE_ENFORCE_EQ(query.dims()[0],
                      value.dims()[0],
                      common::errors::InvalidArgument(
                          "The batch size of query's dimensions "
                          "should be equal to value. But received query's "
                          "batch size = %d, value's batch size = %d.",
                          query.dims()[0],
                          value.dims()[0]));

    // seqlen
    PADDLE_ENFORCE_EQ(
        key.dims()[1],
        value.dims()[1],
        common::errors::InvalidArgument(
            "The sequence length of key"
            "should be equal to value. But received key's sequence length = "
            "%d, value's sequence length = %d.",
            key.dims()[1],
            value.dims()[1]));
    PADDLE_ENFORCE_EQ(query.dims()[1],
                      output_grad.dims()[1],
                      common::errors::InvalidArgument(
                          "The sequence length of query"
                          "should be equal to output grad. But received "
                          "query's sequence length = "
                          "%d, output grad's sequence length = %d.",
                          query.dims()[1],
                          output_grad.dims()[1]));

    // Num heads
    PADDLE_ENFORCE_EQ(
        query.dims()[2],
        key.dims()[2],
        common::errors::InvalidArgument(
            "The head number of query"
            "should be equal to key. But received query's head number = "
            "%d, key's head number = %d.",
            query.dims()[2],
            key.dims()[2]));
    PADDLE_ENFORCE_EQ(
        query.dims()[2],
        value.dims()[2],
        common::errors::InvalidArgument(
            "The head number of query"
            "should be equal to value. But received query's head number = "
            "%d, value's head number = %d.",
            query.dims()[2],
            value.dims()[2]));
    PADDLE_ENFORCE_EQ(query.dims()[2],
                      output_grad.dims()[2],
                      common::errors::InvalidArgument(
                          "The head number of query"
                          "should be equal to output grad. But received "
                          "query's head number = "
                          "%d, output grad's head number = %d.",
                          query.dims()[2],
                          output_grad.dims()[2]));

    // Embedding per head
    PADDLE_ENFORCE_EQ(
        query.dims()[3],
        key.dims()[3],
        common::errors::InvalidArgument(
            "The head size of query"
            "should be equal to key. But received query's head size = "
            "%d, key's head size = %d.",
            query.dims()[3],
            key.dims()[3]));
    PADDLE_ENFORCE_EQ(
        value.dims()[3],
        output_grad.dims()[3],
        common::errors::InvalidArgument(
            "The head size of value"
            "should be equal to output grad. But received value's head size = "
            "%d, output grad's head size = %d.",
            value.dims()[3],
            output_grad.dims()[3]));

    if (cu_seqlens_q) {
      PADDLE_ENFORCE_EQ((cu_seqlens_q && bias),
                        false,
                        common::errors::InvalidArgument(
                            "cu_seqlens_q or bias should be None"));
      PADDLE_ENFORCE_EQ(
          (cu_seqlens_k && cu_seqlens_q),
          true,
          common::errors::InvalidArgument(
              "cu_seqlens_q and cu_seqlens_k should be same condition"));
    } else {
      PADDLE_ENFORCE_EQ(
          (cu_seqlens_k || cu_seqlens_q),
          false,
          common::errors::InvalidArgument(
              "cu_seqlens_q and cu_seqlens_k should be same condition"));
    }

    const auto& k_dims = key.dims();
    const auto& q_dims = query.dims();
    const auto& v_dims = value.dims();

    int64_t max_seqlen_q_tmp, max_seqlen_k_tmp;
    if (cu_seqlens_q) {
      PADDLE_ENFORCE_EQ(cu_seqlens_q.get().dtype(),
                        DataType::INT32,
                        common::errors::InvalidArgument(
                            "data type of cu_seqlens_q should be INT32"));
      PADDLE_ENFORCE_EQ(cu_seqlens_k.get().dtype(),
                        DataType::INT32,
                        common::errors::InvalidArgument(
                            "data type of cu_seqlens_k should be INT32"));
      PADDLE_ENFORCE_EQ(cu_seqlens_q.get().dims().size(),
                        1,
                        common::errors::InvalidArgument(
                            "dims of cu_seqlens_q should be one"));
      PADDLE_ENFORCE_EQ(cu_seqlens_k.get().dims().size(),
                        1,
                        common::errors::InvalidArgument(
                            "dims of cu_seqlens_k should be one"));
      max_seqlen_q_tmp = max_seqlen_q.to<int64_t>();
      max_seqlen_k_tmp = max_seqlen_k.to<int64_t>();
      VLOG(3) << "max_seqlen_q_tmp" << max_seqlen_q_tmp;
      VLOG(3) << "max_seqlen_k_tmp" << max_seqlen_k_tmp;
      PADDLE_ENFORCE_EQ(
          cu_seqlens_q.get().dims()[0],
          cu_seqlens_k.get().dims()[0],
          common::errors::InvalidArgument("The first dimension of cu_seqlens_q"
                                          "should be equal to cu_seqlens_q."));
      PADDLE_ENFORCE_EQ(
          q_dims[0],
          1,
          common::errors::InvalidArgument(
              "The batch number of query"
              "should be one. But received batch number of query = %d.",
              q_dims[0]));
      PADDLE_ENFORCE_LT(0,
                        max_seqlen_q_tmp,
                        common::errors::InvalidArgument(
                            "The max sequence length of query"
                            "should more than zero. But received the max "
                            "sequence length of query = %d.",
                            max_seqlen_q_tmp));
      PADDLE_ENFORCE_LT(0,
                        max_seqlen_k_tmp,
                        common::errors::InvalidArgument(
                            "The max sequence length of key"
                            "should more than zero. But received the max "
                            "sequence length of key = %d.",
                            max_seqlen_k_tmp));
      PADDLE_ENFORCE_LE(max_seqlen_q_tmp,
                        q_dims[1],
                        common::errors::InvalidArgument(
                            "The max sequence length of query"
                            "should larger than sequence length of query. But "
                            "received the max sequence length of query = %d,"
                            "the sequence length of query = %d",
                            max_seqlen_q_tmp,
                            q_dims[1]));
      PADDLE_ENFORCE_LE(max_seqlen_k_tmp,
                        k_dims[1],
                        common::errors::InvalidArgument(
                            "The max sequence length of key"
                            "should larger than sequence length of key. But "
                            "received the max sequence length of key = %d,"
                            "the sequence length of key = %d",
                            max_seqlen_k_tmp,
                            k_dims[1]));
    } else {
      max_seqlen_q_tmp = q_dims[1];
      max_seqlen_k_tmp = k_dims[1];
    }
    VLOG(3) << "max_seqlen_q_tmp has been set " << max_seqlen_q_tmp
            << " max_seqlen_k_tmp " << max_seqlen_k_tmp;

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

    VLOG(3) << "smem has been set " << smem_bytes << " " << max_shmem;

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
    VLOG(3) << "p.output" << output.dtype();
    VLOG(3) << "p.output_grad" << output_grad.dtype();

    PADDLE_ENFORCE_EQ(
        delta.dims()[0],
        query.dims()[0],
        common::errors::InvalidArgument(
            "The first dimension of delta"
            "should be equal to query. But received delta's first dimension = "
            "%d, query's first dimension = %d.",
            delta.dims()[0],
            query.dims()[0]));
    PADDLE_ENFORCE_EQ(delta.dims()[1],
                      query.dims()[2],
                      common::errors::InvalidArgument(
                          "The second dimension of delta"
                          "should be equal to third dimension query. But "
                          "received delta's second dimension = "
                          "%d, query's third dimension = %d.",
                          delta.dims()[1],
                          query.dims()[2]));
    PADDLE_ENFORCE_EQ(delta.dims()[2],
                      query.dims()[1],
                      common::errors::InvalidArgument(
                          "The third dimension of delta"
                          "should be equal to second dimension query. But "
                          "received delta's third dimension = "
                          "%d, query's second dimension = %d.",
                          delta.dims()[2],
                          query.dims()[1]));

    VLOG(3) << "delta has been set" << delta.data();

    typename KernelType::Params p;
    p.query_ptr = phi::SafeGetTensorPtr<scalar_t>(query);
    p.key_ptr = phi::SafeGetTensorPtr<scalar_t>(key);
    p.value_ptr = phi::SafeGetTensorPtr<scalar_t>(value);

    bool force_pad_inf = (compute_capacity == 75);
    const std::string data_format = "NCHW";
    DenseTensor padded_lse =
        phi::funcs::get_pad_lse<float>(ctx,
                                       const_cast<DenseTensor*>(&logsumexp),
                                       static_cast<int>(output.dims()[1]),
                                       32,
                                       data_format,
                                       force_pad_inf);
    p.logsumexp_ptr = phi::SafeGetTensorPtr<float>(padded_lse);
    VLOG(3) << "logsumexp_ptr" << p.logsumexp_ptr;
    p.output_ptr = phi::SafeGetTensorPtr<scalar_t>(output);
    p.grad_output_ptr = phi::SafeGetTensorPtr<scalar_t>(output_grad);

    if (!has_query_grad) {
      dq_tmp.clear();
      dq_tmp = EmptyLike<T, Context>(ctx, query);
      query_grad = &dq_tmp;
    }
    p.grad_query_ptr = phi::SafeAllocTensor<scalar_t, Context>(ctx, query_grad);

    if (!has_key_grad) {
      dk_tmp.clear();
      dk_tmp = EmptyLike<T, Context>(ctx, key);
      key_grad = &dk_tmp;
    }
    p.grad_key_ptr = phi::SafeAllocTensor<scalar_t, Context>(ctx, key_grad);

    if (!has_value_grad) {
      dv_tmp.clear();
      dv_tmp = EmptyLike<T, Context>(ctx, value);
      value_grad = &dv_tmp;
    }
    p.grad_value_ptr = phi::SafeAllocTensor<scalar_t, Context>(ctx, value_grad);

    p.delta_ptr = phi::SafeGetTensorPtr<float>(delta);
    PD_MEA_CHECK_OVERFLOW(p.head_dim, q_dims[3]);
    PD_MEA_CHECK_OVERFLOW(p.head_dim_value, v_dims[3]);

    PD_MEA_CHECK_OVERFLOW(p.num_queries, max_seqlen_q_tmp);
    PD_MEA_CHECK_OVERFLOW(p.num_keys, max_seqlen_k_tmp);
    PD_MEA_CHECK_OVERFLOW(
        p.num_batches,
        cu_seqlens_q ? cu_seqlens_q.get().dims()[0] - 1 : q_dims[0]);
    PD_MEA_CHECK_OVERFLOW(p.num_heads, q_dims[2]);
    p.causal = causal;

    if (scale < 0) {
      p.scale = static_cast<float>(1.0 / std::sqrt(p.head_dim));
    } else {
      p.scale = scale;
    }
    VLOG(3) << "p.scale" << p.scale;

    if (cu_seqlens_q) {
      p.cu_seqlens_q_ptr = phi::SafeGetTensorPtr<int32_t>(cu_seqlens_q);
      p.cu_seqlens_k_ptr = phi::SafeGetTensorPtr<int32_t>(cu_seqlens_k);
      VLOG(3) << "p.cu_seqlens_q_ptr" << p.cu_seqlens_q_ptr;
    }

    PD_MEA_CHECK_OVERFLOW(p.lse_strideH, DimStride(logsumexp.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.lse_strideB, DimStride(logsumexp.dims(), 0));
    VLOG(3) << "p.lse_strideH " << p.lse_strideH;

    PD_MEA_CHECK_OVERFLOW(p.gO_strideH, DimStride(output_grad.dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.gO_strideM, DimStride(output_grad.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.gO_strideB, DimStride(output_grad.dims(), 0));

    PD_MEA_CHECK_OVERFLOW(p.o_strideH, DimStride(output.dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.o_strideB, DimStride(output.dims(), 0));

    PD_MEA_CHECK_OVERFLOW(p.gQ_strideH, DimStride(query_grad->dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.gQ_strideB, DimStride(query_grad->dims(), 0));

    PD_MEA_CHECK_OVERFLOW(p.gK_strideH, DimStride(key_grad->dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.gK_strideB, DimStride(key_grad->dims(), 0));

    PD_MEA_CHECK_OVERFLOW(p.gV_strideH, DimStride(value_grad->dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.gV_strideB, DimStride(value_grad->dims(), 0));

    p.gQKV_strideM_multiplier = 1;
    PADDLE_ENFORCE_EQ(q_dims[2] * q_dims[3],
                      DimStride(query_grad->dims(), 1),
                      common::errors::InvalidArgument(
                          "The strideM of grad query"
                          "should be equal to the first dimension size of "
                          "query grad's stride"));
    PADDLE_ENFORCE_EQ(k_dims[2] * k_dims[3],
                      DimStride(key_grad->dims(), 1),
                      common::errors::InvalidArgument(
                          "The strideM of grad key"
                          "should be equal to the first dimension size of key "
                          "grad's stride"));
    PADDLE_ENFORCE_EQ(v_dims[2] * v_dims[3],
                      DimStride(value_grad->dims(), 1),
                      common::errors::InvalidArgument(
                          "The strideM of grad value"
                          "should be equal to the first dimension size of "
                          "value grad's stride"));

    PD_MEA_CHECK_OVERFLOW(p.q_strideB, DimStride(query.dims(), 0));
    PD_MEA_CHECK_OVERFLOW(p.k_strideB, DimStride(key.dims(), 0));
    PD_MEA_CHECK_OVERFLOW(p.v_strideB, DimStride(value.dims(), 0));
    PD_MEA_CHECK_OVERFLOW(p.q_strideM, DimStride(query.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.k_strideM, DimStride(key.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.v_strideM, DimStride(value.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.q_strideH, DimStride(query.dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.k_strideH, DimStride(key.dims(), 2));
    PD_MEA_CHECK_OVERFLOW(p.v_strideH, DimStride(value.dims(), 2));

    PD_MEA_CHECK_OVERFLOW(p.delta_strideH, DimStride(delta.dims(), 1));
    PD_MEA_CHECK_OVERFLOW(p.delta_strideB, DimStride(delta.dims(), 0));

    if (bias) {
      p.bias_ptr = phi::SafeGetTensorPtr<scalar_t>(bias);
      const auto& bias_dims = bias.get().dims();
      PD_MEA_CHECK_OVERFLOW(
          p.bias_strideB,
          GetMemoryEfficientBiasStrideB(bias_dims, q_dims, k_dims));
      PD_MEA_CHECK_OVERFLOW(
          p.bias_strideH,
          GetMemoryEfficientBiasStrideH(bias_dims, q_dims, k_dims));
      PD_MEA_CHECK_OVERFLOW(p.bias_strideM, k_dims[1]);
      VLOG(3) << "p.bias_ptr" << p.bias_ptr;
      if (bias_grad) {
        p.grad_bias_ptr =
            phi::SafeAllocTensor<scalar_t, Context>(ctx, bias_grad);
        PD_MEA_CHECK_OVERFLOW(p.gB_strideB, q_dims[2] * q_dims[1] * k_dims[1]);
        PD_MEA_CHECK_OVERFLOW(p.gB_strideH, q_dims[1] * k_dims[1]);
        PD_MEA_CHECK_OVERFLOW(p.gB_strideM, k_dims[1]);
        VLOG(3) << "p.grad_bias_ptr" << p.grad_bias_ptr;
      } else {
        p.grad_bias_ptr = nullptr;
      }
    } else {
      p.bias_ptr = nullptr;
      p.grad_bias_ptr = nullptr;
    }
    if (dropout_p != 0) {
      int64_t* seed_and_offset_ptr =
          phi::SafeGetTensorPtr<int64_t>(seed_and_offset);
      p.seed = (uint64_t)seed_and_offset_ptr[0];
      p.offset = (uint64_t)seed_and_offset_ptr[1];
      p.dropout_prob = dropout_p;
      VLOG(3) << "seed_and_offset_ptr " << seed_and_offset_ptr;
      VLOG(3) << "p.seed " << p.seed << " " << p.offset;
      VLOG(3) << "p.dropout_prob " << p.dropout_prob;
    }

    int64_t size_bytes = p.workspace_size();
    phi::Allocator::AllocationPtr temp_workspace{nullptr};
    VLOG(3) << "size_bytes " << size_bytes;
    temp_workspace = phi::memory_utils::Alloc(
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
      const void* kernel_fn_void_ptr =
          reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(kernel_fn));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaFuncSetAttribute(kernel_fn_void_ptr,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem_bytes));
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
  PADDLE_ENFORCE_EQ(
      kernel_launched,
      true,
      common::errors::InvalidArgument("the kernel should not be launched"));
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    memory_efficient_attention_grad,
    GPU,
    ALL_LAYOUT,
    phi::fusion::cutlass_internal::MemoryEfficientAttentionGradKernel,
    float,
    phi::dtype::bfloat16,
    phi::dtype::float16) {
  kernel->InputAt(8).SetBackend(phi::Backend::ALL_BACKEND);
}
