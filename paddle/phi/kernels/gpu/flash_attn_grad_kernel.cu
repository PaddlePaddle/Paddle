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

#include "paddle/phi/kernels/flash_attn_grad_kernel.h"
#include <cstddef>
#include "glog/logging.h"  // For VLOG()
#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(cudnn_deterministic);
COMMON_DECLARE_int32(flash_attn_version);

namespace phi {

int get_num_split() {
  // 0 for an internal heuristic, which is optimal
  return FLAGS_cudnn_deterministic ? 1 : 0;
}

template <typename T, uint64_t HeaddimDiv32>
static __global__ void SumStridedKV(const T* src,
                                    T* dst,
                                    const uint64_t sRowDim1,
                                    const uint64_t sRowDim2,
                                    const uint64_t sRowDim3,
                                    const uint64_t sColDim,
                                    const uint64_t sRowStride1,
                                    const uint64_t sRowStride2,
                                    const uint64_t sColStride,
                                    const uint64_t dRowStride1,
                                    const uint64_t dRowStride2) {
  // SrcShape [seqlen, num_heads_k, num_heads/num_heads_k, headdim]
  // AxisName [row1  , row2       , col                  , row3   ]
  // LoopMap  [blockx, thready    , serialreduce         , threadx]
  // Ensure blockDim.x == 32 && blockDim.z == 1
  // Ensure sRowStride3 == dRowStride3 == 1 (headdim dim is contiguous)
  using IndexType = uint64_t;
  constexpr IndexType BlockDimX = 32;
  const IndexType SRow1Begin = blockIdx.x * sRowStride1;
  const IndexType SRow1End = sRowDim1 * sRowStride1;
  const IndexType SRow1Stride = gridDim.x * sRowStride1;

  const IndexType SRow2Begin = threadIdx.y * sRowStride2;
  const IndexType SRow2End = sRowDim2 * sRowStride2;
  const IndexType SRow2Stride = blockDim.y * sRowStride2;

  // const IndexType SRow3Begin = threadIdx.x * sRowStride3;
  // const IndexType SRow3End = sRowDim3 * sRowStride3;
  // const IndexType SRow3Stride = BlockDimX * sRowStride3;

  constexpr IndexType SColBegin = 0;
  const IndexType SColEnd = sColDim * sColStride;
  const IndexType SColStride = sColStride;

  const IndexType DRow1Begin = blockIdx.x * dRowStride1;
  const IndexType DRow1Stride = gridDim.x * dRowStride1;

  const IndexType DRow2Begin = threadIdx.y * dRowStride2;
  const IndexType DRow2Stride = dRowStride2;

  // const IndexType DRow3Begin = threadIdx.x * dRowStride3;
  // const IndexType DRow3Stride = blockDim.x * dRowStride3;

  for (auto row1 = SRow1Begin, drow1 = DRow1Begin; row1 < SRow1End;
       row1 += SRow1Stride, drow1 += DRow1Stride) {
    for (auto row2 = SRow2Begin, drow2 = DRow2Begin; row2 < SRow2End;
         row2 += SRow2Stride, drow2 += DRow2Stride) {
      const auto i1 = row1 + row2 + threadIdx.x;
      const auto di1 = drow1 + drow2 + threadIdx.x;
      T v[HeaddimDiv32];
#pragma unroll
      for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
        v[i] = T{0};
      }
      for (auto col = SColBegin; col < SColEnd; col += SColStride) {
        const auto i2 = i1 + col;
#pragma unroll
        for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
          v[i] += src[i2 + i * BlockDimX];
        }
      }
#pragma unroll
      for (auto i = IndexType(0); i < HeaddimDiv32; i++) {
        dst[di1 + i * BlockDimX] = v[i];
      }
    }
  }
}

template <typename T>
static auto selectSumkernel(int64_t headdim) {
  PADDLE_ENFORCE_LE(headdim,
                    256,
                    common::errors::InvalidArgument(
                        "FlashAttention only support headdim <= 256"));
  PADDLE_ENFORCE_EQ(headdim % 32,
                    0,
                    common::errors::InvalidArgument(
                        "FlashAttention only support headdim %% 32 == 0"));
  PADDLE_ENFORCE_NE(
      headdim, 0, common::errors::InvalidArgument("Headdim can't be zero"));
#define CASEN(n) \
  case n:        \
    return SumStridedKV<T, n>;
  switch (headdim / 32) {
    CASEN(1);
    CASEN(2);
    CASEN(3);
    CASEN(4);
    CASEN(5);
    CASEN(6);
    CASEN(7);
    CASEN(8);
  }
  PADDLE_FATAL("Unreachable in selectSumKernel");
#undef CASEN
}

template <typename T, typename Context>
static void kvReduceForGQA(const Context& ctx,
                           const DenseTensor& dk_tmp,
                           DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[2],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[3],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[2];
  const size_t blockNum =
      std::min((static_cast<int64_t>(dk_tmp.dims()[0] + 31) / 32),
               static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[3]);
  sumkernel<<<blockNum, threadNum, 0, ctx.stream()>>>(
      reinterpret_cast<const T*>(dk_tmp.data()),
      reinterpret_cast<T*>(dk->data()),
      dk_tmp.dims()[0],
      dk_tmp.dims()[1],
      dk_tmp.dims()[3],
      dk_tmp.dims()[2],
      dk_tmp.strides()[0],
      dk_tmp.strides()[1],
      // dk_tmp.strides()[3],
      dk_tmp.strides()[2],
      dk->strides()[0],
      dk->strides()[1]
      // dk->strides()[2]
  );
}
template <typename T, typename Context>
static void kvReduceBatchedForGQA(const Context& ctx,
                                  const DenseTensor& dk_tmp,
                                  DenseTensor* dk) {
  PADDLE_ENFORCE_EQ(
      dk->strides()[3],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(
      dk_tmp.strides()[4],
      1,
      common::errors::InvalidArgument("headdim dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(dk->strides()[0],
                    dk->strides()[1] * dk->dims()[1],
                    common::errors::InvalidArgument(
                        "batchsize dimension must be contiguous"));
  PADDLE_ENFORCE_EQ(dk_tmp.strides()[0],
                    dk_tmp.strides()[1] * dk_tmp.dims()[1],
                    common::errors::InvalidArgument(
                        "batchsize dimension must be contiguous"));
  const int64_t reduceDimSize = dk_tmp.dims()[3];
  const size_t blockNum = std::min(
      (static_cast<int64_t>(dk_tmp.dims()[0] * dk_tmp.dims()[1] + 31) / 32),
      static_cast<int64_t>(1024l));
  const dim3 threadNum{32, 4, 1};
  auto sumkernel = selectSumkernel<T>(dk_tmp.dims()[4]);
  // here implicitly flat [batch,seqlen], and require batch dim to be contiguous
  sumkernel<<<blockNum, threadNum, 0, ctx.stream()>>>(
      reinterpret_cast<const T*>(dk_tmp.data()),
      reinterpret_cast<T*>(dk->data()),
      dk_tmp.dims()[0] * dk_tmp.dims()[1],
      dk_tmp.dims()[2],
      dk_tmp.dims()[4],
      dk_tmp.dims()[3],
      dk_tmp.strides()[1],
      dk_tmp.strides()[2],
      // dk_tmp.strides()[4],
      dk_tmp.strides()[3],
      dk->strides()[1],
      dk->strides()[2]
      // dk->strides()[3]
  );
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv,
    bool varlen_padded) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [total_*, num_heads, head_dim]
  auto dims = q.dims();

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const int64_t num_heads = dims[1];
  const int64_t head_size_og = dout.dims()[2];
  const int64_t head_size = dims[2];
  const int64_t total_k = k.dims()[0];
  const int64_t num_heads_k = k.dims()[1];

  bool is_mha = (num_heads == num_heads_k);

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;
  if (!dq) {
    dq_tmp.Resize(dims);
    ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

#ifdef PADDLE_WITH_HIP
  std::initializer_list<int64_t> dk_dv_shape = {total_k, num_heads, head_size};
#else
  std::initializer_list<int64_t> dk_dv_shape = {
      total_k, num_heads_k, num_heads / num_heads_k, head_size};
#endif

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp;
  if (!dk || !is_mha) {
    dk_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!dv || !is_mha) {
    dv_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

#ifdef PADDLE_WITH_HIP
  const hipStream_t stream = ctx.stream();
#else
  const cudaStream_t stream = ctx.stream();
#endif

  int num_splits = get_num_split();

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           /*version=*/2,
                           batch_size,
                           max_seqlen_q,
                           max_seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           scale,
                           causal,
                           q.dtype(),
                           attn_mask,
                           nullptr,  // startend_row_indices,
                           seed_offset.data<int64_t>());

  VLOG(10) << "FlashAttn bwd seed: " << params.seed
           << ", offset: " << params.offset;
#ifdef PADDLE_WITH_HIP
  bool succ = phi::dynload::flash_attn_varlen_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      kdq->data(),
      kdk->data(),
      kdv->data(),
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      1.0f / params.softmax_scale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
#else
  bool succ = phi::dynload::flash_attn_varlen_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      cu_seqlens_q.data<int32_t>(),
      cu_seqlens_k.data<int32_t>(),
      params.rng_state.data(),
      kdq->data(),
      kdk->data(),
      kdv->data(),
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      1.0f / params.softmax_scale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
      q.strides()[0],
      k.strides()[0],
      v.strides()[0],
      q.strides()[1],
      k.strides()[1],
      v.strides()[1],
      out.strides()[0],
      out.strides()[1],
      max_seqlen_q * q.strides()[0],
      max_seqlen_k * k.strides()[0],
      max_seqlen_k * v.strides()[0],
      max_seqlen_q * out.strides()[0],
      kdq->strides()[0],
      kdk->strides()[0],
      kdv->strides()[0],
      kdq->strides()[1],
      kdk->strides()[kdk->strides().size() - 2],
      kdv->strides()[kdv->strides().size() - 2],
      dout.strides()[0],
      dout.strides()[1],
      max_seqlen_q * kdq->strides()[0],
      max_seqlen_k * kdk->strides()[0],
      max_seqlen_k * kdv->strides()[0],
      max_seqlen_q * dout.strides()[0],
      varlen_padded);
#endif
  CheckFlashAttnStatus(succ);
  if (!is_mha) {
    if (dk) {
#ifdef PADDLE_WITH_HIP
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(
            ctx,
            dk_tmp.Resize(
                {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2},
            dk->type(),
            false,
            dk);
      else
        kvReduceForGQA<T, Context>(
            ctx,
            dk_tmp.Resize(
                {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            dk);
#else
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dk_tmp, {2}, dk->type(), false, dk);
      else
        kvReduceForGQA<T, Context>(ctx, dk_tmp, dk);
#endif
    }
    if (dv) {
#ifdef PADDLE_WITH_HIP
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(
            ctx,
            dv_tmp.Resize(
                {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            {2},
            dv->type(),
            false,
            dv);
      else
        kvReduceForGQA<T, Context>(
            ctx,
            dv_tmp.Resize(
                {total_k, num_heads_k, num_heads / num_heads_k, head_size}),
            dv);
#else
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dv_tmp, {2}, dv->type(), false, dv);
      else
        kvReduceForGQA<T, Context>(ctx, dv_tmp, dv);
#endif
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnUnpaddedGradKernel(const Context& ctx,
                                 const DenseTensor& q,
                                 const DenseTensor& k,
                                 const DenseTensor& v,
                                 const DenseTensor& cu_seqlens_q,
                                 const DenseTensor& cu_seqlens_k,
                                 const DenseTensor& out,
                                 const DenseTensor& softmax_lse,
                                 const DenseTensor& seed_offset,
                                 const paddle::optional<DenseTensor>& attn_mask,
                                 const DenseTensor& dout,
                                 int64_t max_seqlen_q,
                                 int64_t max_seqlen_k,
                                 float scale,
                                 float dropout,
                                 bool causal,
                                 DenseTensor* dq,
                                 DenseTensor* dk,
                                 DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  if (dq) {
    ctx.template Alloc<T>(dq);
  }
  if (dk) {
    ctx.template Alloc<T>(dk);
  }
  if (dv) {
    ctx.template Alloc<T>(dv);
  }
  FlashAttnUnpaddedGradBaseKernel<T>(ctx,
                                     q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     out,
                                     softmax_lse,
                                     seed_offset,
                                     attn_mask,
                                     dout,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     scale,
                                     dropout,
                                     causal,
                                     dq,
                                     dk,
                                     dv,
                                     false /*varlen_padded*/);
#else
  RaiseNotSupportedError();
#endif
}

static void sliceFlattenView(const DenseTensor& in,
                             DenseTensor* out,
                             int axis,
                             int64_t offset,
                             int64_t sliceLength) {
  PADDLE_ENFORCE_LT(
      axis,
      in.dims().size(),
      common::errors::InvalidArgument("sliceView receive axis out of bound"));
  std::array<int64_t, DDim::kMaxRank> dimArr;
  std::array<int64_t, DDim::kMaxRank> strideArr;
  auto id = dimArr.begin(), is = strideArr.begin();
  for (int i = 0; i < in.dims().size(); i++) {
    if (i == axis) continue;
    if (i == axis + 1)
      *id = in.dims()[i] * sliceLength;
    else
      *id = in.dims()[i];
    *is = in.strides()[i];
    id++;
    is++;
  }
  *out = DenseTensor{
      in.Holder(),
      DenseTensorMeta{in.dtype(),
                      DDim{dimArr.data(), in.dims().size() - 1},
                      DDim(strideArr.data(), in.dims().size() - 1)}};
  out->set_offset(in.offset() +
                  offset * in.strides()[axis] * SizeOf(out->dtype()));
}
template <typename OutT>
struct ZeroFunctor {
  __device__ __forceinline__ OutT operator()() const {
    return static_cast<OutT>(0);
  }
};
template <typename T, typename Context>
void FlashAttnVarlenQKVPackedGradKernel(
    const Context& ctx,
    const DenseTensor& qkv,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    float scale,
    float dropout,
    bool causal,
    bool varlen_padded,
    DenseTensor* dqkv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q,k,v [total_*, num_heads, head_dim]
  const auto head_groupnum = qkv.dims()[1];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 1, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 1, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 1, head_groupnum - 1, 1);
  // DenseTensor dqkv_tmp;
  if (!dqkv) {
    return;
    // dqkv is the only output. No need to compute if no dqkv
    // dqkv_tmp.Resize(qkv.dims());
    // dqkv = &dqkv_tmp;
  }
  ctx.template Alloc<T>(dqkv);
  {
    std::vector<const DenseTensor*> inputs{};
    std::vector<DenseTensor*> outputs{dqkv};
    phi::funcs::ElementwiseKernel<T>(ctx, inputs, &outputs, ZeroFunctor<T>());
  }
  DenseTensor dq, dk, dv;
  sliceFlattenView(*dqkv, &dq, 1, 0, head_groupnum - 2);
  sliceFlattenView(*dqkv, &dk, 1, head_groupnum - 2, 1);
  sliceFlattenView(*dqkv, &dv, 1, head_groupnum - 1, 1);
  FlashAttnUnpaddedGradBaseKernel<T>(ctx,
                                     q,
                                     k,
                                     v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     out,
                                     softmax_lse,
                                     seed_offset,
                                     attn_mask,
                                     dout,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     scale,
                                     dropout,
                                     causal,
                                     &dq,
                                     &dk,
                                     &dv,
                                     varlen_padded);
#else
  RaiseNotSupportedError();
#endif
}
template <typename T, typename Context>
void FlashAttnGradBaseKernel(
    const Context& ctx,
    const DenseTensor& q,
    const DenseTensor& k,
    const DenseTensor& v,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const paddle::optional<DenseTensor>& startend_row_indices,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    DenseTensor* dq,
    DenseTensor* dk,
    DenseTensor* dv) {
#ifdef PADDLE_WITH_FLASHATTN
  // q, k, v [batch_size, seq_len, num_heads, head_dim]
  const auto& dims = q.dims();

  const int64_t batch_size = dims[0];
  const int64_t seqlen_q = dims[1];
  const int64_t num_heads = dims[2];
  const int64_t head_size_og = dout.dims()[3];
  const int64_t head_size = dims[3];
  const int64_t seqlen_k = k.dims()[1];
  const int64_t num_heads_k = k.dims()[2];

  bool is_mha = (num_heads == num_heads_k);

#ifdef PADDLE_WITH_HIP
  std::initializer_list<int64_t> dk_dv_shape = {
      batch_size, seqlen_k, num_heads, head_size};
#else
  std::initializer_list<int64_t> dk_dv_shape = {
      batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size};
#endif

  DenseTensor* kdq = dq;
  DenseTensor dq_tmp;
  if (!dq) {
    dq_tmp.Resize(dims);
    ctx.template Alloc<T>(&dq_tmp);
    kdq = &dq_tmp;
  }

  DenseTensor *kdk = dk, *kdv = dv;
  DenseTensor dk_tmp;
  if (!dk || !is_mha) {
    dk_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dk_tmp);
    kdk = &dk_tmp;
  }

  DenseTensor dv_tmp;
  if (!dv || !is_mha) {
    dv_tmp.Resize(dk_dv_shape);
    ctx.template Alloc<T>(&dv_tmp);
    kdv = &dv_tmp;
  }

#ifdef PADDLE_WITH_HIP
  const hipStream_t stream = ctx.stream();
#else
  const cudaStream_t stream = ctx.stream();
#endif

  // TODO(umiswing): add shape check
  PADDLE_ENFORCE_EQ(
      head_size_og,
      head_size,
      common::errors::InvalidArgument(
          "flash_attn_bwd receive input with head_size_og == head_size"));

  const float softmax_scale = 1.0f / std::sqrt(head_size);
  const float softmax_unscale = std::sqrt(head_size);

  int version =
      FLAGS_flash_attn_version == 3 &&
              (head_size == 64 || head_size == 128 || head_size == 256)
          ? FLAGS_flash_attn_version
          : 2;
  FlashAttnBwdParamsV2 params =
      FlashAttnBwdParamsV2(ctx,
                           version,
                           batch_size,
                           seqlen_q,
                           seqlen_k,
                           num_heads,
                           num_heads_k,
                           head_size,
                           dropout,
                           softmax_scale,
                           causal,
                           q.dtype(),
                           attn_mask,
                           startend_row_indices,
                           seed_offset.data<int64_t>());

  VLOG(10) << "[FlashAttn Backward" << version << "] q.shape=[" << q.dims()
           << "], k.shape=[" << k.dims() << "], v.shape=[" << v.dims() << "]";
  VLOG(10) << "[FlashAttn Backward" << version << "] dropout=" << dropout
           << ", seed=" << params.seed << ", offset=" << params.offset;
  VLOG(10) << "[FlashAttn Backward" << version
           << "] softmax_scale=" << softmax_scale
           << ", softmax_unscale=" << softmax_unscale;
  if (attn_mask.get_ptr()) {
    VLOG(10) << "[FlashAttn Backward" << version << "] attn_mask.shape=["
             << (attn_mask.get_ptr())->dims() << "]";
  }

  int num_splits = get_num_split();

  DenseTensor flashmask_maxmin, downstart_row_indices, upend_row_indices,
      downend_row_indices, upstart_row_indices;
  void *downstart_row_indices_data = nullptr, *upend_row_indices_data = nullptr,
       *downend_row_indices_data = nullptr, *upstart_row_indices_data = nullptr;
  bool is_flashmask = params.startend_row_indices != nullptr;
  if (is_flashmask) {
    PADDLE_ENFORCE_EQ(
        startend_row_indices->dims().size(),
        4,
        common::errors::InvalidArgument(
            "flashmask_attention receive startend_row_indices with dim "
            "[batch_size, num_heads,seq_len, mask_bounds]"));
    PADDLE_ENFORCE_EQ(startend_row_indices->dims()[3] == 1 ||
                          startend_row_indices->dims()[3] == 2 ||
                          startend_row_indices->dims()[3] == 4,
                      true,
                      common::errors::InvalidArgument(
                          "flashmask_attention startend_row_indices "
                          "mask_bounds must in [1,2,4]"));
    auto flashmask_maxmin_shape = params.startend_row_indices->dims();
    flashmask_maxmin_shape[2] = (flashmask_maxmin_shape[2] + 31) / 32 * 8;
    flashmask_maxmin.set_type(phi::DataType::INT32);
    flashmask_maxmin.Resize(flashmask_maxmin_shape);
    ctx.template Alloc<T>(&flashmask_maxmin);

    downstart_row_indices =
        phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {0}, {1});
    downstart_row_indices_data = downstart_row_indices.data();
    if (startend_row_indices->dims()[3] == 2) {
      if (!causal) {
        upend_row_indices =
            phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {1}, {2});
        upend_row_indices_data = upend_row_indices.data();
      } else {
        downend_row_indices =
            phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {1}, {2});
        downend_row_indices_data = downend_row_indices.data();
      }
    } else if (startend_row_indices->dims()[3] == 4) {
      upend_row_indices =
          phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {3}, {4});
      upend_row_indices_data = upend_row_indices.data();
      downend_row_indices =
          phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {1}, {2});
      downend_row_indices_data = downend_row_indices.data();
      upstart_row_indices =
          phi::Slice<int32_t>(ctx, startend_row_indices.get(), {3}, {2}, {3});
      upstart_row_indices_data = upstart_row_indices.data();
    }
  }

#ifdef PADDLE_WITH_HIP
  bool succ = phi::dynload::flash_attn_bwd(
      dout.data(),
      q.data(),
      k.data(),
      v.data(),
      out.data(),
      params.softmax_d.data(),
      softmax_lse.data(),
      params.rng_state.data(),
      kdq->data(),
      kdk->data(),
      kdv->data(),
      params.dq_accum.data(),
      params.batch_size,
      params.max_seqlen_q,
      params.max_seqlen_k,
      params.seqlen_q_rounded,
      params.seqlen_k_rounded,
      params.num_heads,
      params.num_heads_k,
      params.head_size,
      params.head_size_rounded,
      params.dropout,
      params.softmax_scale,
      softmax_unscale,
      params.causal,
      params.is_bf16,
      num_splits,
      stream,
      params.seed,
      params.offset,
      params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
      params.attn_mask_tensor ? params.mask_dims.data() : nullptr);
#else
  bool succ;
  int arch =
      backends::gpu::GetGPUComputeCapability(ctx.GetPlace().GetDeviceId());

  if (arch == 80 && version == 3) {
    RaiseNotSupportedError(3);
  }

  if (arch == 90 && version == 3) {
#ifdef PADDLE_WITH_FLASHATTN_V3
    if (is_flashmask || params.attn_mask_tensor) {
      PADDLE_THROW(common::errors::Unimplemented(
          "FlashMask or Dense Mask is unsupported in FlashAttention V3"));
    }

    bool deterministic = FLAGS_cudnn_deterministic ? true : false;
    succ = phi::dynload::flash_attn_v3_bwd(
        dout.data(),
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        params.softmax_d.data(),
        softmax_lse.data(),
        params.softmax_lse_log2.data(),
        params.rng_state.data(),
        kdq->data(),
        kdk->data(),
        kdv->data(),
        params.dq_accum.data(),
        params.batch_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.seqlen_q_rounded,
        params.seqlen_k_rounded,
        params.num_heads,
        params.num_heads_k,
        params.head_size,
        params.head_size_rounded,
        params.dropout,
        params.softmax_scale,
        softmax_unscale,
        params.causal,
        params.is_bf16,
        num_splits,
        deterministic,
        stream,
        params.seed,
        params.offset,
        params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
        params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
        is_flashmask ? downstart_row_indices_data : nullptr,
        is_flashmask ? downend_row_indices_data : nullptr,
        is_flashmask ? upend_row_indices_data : nullptr,
        is_flashmask ? upstart_row_indices_data : nullptr,
        is_flashmask ? flashmask_maxmin.data() : nullptr,
        is_flashmask ? params.startend_row_indices_dims.data() : nullptr,
        q.strides()[0],
        k.strides()[0],
        v.strides()[0],
        q.strides()[1],
        k.strides()[1],
        v.strides()[1],
        q.strides()[2],
        k.strides()[2],
        v.strides()[2],
        out.strides()[0],
        out.strides()[1],
        out.strides()[2],
        kdq->strides()[0],
        kdk->strides()[0],
        kdv->strides()[0],
        kdq->strides()[1],
        kdk->strides()[1],
        kdv->strides()[1],
        kdq->strides()[2],
        kdk->strides()[kdk->strides().size() - 2],
        kdv->strides()[kdv->strides().size() - 2],
        dout.strides()[0],
        dout.strides()[1],
        dout.strides()[2],
        params.dq_semaphore.data());
#else
    RaiseNotSupportedError(3);
#endif
  } else {
    succ = phi::dynload::flash_attn_bwd(
        dout.data(),
        q.data(),
        k.data(),
        v.data(),
        out.data(),
        params.softmax_d.data(),
        softmax_lse.data(),
        params.rng_state.data(),
        kdq->data(),
        kdk->data(),
        kdv->data(),
        params.dq_accum.data(),
        params.batch_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.seqlen_q_rounded,
        params.seqlen_k_rounded,
        params.num_heads,
        params.num_heads_k,
        params.head_size,
        params.head_size_rounded,
        params.dropout,
        params.softmax_scale,
        softmax_unscale,
        params.causal,
        params.is_bf16,
        num_splits,
        stream,
        params.seed,
        params.offset,
        params.attn_mask_tensor ? params.attn_mask_tensor->data() : nullptr,
        params.attn_mask_tensor ? params.mask_dims.data() : nullptr,
        is_flashmask ? downstart_row_indices_data : nullptr,
        is_flashmask ? params.startend_row_indices_dims.data() : nullptr,
        is_flashmask ? upend_row_indices_data : nullptr,
        is_flashmask ? downend_row_indices_data : nullptr,
        is_flashmask ? upstart_row_indices_data : nullptr,
        is_flashmask ? flashmask_maxmin.data() : nullptr,
        q.strides()[1],
        k.strides()[1],
        v.strides()[1],
        q.strides()[2],
        k.strides()[2],
        v.strides()[2],
        out.strides()[1],
        out.strides()[2],
        q.strides()[0],
        k.strides()[0],
        v.strides()[0],
        out.strides()[0],
        kdq->strides()[1],
        kdk->strides()[1],
        kdv->strides()[1],
        kdq->strides()[2],
        kdk->strides()[kdk->strides().size() - 2],
        kdv->strides()[kdv->strides().size() - 2],
        dout.strides()[1],
        dout.strides()[2],
        kdq->strides()[0],
        kdk->strides()[0],
        kdv->strides()[0],
        dout.strides()[0]);
  }
#endif
  CheckFlashAttnStatus(succ);
  if (!is_mha) {
    if (dk) {
#ifdef PADDLE_WITH_HIP
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx,
                                   dk_tmp.Resize({batch_size,
                                                  seqlen_k,
                                                  num_heads_k,
                                                  num_heads / num_heads_k,
                                                  head_size}),
                                   {3},
                                   dk->type(),
                                   false,
                                   dk);
      else
        kvReduceBatchedForGQA<T, Context>(
            ctx,
            dk_tmp.Resize({batch_size,
                           seqlen_k,
                           num_heads_k,
                           num_heads / num_heads_k,
                           head_size}),
            dk);
#else
      if (dk->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dk_tmp, {3}, dk->type(), false, dk);
      else
        kvReduceBatchedForGQA<T, Context>(ctx, dk_tmp, dk);
#endif
    }

    if (dv) {
#ifdef PADDLE_WITH_HIP
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx,
                                   dv_tmp.Resize({batch_size,
                                                  seqlen_k,
                                                  num_heads_k,
                                                  num_heads / num_heads_k,
                                                  head_size}),
                                   {3},
                                   dv->type(),
                                   false,
                                   dv);
      else
        kvReduceBatchedForGQA<T, Context>(
            ctx,
            dv_tmp.Resize({batch_size,
                           seqlen_k,
                           num_heads_k,
                           num_heads / num_heads_k,
                           head_size}),
            dv);
#else
      if (dv->meta().is_contiguous())
        phi::SumKernel<T, Context>(ctx, dv_tmp, {3}, dv->type(), false, dv);
      else
        kvReduceBatchedForGQA<T, Context>(ctx, dv_tmp, dv);
#endif
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashAttnGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const paddle::optional<DenseTensor>& attn_mask,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  if (dq) {
    ctx.template Alloc<T>(dq);
  }
  if (dk) {
    ctx.template Alloc<T>(dk);
  }
  if (dv) {
    ctx.template Alloc<T>(dv);
  }
  FlashAttnGradBaseKernel<T, Context>(ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      causal,
                                      dq,
                                      dk,
                                      dv);
}

template <typename T, typename Context>
void FlashAttnQKVPackedGradKernel(
    const Context& ctx,
    const DenseTensor& qkv,
    const DenseTensor& out,
    const DenseTensor& softmax_lse,
    const DenseTensor& seed_offset,
    const paddle::optional<DenseTensor>& attn_mask,
    const DenseTensor& dout,
    float dropout,
    bool causal,
    DenseTensor* dqkv) {
#ifdef PADDLE_WITH_FLASHATTN
  // qkv [batchsize, seqlen, nheads/nheads_k+2, nheads_k, head_dim]
  const auto head_groupnum = qkv.dims()[2];  // nheads/nheads_k + 1 + 1
  DenseTensor q, k, v;
  sliceFlattenView(qkv, &q, 2, 0, head_groupnum - 2);
  sliceFlattenView(qkv, &k, 2, head_groupnum - 2, 1);
  sliceFlattenView(qkv, &v, 2, head_groupnum - 1, 1);
  // DenseTensor dqkv_tmp;
  if (!dqkv) {
    return;
    // dqkv is the only output. No need to compute if no dqkv
    // dqkv_tmp.Resize(qkv.dims());
    // dqkv = &dqkv_tmp;
  }
  ctx.template Alloc<T>(dqkv);
  DenseTensor dq, dk, dv;
  sliceFlattenView(*dqkv, &dq, 2, 0, head_groupnum - 2);
  sliceFlattenView(*dqkv, &dk, 2, head_groupnum - 2, 1);
  sliceFlattenView(*dqkv, &dv, 2, head_groupnum - 1, 1);
  FlashAttnGradBaseKernel<T, Context>(ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      attn_mask,
                                      paddle::none,
                                      dout,
                                      dropout,
                                      causal,
                                      &dq,
                                      &dk,
                                      &dv);
#else
  RaiseNotSupportedError();
#endif
}

template <typename T, typename Context>
void FlashMaskGradKernel(const Context& ctx,
                         const DenseTensor& q,
                         const DenseTensor& k,
                         const DenseTensor& v,
                         const DenseTensor& startend_row_indices,
                         const DenseTensor& out,
                         const DenseTensor& softmax_lse,
                         const DenseTensor& seed_offset,
                         const DenseTensor& dout,
                         float dropout,
                         bool causal,
                         DenseTensor* dq,
                         DenseTensor* dk,
                         DenseTensor* dv) {
  if (dq) {
    ctx.template Alloc<T>(dq);
  }
  if (dk) {
    ctx.template Alloc<T>(dk);
  }
  if (dv) {
    ctx.template Alloc<T>(dv);
  }
  FlashAttnGradBaseKernel<T, Context>(ctx,
                                      q,
                                      k,
                                      v,
                                      out,
                                      softmax_lse,
                                      seed_offset,
                                      paddle::none,
                                      startend_row_indices,
                                      dout,
                                      dropout,
                                      causal,
                                      dq,
                                      dk,
                                      dv);
}
}  // namespace phi

PD_REGISTER_KERNEL(flash_attn_unpadded_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnUnpaddedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(7).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_varlen_qkvpacked_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnVarlenQKVPackedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(5).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_KERNEL(flash_attn_qkvpacked_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashAttnQKVPackedGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(3).SetBackend(phi::Backend::CPU);  // seed_offset
}

PD_REGISTER_KERNEL(flashmask_attention_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlashMaskGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(6).SetBackend(phi::Backend::CPU);  // seed_offset
}
