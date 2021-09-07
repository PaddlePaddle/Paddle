/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "paddle/fluid/operators/contrib/fmha_op.h"
#include "paddle/fluid/platform/float16.h"

#include "paddle/fluid/operators/contrib/fmha.h"
#include "paddle/fluid/operators/contrib/fmha_dgrad_kernel_1xN_reload.h"
#include "paddle/fluid/operators/contrib/fmha_fprop_kernel_1xN.h"

namespace paddle {
namespace operators {

template <typename DTYPE>
static inline void set_alpha(uint32_t *alpha, float norm) {
  if (std::is_same<DTYPE, paddle::platform::float16>::value) {
    half x = __float2half_rn(norm);
    uint16_t h = reinterpret_cast<const uint16_t &>(x);
    ushort2 h2 = {h, h};
    *alpha = reinterpret_cast<const uint32_t &>(h2);
  } else if (std::is_same<DTYPE, float>::value) {
    *alpha = reinterpret_cast<const uint32_t &>(norm);
  } else if (std::is_same<DTYPE, int32_t>::value) {
    int32_t inorm = static_cast<int32_t>(norm);
    *alpha = reinterpret_cast<const uint32_t &>(inorm);
  } else {
    static_assert("Unsupported DTYPE");
    // sometime static_assert not works
    assert(false && "unsupported dtype");
  }
}

template <typename DTYPE, typename ACCTYPE>
void set_params(Fused_multihead_attention_fprop_params *params, const size_t b,
                const size_t s, const size_t h, const size_t d,
                void *qkv_packed_d, void *cu_seqlens_d, void *seqlens_d,
                void *o_packed_d, void *s_d, float p_dropout) {
  // Reset the parameters
  memset(params, 0, sizeof(params));

  // Set the pointers and strides.
  params->qkv_ptr = qkv_packed_d;
  params->qkv_stride_in_bytes = h * 3 * d * sizeof(DTYPE);
  params->o_ptr = o_packed_d;
  params->o_stride_in_bytes = h * d * sizeof(DTYPE);

  params->cu_seqlens = static_cast<int *>(cu_seqlens_d);
  params->seqlens = static_cast<int *>(seqlens_d);

  // S = softmax(P)
  params->s_ptr = s_d;
  params->s_stride_in_bytes = b * h * s * sizeof(DTYPE);

  // Set the dimensions.
  params->b = b;
  params->h = h;
  params->s = s;
  params->d = d;

  // Set the different scale values.
  const float scale_bmm1 = 1.f / sqrtf(d);
  constexpr float scale_softmax = 1.f;
  constexpr float scale_bmm2 = 1.f;

  set_alpha<ACCTYPE>(&params->scale_bmm1, scale_bmm1);
  set_alpha<ACCTYPE>(&params->scale_softmax, scale_softmax);
  set_alpha<DTYPE>(&params->scale_bmm2, scale_bmm2);

  // Set this to probability of keeping an element to simplify things.
  params->p_dropout = 1.f - p_dropout;
  params->rp_dropout = 1.f / params->p_dropout;
  set_alpha<DTYPE>(&params->scale_dropout, params->rp_dropout);
}

template <typename Kernel_traits>
__global__ void fmha_fprop_fp16_train_kernel(
    Fused_multihead_attention_fprop_params params) {
  fmha::device_1xN<Kernel_traits, true>(params);
}

template <typename Kernel_traits>
__global__ void fmha_dgrad_fp16_kernel(
    Fused_multihead_attention_fprop_params params) {
  fmha::compute_dv_1xN<Kernel_traits>(params);
  fmha::compute_dq_dk_1xN<Kernel_traits>(params);
}

template <int MAX_SLEN, int HEAD_SIZE>
uint64_t get_nrandom_in_fprop() {
  constexpr int STEP = 16;
  using Kernel_traits = FMHA_kernel_traits<MAX_SLEN, HEAD_SIZE, STEP, 1,
                                           MAX_SLEN == 512 ? 8 : 4, 0x08u>;
  using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
  return (Kernel_traits::Cta_tile_p::N / Kernel_traits::Cta_tile_p::M + 1) *
         Mma_tile_p::MMAS_M * 2 * Mma_tile_p::MMAS_N * 4;
}

template <int MAX_SLEN, int HEAD_SIZE, typename Params>
void run_fmha_fprop_kernel(Params params, cudaStream_t stream) {
  constexpr int STEP = 16;

  using Kernel_traits = FMHA_kernel_traits<MAX_SLEN, HEAD_SIZE, STEP, 1,
                                           MAX_SLEN == 512 ? 8 : 4, 0x08u>;

  constexpr size_t smem_size_softmax = Kernel_traits::Cta_tile_p::M *
                                       Kernel_traits::Cta_tile_p::WARPS_N *
                                       sizeof(float);
  constexpr size_t smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
  constexpr size_t smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
  constexpr size_t smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;
  constexpr size_t smem_size =
      smem_size_q + std::max(smem_size_v, smem_size_o + smem_size_softmax);

  if (smem_size >= 48 * 1024) {
    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
        fmha_fprop_fp16_train_kernel<Kernel_traits>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  dim3 grid(params.h, params.b);
  fmha_fprop_fp16_train_kernel<
      Kernel_traits><<<grid, Kernel_traits::THREADS, smem_size, stream>>>(
      params);
}

template <int MAX_SLEN, int HEAD_SIZE, typename Params>
void run_fmha_dgrad_kernel(Params params, cudaStream_t stream) {
  constexpr int STEP = 16;

  using Kernel_traits = FMHA_kernel_traits<MAX_SLEN, HEAD_SIZE, STEP, 1,
                                           MAX_SLEN == 512 ? 8 : 4, 0x08u>;

  constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M *
                                    Kernel_traits::Cta_tile_p::WARPS_N *
                                    sizeof(float);
  constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
  constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
  constexpr int smem_size_o = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

  using Smem_tile_s =
      fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
  constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;

  constexpr int smem_size_dv =
      smem_size_s + 2 * smem_size_q + smem_size_v + smem_size_softmax;
  constexpr int smem_size_dq_dk =
      smem_size_s + smem_size_o + smem_size_q + smem_size_v;
  constexpr int smem_size = std::max(smem_size_dv, smem_size_dq_dk);

  if (smem_size >= 48 * 1024) {
    FMHA_CHECK_CUDA(cudaFuncSetAttribute(
        fmha_dgrad_fp16_kernel<Kernel_traits>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  dim3 grid(params.h, params.b);
  fmha_dgrad_fp16_kernel<
      Kernel_traits><<<grid, Kernel_traits::THREADS, smem_size, stream>>>(
      params);
}

template <typename T, typename Generator>
void run_fmha_fprop_fp16(size_t batch_size, size_t max_seqlen, size_t num_head,
                         size_t head_size, float dropout_rate, void *d_qkv,
                         void *d_cu_seqlens, void *d_seqlens, void *d_out,
                         void *d_softmax_mask, Generator generator,
                         cudaStream_t stream) {
  Fused_multihead_attention_fprop_params params;
  set_params<T, float>(
      &params, batch_size, max_seqlen, num_head, head_size,
      reinterpret_cast<void *>(d_qkv), reinterpret_cast<void *>(d_cu_seqlens),
      reinterpret_cast<void *>(d_seqlens), reinterpret_cast<void *>(d_out),
      reinterpret_cast<void *>(d_softmax_mask), dropout_rate);

  // get seed from generator
  uint64_t offset;

  if (head_size == 64) {
    if (max_seqlen == 128) {
      offset = get_nrandom_in_fprop<128, 64>();
    } else if (max_seqlen == 256) {
      offset = get_nrandom_in_fprop<256, 64>();
    } else if (max_seqlen == 384) {
      offset = get_nrandom_in_fprop<384, 64>();
    } else if (max_seqlen == 512) {
      offset = get_nrandom_in_fprop<512, 64>();
    } else {
      assert(false && "max_seqlen should be 128/256/384/512");
    }
  } else {
    assert(false && "head size should be 32/64.");
  }

  auto seed_offset = generator->IncrementOffset(offset);
  params.seed = seed_offset.first;
  params.offset = seed_offset.second;

  if (head_size == 64) {
    if (max_seqlen == 128) {
      run_fmha_fprop_kernel<128, 64>(params, stream);
    } else if (max_seqlen == 256) {
      run_fmha_fprop_kernel<256, 64>(params, stream);
    } else if (max_seqlen == 384) {
      run_fmha_fprop_kernel<384, 64>(params, stream);
    } else if (max_seqlen == 512) {
      run_fmha_fprop_kernel<512, 64>(params, stream);
    } else {
      assert(false && "max_seqlen should be 128/256/384/512");
    }
  } else {
    assert(false && "head size should be 32/64.");
  }
}

template <typename T>
void run_fmha_dgrad_fp16(size_t batch_size, size_t max_seqlen, size_t num_head,
                         size_t head_size, float dropout_rate, void *d_qkv,
                         void *d_cu_seqlens, void *d_seqlens, void *d_out,
                         void *d_softmax_mask, void *d_dqkv,
                         cudaStream_t stream) {
  Fused_multihead_attention_fprop_params params;
  set_params<T, float>(
      &params, batch_size, max_seqlen, num_head, head_size,
      reinterpret_cast<void *>(d_qkv), reinterpret_cast<void *>(d_cu_seqlens),
      reinterpret_cast<void *>(d_seqlens), reinterpret_cast<void *>(d_out),
      reinterpret_cast<void *>(d_softmax_mask), dropout_rate);

  set_alpha<float>(&params.scale_bmm1, 1.f);
  set_alpha<float>(&params.scale_softmax, 1.f / sqrtf(head_size));
  set_alpha<T>(&params.scale_bmm2, 1.f);
  params.dqkv_ptr = d_dqkv;

  if (head_size == 64) {
    if (max_seqlen == 128) {
      run_fmha_dgrad_kernel<128, 64>(params, stream);
    } else if (max_seqlen == 256) {
      run_fmha_dgrad_kernel<256, 64>(params, stream);
    } else if (max_seqlen == 384) {
      run_fmha_dgrad_kernel<384, 64>(params, stream);
    } else if (max_seqlen == 512) {
      run_fmha_dgrad_kernel<512, 64>(params, stream);
    } else {
      assert(false && "max_seqlen should be 128/256/384/512");
    }
  } else {
    assert(false && "head size should be 32/64.");
  }
}

template <typename DeviceContext, typename T>
class FMHAGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;

    auto &device_ctx = context.template device_context<DeviceContext>();

    auto *x = context.Input<framework::Tensor>("X");
    auto *seqlen = context.Input<framework::Tensor>("Seqlen");
    auto *cu_seqlen = context.Input<framework::Tensor>("Cu_seqlen");
    auto *output = context.Output<framework::Tensor>("Out");
    auto *softmax_mask = context.Output<framework::Tensor>("SoftmaxMask");

    auto *d_x = x->data<T>();
    auto *d_seqlen = seqlen->data<int32_t>();
    auto *d_cu_seqlen = cu_seqlen->data<int32_t>();
    auto *d_output = output->mutable_data<T>(context.GetPlace());
    auto *d_softmax_mask = softmax_mask->mutable_data<T>(context.GetPlace());

    auto batch_size = seqlen->dims()[0];

    int device_id =
        BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace()).GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);

    // To complete this op, place following params to attributes
    // Here we only validate run_fmha_fprop_fp16()
    constexpr size_t MAX_SEQ_LEN = 128;
    constexpr size_t NUM_HEAD = 16;
    constexpr size_t HEAD_SIZE = 64;

    run_fmha_fprop_fp16<T>(
        batch_size, MAX_SEQ_LEN, NUM_HEAD, HEAD_SIZE, 0.f,
        reinterpret_cast<void *>(const_cast<T *>(d_x)),
        reinterpret_cast<void *>(const_cast<int32_t *>(d_cu_seqlen)),
        reinterpret_cast<void *>(const_cast<int32_t *>(d_seqlen)),
        reinterpret_cast<void *>(d_output),
        reinterpret_cast<void *>(d_softmax_mask), gen_cuda,
        device_ctx.stream());
  }
};

template <typename DeviceContext, typename T>
class FMHAGradGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &device_ctx = context.template device_context<DeviceContext>();

    auto *grad_x =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *grad_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *x = context.Input<framework::Tensor>("X");
    auto *seqlen = context.Input<framework::Tensor>("Seqlen");
    auto *cu_seqlen = context.Input<framework::Tensor>("Cu_seqlen");
    auto *softmax_mask = context.Input<framework::Tensor>("SoftmaxMask");

    auto *d_x = x->data<T>();
    auto *d_seqlen = seqlen->data<int32_t>();
    auto *d_cu_seqlen = cu_seqlen->data<int32_t>();
    auto *d_softmax_mask = softmax_mask->data<T>();
    auto *d_grad_out = grad_out->data<T>();
    auto *d_grad_x = grad_x->mutable_data<T>(context.GetPlace());

    auto batch_size = seqlen->dims()[0];

    // To complete this op, place following params to attributes
    // Here we only validate run_fmha_dgrad_fp16()
    constexpr size_t MAX_SEQ_LEN = 128;
    constexpr size_t NUM_HEAD = 16;
    constexpr size_t HEAD_SIZE = 64;

    run_fmha_dgrad_fp16<T>(
        batch_size, MAX_SEQ_LEN, NUM_HEAD, HEAD_SIZE, 0.f,
        reinterpret_cast<void *>(const_cast<T *>(d_x)),
        reinterpret_cast<void *>(const_cast<int32_t *>(d_cu_seqlen)),
        reinterpret_cast<void *>(const_cast<int32_t *>(d_seqlen)),
        reinterpret_cast<void *>(const_cast<T *>(d_grad_out)),
        reinterpret_cast<void *>(const_cast<T *>(d_softmax_mask)),
        reinterpret_cast<void *>(d_grad_x), device_ctx.stream());
  }
};

template void run_fmha_fprop_fp16<
    paddle::platform::float16, std::shared_ptr<paddle::framework::Generator>>(
    size_t batch_size, size_t max_seqlen, size_t num_head, size_t head_size,
    float dropout_rate, void *d_qkv, void *d_cu_seqlens, void *d_seqlens,
    void *d_out, void *d_softmax_mask,
    std::shared_ptr<paddle::framework::Generator> generator,
    cudaStream_t stream);

template void run_fmha_dgrad_fp16<paddle::platform::float16>(
    size_t batch_size, size_t max_seqlen, size_t num_head, size_t head_size,
    float dropout_rate, void *d_qkv, void *d_cu_seqlens, void *d_seqlens,
    void *d_out, void *d_softmax_mask, void *d_dqkv, cudaStream_t stream);

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fmha,
                        ops::FMHAGPUKernel<paddle::platform::CUDADeviceContext,
                                           paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    fmha_grad, ops::FMHAGradGPUKernel<paddle::platform::CUDADeviceContext,
                                      paddle::platform::float16>);
