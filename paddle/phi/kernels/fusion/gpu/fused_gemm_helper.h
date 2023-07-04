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
#pragma once

#include "paddle/phi/common/datatype_traits.h"
#if defined(PADDLE_WITH_CUTLASS)
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"
#endif

#ifndef PADDLE_WITH_HIP
#include "paddle/phi/kernels/impl/llm_int8_mat_mul_kernel_impl.h"
#endif

#if defined(PADDLE_WITH_CUDA)
#include "paddle/phi/kernels/fusion/gpu/attn_gemm.h"
#endif

#include "paddle/phi/kernels/gemv_weightonly_int8_kernel.h"

namespace phi {
namespace fusion {

template <typename T, typename nvT = typename PDDataTypeTraits<T>::DataType>
class GEMMHelper {
 public:
  GEMMHelper(
      const GPUContext &dev_ctx,
      int token_num,
      int dim_ffn,
      int dim_embed,
      const std::string gemm_method,
      CutlassFpAIntBGemmRunner<nvT, uint8_t> *int8_mixed_gemm_runner,
      CutlassFpAIntBGemmRunner<nvT, cutlass::uint4b_t> *int4_mixed_gemm_runner,
      bool transpose_weight = false)
      : dev_ctx_(dev_ctx),
        token_num_(token_num),
        dim_ffn_(dim_ffn),
        dim_embed_(dim_embed),
        gemm_method_(gemm_method),
        int8_mixed_gemm_runner_(int8_mixed_gemm_runner),
        int4_mixed_gemm_runner_(int4_mixed_gemm_runner),
        transpose_weight_(transpose_weight) {}

  // dst = act(fc(src[0]) + bias) * src[1]
  void Compute(const DenseTensor *input,
               const DenseTensor *weight,
               const DenseTensor *scale,
               const DenseTensor *bias,
               DenseTensor *workspace,
               DenseTensor *output) {
    VLOG(5) << "GEMMHelper,"
            << " token_num_:" << token_num_ << " dim_ffn_:" << dim_ffn_
            << " dim_embed_:" << dim_embed_;
    bool compute_bias = true;
    if (bias == nullptr) {
      compute_bias = false;
    }
    using NvType = typename PDDataTypeTraits<T>::DataType;

    if (gemm_method_ == "weight-only-int8") {
      VLOG(5) << "do weight-only gemm int8";
      if (bias) {
        int8_mixed_gemm_runner_->gemm_bias_act(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
            scale->data<float>(),
            reinterpret_cast<const NvType *>(bias->data<T>()),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            "none",
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      } else {
        int8_mixed_gemm_runner_->gemm(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const uint8_t *>(weight->data<int8_t>()),
            scale->data<float>(),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      }
      VLOG(5) << "input:" << *input;
      VLOG(5) << "output:" << *output;
    } else if (gemm_method_ == "weight-only-int4") {
      VLOG(5) << "do weight-only gemm";
      if (bias) {
        int4_mixed_gemm_runner_->gemm_bias_act(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const cutlass::uint4b_t *>(weight->data<int8_t>()),
            scale->data<float>(),
            reinterpret_cast<const NvType *>(bias->data<T>()),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            "none",
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      } else {
        int4_mixed_gemm_runner_->gemm(
            reinterpret_cast<const NvType *>(input->data<T>()),
            reinterpret_cast<const cutlass::uint4b_t *>(weight->data<int8_t>()),
            scale->data<float>(),
            reinterpret_cast<NvType *>(output->data<T>()),
            token_num_,
            dim_ffn_,
            dim_embed_,
            reinterpret_cast<char *>(workspace->data<uint8_t>()),
            workspace->numel(),
            dev_ctx_.stream());
      }
      VLOG(5) << "input:" << *input;
      VLOG(5) << "output:" << *output;
    } else if (gemm_method_ == "weightonly_gemv") {
      // TODO(zhengzekang): support weightonly gemv int4
      const T *bias_data = bias ? bias->data<T>() : nullptr;
      GemvWeightonlyInt8Wrapper<T, GPUContext>(dev_ctx_,
                                               input->data<T>(),
                                               weight->data<int8_t>(),
                                               bias_data,
                                               scale->data<float>(),
                                               dim_ffn_,
                                               dim_embed_,
                                               "None", /*act_method*/
                                               output->data<T>());
    } else if (gemm_method_ == "LLM.int8") {
      // Note(Zhengzekang): LLM Gemm donot support fused add_bias.
      LLMGemm<T>(dev_ctx_,
                 weight,
                 input,
                 scale,
                 FLAGS_custom_llm_int8_threshold,
                 output,
                 workspace,
                 "LLMGemm",
                 token_num_,
                 dim_embed_,
                 dim_ffn_);
    } else if (gemm_method_ == "None") {
      auto ffn_linear_compute = AttnMatMul<T>(dev_ctx_,
                                              false,
                                              transpose_weight_,
                                              token_num_,
                                              dim_ffn_,
                                              dim_embed_,
                                              compute_bias);
      ffn_linear_compute.ComputeForward(weight, input, bias, output, output);
    } else {
      PADDLE_THROW(errors::Unimplemented(
          "Currently GemmHelper only support `weight-only`, `LLM.int8`, "
          "`None`. "));
    }
  }

 private:
  const GPUContext &dev_ctx_;
  int token_num_;
  int dim_ffn_;
  int dim_embed_;
  std::string gemm_method_;
  CutlassFpAIntBGemmRunner<nvT, uint8_t> *int8_mixed_gemm_runner_;
  CutlassFpAIntBGemmRunner<nvT, cutlass::uint4b_t> *int4_mixed_gemm_runner_;
  bool transpose_weight_;  // Just For AttnMatmul.
};

}  // namespace fusion
}  // namespace phi
