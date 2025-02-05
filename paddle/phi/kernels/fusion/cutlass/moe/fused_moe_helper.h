
/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_kernels/moe_gemm/fused_moe_gemm_kernels.h"
#include "paddle/phi/kernels/fusion/cutlass/moe/fused_moe_op.h"
#include "paddle/phi/kernels/fusion/gpu/fused_multi_transformer_helper.cu.h"

namespace phi {

template <typename T, int VecSize>
__global__ void moe_token_type_ids_kernel(T *gating_output,
                                          const int *moe_token_type_ids_out,
                                          const int num_rows,
                                          const int num_experts,
                                          const int k) {
  const int moe_token_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (moe_token_index >= num_rows) {
    return;
  }

  gating_output[moe_token_index * 2] =
      gating_output[moe_token_index * 2] +
      (moe_token_type_ids_out[moe_token_index]) * -1e10;
  gating_output[moe_token_index * 2 + 1] =
      gating_output[moe_token_index * 2 + 1] +
      (1 - moe_token_type_ids_out[moe_token_index]) * -1e10;
}

template <typename T>
void moe_token_type_ids_kernelLauncher(T *gating_output,
                                       const int *moe_token_type_ids_out,
                                       const int num_rows,
                                       const int num_experts,
                                       const int k,
                                       cudaStream_t stream) {
  const int blocks = num_rows * k / 512 + 1;
  const int threads = 512;
  moe_token_type_ids_kernel<T, 1><<<blocks, 512, 0, stream>>>(
      gating_output, moe_token_type_ids_out, num_rows, num_experts, k);
}

namespace fusion {

template <typename T,
          typename nvT = typename phi::PDDataTypeTraits<T>::DataType>
class MoeHelper {
 public:
  MoeHelper(const GPUContext &dev_ctx,
            const std::string gemm_method,
            MoeGemmRunner<nvT, nvT> *fp16_moe_gemm_runner,
            MoeGemmRunner<nvT, uint8_t> *int8_moe_gemm_runner,
            MoeGemmRunner<nvT, cutlass::uint4b_t> *int4_moe_gemm_runner,
            int layernum = 0)
      : ctx(dev_ctx),
        gemm_method_(gemm_method),
        fp16_moe_gemm_runner_(fp16_moe_gemm_runner),
        int8_moe_gemm_runner_(int8_moe_gemm_runner),
        int4_moe_gemm_runner_(int4_moe_gemm_runner),
        layernum_(layernum) {}

  // --------      getWorkspaceSize      -------- //
  template <typename KeyT>
  size_t getWorkspaceSize(const int num_rows,
                          const int hidden_size,
                          const int inter_size,
                          const int num_experts,
                          const int k) {
    const int buf_size = AlignTo16(k * num_rows * hidden_size);
    const int interbuf_size = AlignTo16(k * num_rows * inter_size);
    const int padded_experts = AlignTo16(num_experts);
    const int num_moe_inputs = AlignTo16(k * num_rows);
    // softmax output, permuted_rows and permuted_experts have moved to outside
    // of moe kernel, allocate them in Encoder or Decoder before invoking
    // FfnLayer forward.
    size_t total_ws_bytes =
        5 * num_moe_inputs *
        sizeof(int);  // source_rows_, permuted_rows_, permuted_experts_
    total_ws_bytes += buf_size * sizeof(KeyT);  // permuted_data
    total_ws_bytes +=
        padded_experts * sizeof(int64_t);  // Hold total_rows_before_expert_

    const int bytes_for_fc1_result = interbuf_size * sizeof(KeyT);
    const int sorter_ws_size_bytes =
        AlignTo16(sorter_.getWorkspaceSize(num_rows));
    sorter_.update_num_experts(num_experts);

    int bytes_for_intermediate_and_sorting = bytes_for_fc1_result;
    if (sorter_ws_size_bytes > bytes_for_fc1_result) {
      int remaining_bytes =
          AlignTo16(sorter_ws_size_bytes - bytes_for_fc1_result);
      bytes_for_intermediate_and_sorting += remaining_bytes;
    }

    total_ws_bytes +=
        bytes_for_intermediate_and_sorting;  // intermediate (fc1) output + cub
                                             // sorting workspace

    int num_softmax_outs = 0;
    const bool is_pow_2 =
        (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
      num_softmax_outs = AlignTo16(num_rows * num_experts);
    }

    total_ws_bytes += num_softmax_outs * sizeof(float);

    return total_ws_bytes;
  }

  void ComputeFFN(const DenseTensor *X,
                  const DenseTensor *gate_weight,
                  const DenseTensor *ffn1_weight,
                  const DenseTensor *ffn1_scale,
                  const DenseTensor *ffn1_bias,
                  const DenseTensor *ffn2_weight,
                  const DenseTensor *ffn2_scale,
                  const DenseTensor *ffn2_bias,
                  const DenseTensor *moe_token_type_ids,
                  const int moe_topk,
                  const bool norm_topk_prob,
                  const std::string moe_type,
                  DenseTensor *output) {
    auto *input_activations = X->data<T>();
    auto *gating_weights = gate_weight->data<float>();
    const T *fc1_expert_biases = ffn1_bias ? ffn1_bias->data<T>() : nullptr;
    const T *fc2_expert_biases = ffn2_bias ? ffn2_bias->data<T>() : nullptr;

    auto *output_ = output->data<T>();
    auto stream = ctx.stream();

    auto input_dims = X->dims();
    auto ffn1_dims = ffn1_weight->dims();
    int token_num = 0;
    if (input_dims.size() == 3) {
      token_num = input_dims[0] * input_dims[1];
    } else {
      token_num = input_dims[0];
    }
    const int num_rows = token_num;

    const int hidden_size = ffn1_dims[1];
    int inter_dim = 0;
    if (moe_type == "qkv") {
      inter_dim = ffn1_dims[2] * ffn1_dims[3] * ffn1_dims[4];
    } else {
      inter_dim = ffn1_dims[2];
    }

    if (gemm_method_ == "weight_only_int4") {
      inter_dim = inter_dim * 2;
    }

    const int inter_size = inter_dim;
    const int num_experts = ffn1_dims[0];
    const int k = moe_topk;

    VLOG(4) << "num_rows: " << num_rows << "   " << hidden_size << "   "
            << inter_size << "    " << num_experts << "k " << k;

    DenseTensor gate_tensor = Empty<float>(ctx, {num_rows, num_experts});
    DenseTensor X_tensor = Empty<float>(ctx, {num_rows, hidden_size});

    int64_t bytes =
        getWorkspaceSize<T>(num_rows, hidden_size, inter_size, num_experts, k);
    VLOG(4) << "bytes ---- " << bytes;

    // Pointers
    int *expert_for_source_row;
    int *source_rows_;
    int *permuted_rows_;
    int *permuted_experts_;
    int *expanded_source_row_to_expanded_dest_row;

    T *permuted_data_;
    int64_t *total_rows_before_expert_;
    T *fc1_result_;
    float *softmax_out_;

    DenseTensor ws_ptr_tensor = Empty<int8_t>(ctx, {bytes});
    int8_t *ws_ptr = ws_ptr_tensor.data<int8_t>();

    const int buf_size = AlignTo16(k * num_rows * hidden_size);
    const int interbuf_size = AlignTo16(k * num_rows * inter_size);
    const int padded_experts = AlignTo16(num_experts);
    const int num_moe_inputs = AlignTo16(k * num_rows);

    expert_for_source_row = reinterpret_cast<int *>(ws_ptr);
    source_rows_ = expert_for_source_row + num_moe_inputs;
    permuted_rows_ = source_rows_ + num_moe_inputs;
    permuted_experts_ = permuted_rows_ + num_moe_inputs;
    expanded_source_row_to_expanded_dest_row =
        permuted_experts_ + num_moe_inputs;
    permuted_data_ = reinterpret_cast<T *>(
        expanded_source_row_to_expanded_dest_row + num_moe_inputs);
    total_rows_before_expert_ =
        reinterpret_cast<int64_t *>(permuted_data_ + buf_size);
    fc1_result_ =
        reinterpret_cast<T *>(total_rows_before_expert_ + padded_experts);

    const bool is_pow_2 =
        (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256) {
      softmax_out_ = reinterpret_cast<float *>(fc1_result_ + interbuf_size);
    } else {
      softmax_out_ = nullptr;
    }

    DenseTensor finished_tensor = Empty<bool>(ctx, {num_rows});
    bool *finished = finished_tensor.data<bool>();
    // set false
    funcs::SetConstant<GPUContext, bool> zero;
    zero(ctx, &finished_tensor, false);

    DenseTensor expert_scales_tensor_float = Empty<float>(ctx, {num_rows, k});
    float *expert_scales_float = expert_scales_tensor_float.data<float>();

    DenseTensor fc1_out_tensor = Empty<T>(ctx, {num_rows * k, inter_size});
    T *fc1_out = fc1_out_tensor.data<T>();

    VLOG(4) << " gemm_method_ :" << gemm_method_;

    DenseTensor mixgemm_workspace;
    auto gate_compute = GEMMHelper<float>(
        ctx, num_rows, num_experts, hidden_size, "None", false);

    CastKernel<T>(ctx, *X, DataType::FLOAT32, &X_tensor);

    gate_compute.Compute(&X_tensor,
                         gate_weight,
                         /*weight_scale*/ nullptr,
                         /*bias*/ nullptr,
                         &mixgemm_workspace,
                         &gate_tensor);

    float *gating_output = gate_tensor.data<float>();

    if (moe_token_type_ids) {
      VLOG(4) << "moe_token_type_ids is on";
      auto *moe_token_type_ids_out = moe_token_type_ids->data<int>();
      moe_token_type_ids_kernelLauncher<float>(gating_output,
                                               moe_token_type_ids_out,
                                               num_rows,
                                               num_experts,
                                               k,
                                               ctx.stream());
    }

    topk_gating_softmax_kernelLauncher<float>(gating_output,
                                              finished,
                                              expert_scales_float,
                                              softmax_out_,
                                              expert_for_source_row,
                                              source_rows_,
                                              num_rows,
                                              num_experts,
                                              k,
                                              ctx.stream());

    const int sorter_ws_size_bytes =
        AlignTo16(sorter_.getWorkspaceSize(k * num_rows));

    sorter_.run(fc1_result_,
                sorter_ws_size_bytes,
                expert_for_source_row,
                permuted_experts_,
                source_rows_,
                permuted_rows_,
                k * num_rows,
                false,
                ctx.stream());

    initialize_moe_routing_kernelLauncher(
        input_activations,
        permuted_data_,
        permuted_rows_,
        expanded_source_row_to_expanded_dest_row,
        num_rows,
        num_rows,
        hidden_size,
        k,
        ctx.stream());

    const int expanded_active_expert_rows = k * num_rows;

    compute_total_rows_before_expert<T>(permuted_experts_,
                                        input_activations,
                                        expanded_active_expert_rows,
                                        num_experts,
                                        total_rows_before_expert_,
                                        ctx.stream());

    using NvType = typename phi::PDDataTypeTraits<T>::DataType;

    VLOG(4) << " ENTER EXPERT \n";

    if (gemm_method_ == "weight_only_int8") {
      int8_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const uint8_t *>(ffn1_weight->data<int8_t>()),
          reinterpret_cast<const NvType *>(ffn1_scale->data<T>()),
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          ctx.stream());
    } else if (gemm_method_ == "weight_only_int4") {
      int4_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const cutlass::uint4b_t *>(
              ffn1_weight->data<int8_t>()),
          reinterpret_cast<const NvType *>(ffn1_scale->data<T>()),
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          ctx.stream());
    } else {
      fp16_moe_gemm_runner_->moe_gemm_bias_act(
          reinterpret_cast<NvType *>(permuted_data_),
          reinterpret_cast<const NvType *>(ffn1_weight->data<T>()),
          nullptr,
          reinterpret_cast<const NvType *>(fc1_expert_biases),
          reinterpret_cast<NvType *>(fc1_out),
          total_rows_before_expert_,
          expanded_active_expert_rows,
          inter_size,
          hidden_size,
          num_experts,
          "none",
          ctx.stream());
    }

    if (moe_type == "ffn") {
      DenseTensor act_out_tensor =
          Empty<T>(ctx, {num_rows * k, inter_size / 2});
      T *act_out = act_out_tensor.data<T>();

      DenseTensor fc2_output_tensor =
          Empty<T>(ctx, {k * num_rows, hidden_size});
      T *fc2_result = fc2_output_tensor.data<T>();

      const std::string act_type = "swiglu";
      auto bias_act_helper =
          BiasActHelper<T>(ctx, act_type, num_rows * k, inter_size);

      bias_act_helper.Compute(&fc1_out_tensor, nullptr, &act_out_tensor);

      if (gemm_method_ == "weight_only_int8") {
        int8_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const uint8_t *>(ffn2_weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(ffn2_scale->data<T>()),
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            ctx.stream());
      } else if (gemm_method_ == "weight_only_int4") {
        int4_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const cutlass::uint4b_t *>(
                ffn2_weight->data<int8_t>()),
            reinterpret_cast<const NvType *>(ffn2_scale->data<T>()),
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            ctx.stream());
      } else {
        fp16_moe_gemm_runner_->moe_gemm(
            reinterpret_cast<NvType *>(act_out),
            reinterpret_cast<const NvType *>(ffn2_weight->data<T>()),
            nullptr,
            reinterpret_cast<NvType *>(fc2_result),
            total_rows_before_expert_,
            expanded_active_expert_rows,
            hidden_size,
            inter_size / 2,
            num_experts,
            ctx.stream());
      }

      finalize_moe_routing_kernelLauncher(
          fc2_result,
          output_,
          fc2_expert_biases,
          reinterpret_cast<float *>(expert_scales_float),
          expanded_source_row_to_expanded_dest_row,
          expert_for_source_row,
          num_rows,
          hidden_size,
          k,
          static_cast<int>(1),
          norm_topk_prob,
          ctx.stream());
    } else {
      finalize_moe_routing_kernelLauncher(
          // fc2_result,
          fc1_out,
          output_,
          fc1_expert_biases,  // fc2_expert_biases,
          reinterpret_cast<float *>(expert_scales_float),
          expanded_source_row_to_expanded_dest_row,
          expert_for_source_row,
          num_rows,
          inter_size,
          k,
          static_cast<int>(0),
          norm_topk_prob,
          ctx.stream());
    }
    VLOG(4) << " Finished EXPERT \n";
  }

 private:
  const phi::GPUContext &ctx;
  std::string gemm_method_;
  MoeGemmRunner<nvT, nvT> *fp16_moe_gemm_runner_;
  MoeGemmRunner<nvT, uint8_t> *int8_moe_gemm_runner_;
  MoeGemmRunner<nvT, cutlass::uint4b_t> *int4_moe_gemm_runner_;
  int layernum_;
  CubKeyValueSorter sorter_;
};

}  // namespace fusion
}  // namespace phi
