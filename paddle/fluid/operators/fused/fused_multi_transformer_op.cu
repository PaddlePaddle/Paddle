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

#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/attn_gemm.h"
#include "paddle/fluid/operators/fused/fmha_ref.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
static void AllReduce(framework::Tensor &tensor,  // NOLINT
                      const int ring_id,
                      const platform::CUDADeviceContext &ctx) {
  if (ring_id == -1) return;
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  auto dtype =
      platform::ToNCCLDataType(framework::TransToProtoVarType(tensor.dtype()));
  int64_t numel = tensor.numel();
  const void *sendbuff = tensor.data<T>();
  auto place = ctx.GetPlace();
  void *recvbuff = tensor.mutable_data<T>(place);
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
  auto stream = ctx.stream();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
      sendbuff, recvbuff, numel, dtype, ncclSum, comm->comm(), stream));
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "PaddlePaddle should compile with NCCL or RCCL when used tensor model "
      "parallel op."));
#endif
}

template <typename T>
class FusedMultiTransformerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;
    auto place = ctx.GetPlace();
    auto &dev_ctx = ctx.cuda_device_context();

    // 0. input
    auto *input_x = ctx.Input<Tensor>("X");
    const auto input_x_dims = input_x->dims();
    int bsz = input_x_dims[0];
    int seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    int bsz_seq = bsz * seq_len;
    VLOG(0) << "0. input";

    // 1. layer norm
    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");
    auto ln_scales = ctx.MultiInput<Tensor>("LnScale");
    auto ln_biases = ctx.MultiInput<Tensor>("LnBias");

    auto ln_compute = AttnLayerNorm<T>(dev_ctx, epsilon, bsz_seq, dim_embed);
    Tensor ln_mean, ln_var;
    auto *ln_mean_data = ln_mean.mutable_data<U>({bsz_seq}, place);
    auto *ln_var_data = ln_var.mutable_data<U>({bsz_seq}, place);
    VLOG(0) << "1. ln";

    // 2. qkv
    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto qkv_weights = ctx.MultiInput<Tensor>("QKVW");
    auto qkv_biases = ctx.MultiInput<Tensor>("QKVBias");
    const auto qkv_w_dims = qkv_weights[0]->dims();
    int num_head = qkv_w_dims[1];
    int dim_head = qkv_w_dims[2];
    int hidden_size = num_head * dim_head;
    int output_size = 3 * hidden_size;
    int input_size = dim_embed;

    bool compute_bias = qkv_biases.size() > 0;
    // (transA, transB, compute_bias) = (false, true, false)
    auto qkv_compute = AttnMatMul<T>(dev_ctx, false, true, bsz_seq, output_size,
                                     input_size, compute_bias);
    Tensor qkv_out;
    auto *qkv_out_data =
        qkv_out.mutable_data<T>({bsz, seq_len, 3, num_head, dim_head}, place);
    VLOG(0) << "2. qkv";

    // 3. fmha
    AttnDropoutParam attn_param(true, "upscale_in_train", 0.0, true, true, 0,
                                nullptr);
    auto fmha_compute =
        FMHARef<T>(dev_ctx, bsz, seq_len, num_head, dim_head, attn_param);
    auto *src_mask = ctx.Input<Tensor>("SrcMask");
    auto cache_kvs = ctx.MultiInput<Tensor>("CacheKV");
    auto cache_kv_outs = ctx.MultiOutput<Tensor>("CacheKVOut");

    auto out_seq_len = seq_len;
    if (cache_kvs.size() > 0) {
      // [2, batch_size, num_head, cache_seq_len, head_size]
      const auto cache_kv_dims = cache_kvs[0]->dims();
      out_seq_len += cache_kv_dims[3];
    }

    Tensor transpose_out_2, qk_out;
    auto *transpose_out_2_data = transpose_out_2.mutable_data<T>(
        {3, bsz, num_head, seq_len, dim_head}, place);
    auto *qk_out_data =
        qk_out.mutable_data<T>({bsz, num_head, seq_len, out_seq_len}, place);

    Tensor src_mask_out, softmax_out;
    Tensor attn_dropout_mask_out, attn_dropout_out;
    Tensor qktv_out, fmha_out;
    auto *src_mask_out_data = src_mask_out.mutable_data<T>(
        {bsz, num_head, seq_len, out_seq_len}, place);
    auto *softmax_out_data = softmax_out.mutable_data<T>(
        {bsz, num_head, seq_len, out_seq_len}, place);

    auto *attn_dropout_mask_out_data = attn_dropout_mask_out.mutable_data<T>(
        {bsz, num_head, seq_len, out_seq_len}, place);
    auto *attn_dropout_data_data = attn_dropout_out.mutable_data<T>(
        {bsz, num_head, seq_len, out_seq_len}, place);

    auto *qktv_out_data =
        qktv_out.mutable_data<T>({bsz, num_head, seq_len, dim_head}, place);
    auto *fmha_out_data =
        fmha_out.mutable_data<T>({bsz, seq_len, num_head, dim_head}, place);
    VLOG(0) << "3. fmha";

    // 4. out_linear
    auto out_linear_weights = ctx.MultiInput<Tensor>("OutLinearW");
    auto out_linear_biases = ctx.MultiInput<Tensor>("OutLinearBias");
    int ring_id = ctx.Attr<int>("ring_id");
    // (transA, transB, compute_bias) = (false, false, false)
    auto out_linear_compute = AttnMatMul<T>(dev_ctx, false, false, bsz_seq,
                                            dim_embed, hidden_size, false);
    VLOG(0) << "4. out_linear";

    // 5. ln(residual + bias)
    DropoutParam dropout_param2(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
        dev_ctx, bsz_seq, dim_embed, dropout_param2, epsilon);
    auto ffn_ln_scales = ctx.MultiInput<Tensor>("FFNLnScale");
    auto ffn_ln_biases = ctx.MultiInput<Tensor>("FFNLnBias");
    Tensor bias_dropout_residual_out, dropout_mask_out;
    auto *bias_dropout_residual_out_data =
        bias_dropout_residual_out.mutable_data<T>({bsz, seq_len, dim_embed},
                                                  place);
    auto *dropout_mask_out_data = dropout_mask_out.mutable_data<uint8_t>(
        {bsz, seq_len, dim_embed}, place);
    VLOG(0) << "5. ln(redis)";

    // 6. ffn matmul1
    auto ffn1_weights = ctx.MultiInput<Tensor>("FFN1Weight");
    auto ffn1_biases = ctx.MultiInput<Tensor>("FFN1Bias");
    auto ffn1_weight_dim = ffn1_weights[0]->dims();

    int dim_ffn = ffn1_weight_dim[1];
    auto ffn1_linear_compute = AttnMatMul<T>(dev_ctx, false, false, bsz_seq,
                                             dim_ffn, dim_embed, false);
    Tensor ffn1_out;
    auto *ffn1_out_data = ffn1_out.mutable_data<T>({bsz_seq, dim_ffn}, place);
    VLOG(0) << "6. ffn1";

    // 7. ffn act + bias
    DropoutParam ffn1_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutHelper<T, uint8_t> fused_act_dropout_helper(
        dev_ctx, bsz_seq, dim_ffn, ffn1_dropout_param);
    Tensor ffn1_dropout_out, ffn1_dropout_mask;
    auto *ffn1_dropout_out_data =
        ffn1_dropout_out.mutable_data<T>({bsz_seq, dim_ffn}, place);
    auto *ffn1_dropout_mask_data =
        ffn1_dropout_mask.mutable_data<uint8_t>({bsz_seq, dim_ffn}, place);
    VLOG(0) << "7. ffn act";

    // 8. ffn2 matmul
    auto ffn2_weights = ctx.MultiInput<Tensor>("FFN2Weight");
    auto ffn2_biases = ctx.MultiInput<Tensor>("FFN2Bias");
    auto ffn2_linear_compute = AttnMatMul<T>(dev_ctx, false, false, bsz_seq,
                                             dim_embed, dim_ffn, false);
    VLOG(0) << "8. ffn2 matmul";

    // 9. ffn2 residual bias
    DropoutParam ffn2_dropout_param(true, 0, true, true, 0.0, nullptr, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> ffn2_fused_dropout_helper(
        dev_ctx, bsz_seq, dim_embed, ffn2_dropout_param, epsilon);
    VLOG(0) << "9. ffn2 redis";

    // calc
    auto *out = ctx.Output<Tensor>("Out");
    auto *from_data = out->mutable_data<T>(place);
    Tensor *from_tensor = out;
    Tensor tmp_out;
    auto *tmp_out_data =
        tmp_out.mutable_data<T>({bsz, seq_len, dim_embed}, place);

    auto *x_data = input_x->data<T>();
    Tensor *buf0 = nullptr;
    Tensor *buf1 = nullptr;

    // step0:  x   --> buf1
    // step1: buf1 --> buf0
    // step2: buf0 --> buf1
    int layers = qkv_weights.size();
    if (layers & 1) {
      // odd, set buf1 as out
      buf0 = &tmp_out;
      buf1 = out;
    } else {
      // even, set buf0 as out
      buf0 = out;
      buf1 = &tmp_out;
    }

    for (int i = 0; i < layers; ++i) {
      // step1. layer_norm
      if (pre_layer_norm) {
        auto *ln_scale_data = ln_scales[i]->data<U>();
        auto *ln_bias_data = ln_biases[i]->data<U>();
        // TODO(wangxi): can remove mean var in inference
        ln_compute.ComputeForward(x_data, ln_scale_data, ln_bias_data,
                                  buf1->data<T>(), ln_mean_data, ln_var_data);
      } else {
        //        from_data = x_data;
        //        from_tensor = input_x;
      }
      VLOG(0) << "step1";

      // step2. qkv
      const Tensor *qkv_bias = qkv_biases.size() > 0 ? qkv_biases[i] : nullptr;
      qkv_compute.ComputeForward(qkv_weights[i], buf1, qkv_bias, &qkv_out,
                                 &qkv_out);
      VLOG(0) << "step2";

      // step3. fmha
      const Tensor *cache_kv = cache_kvs.size() > 0 ? cache_kvs[i] : nullptr;
      Tensor *cache_kv_out = cache_kv ? cache_kv_outs[i] : nullptr;
      if (cache_kv_out) {
        cache_kv_out->mutable_data<T>(place);
      }

      // TODO(wangxi): can remove dropout in inference
      fmha_compute.ComputeForward(qkv_out, cache_kv, src_mask, &transpose_out_2,
                                  cache_kv_out, &qk_out, &src_mask_out,
                                  &softmax_out, &attn_dropout_mask_out,
                                  &attn_dropout_out, &qktv_out, &fmha_out);
      VLOG(0) << "step3";

      // step4. out_linear
      out_linear_compute.ComputeForward(out_linear_weights[i], &fmha_out,
                                        nullptr, buf1, nullptr);
      AllReduce<T>(*buf1, ring_id, dev_ctx);
      VLOG(0) << "step4";

      // step5. ln(residual + dropout(input + bias))
      if (pre_layer_norm) {
        auto *ln_scale_data = ffn_ln_scales[i]->data<U>();
        auto *ln_bias_data = ffn_ln_biases[i]->data<U>();
        auto *out_linear_bias_data = out_linear_biases[i]->data<T>();

        // inplace
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            dev_ctx, buf1->data<T>(), x_data, out_linear_bias_data,
            ln_scale_data, ln_bias_data, bias_dropout_residual_out_data,
            dropout_mask_out_data, buf1->data<T>(), ln_mean_data, ln_var_data);
        // dropout_mask_out_data, from_data, ln_mean_data, ln_var_data);
      } else {
      }
      VLOG(0) << "step5";

      // step6. ffn matmul1
      // ffn1_linear_compute.ComputeForward(ffn1_weights[i], out, nullptr,
      ffn1_linear_compute.ComputeForward(ffn1_weights[i], buf1, nullptr,
                                         &ffn1_out, nullptr);
      VLOG(0) << "step6";

      // step7. act bias
      // TODO(wangxi): remove dropout mask in inference
      fused_act_dropout_helper.DropoutActBias(
          dev_ctx, ffn1_out_data, ffn1_biases[i]->data<T>(), "gelu",
          ffn1_dropout_out_data, ffn1_dropout_mask_data);
      VLOG(0) << "step7";

      // step8. ffn matmul2
      ffn2_linear_compute.ComputeForward(ffn2_weights[i], &ffn1_dropout_out,
                                         nullptr, buf1, nullptr);
      VLOG(0) << "step8.0";

      AllReduce<T>(*buf1, ring_id, dev_ctx);
      VLOG(0) << "step8.1";

      // step9. residual bias
      if (pre_layer_norm) {
        // TODO(wangxi): remove dropout mask in inference
        ffn2_fused_dropout_helper.ResidualDropoutBias(
            dev_ctx, buf1->data<T>(), bias_dropout_residual_out_data,
            ffn2_biases[i]->data<T>(), buf1->data<T>(), dropout_mask_out_data);
      } else {
      }
      VLOG(0) << "step9";
      x_data = buf1->data<T>();
      std::swap(buf0, buf1);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_multi_transformer,
                        ops::FusedMultiTransformerOpKernel<float>,
                        ops::FusedMultiTransformerOpKernel<double>,
                        ops::FusedMultiTransformerOpKernel<plat::float16>);
