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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

#include "paddle/fluid/platform/cuda_device_function.h"

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/fused/attention_layer_norm.h"
#include "paddle/fluid/operators/fused/fused_dropout_helper.h"
#include "paddle/fluid/operators/fused/cudnn_mha.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Add
template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b, const T& c) const { return a + b + c; }
};

template <typename T>
class FusedAttentionCuDNNFMHAOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using U = LayerNormParamType<T>;

    auto *input_x = ctx.Input<Tensor>("X");

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const auto has_bias = ctx.Attr<bool>("has_bias");
    const float epsilon = ctx.Attr<float>("epsilon");
    // const float ln2epsilon = ctx.Attr<float>("ln2epsilon");

    const auto input_x_dims = input_x->dims();

    auto *x_data = input_x->data<T>();
    auto *ln_scale = ctx.Input<Tensor>("LnScale");
    auto *ln_bias = ctx.Input<Tensor>("LnBias");
    auto *ln_mean = ctx.Output<Tensor>("LnMean");
    auto *ln_var = ctx.Output<Tensor>("LnVariance");
    auto *ln_out = ctx.Output<Tensor>("LnOut");
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());
    auto *ln_bias_data = (ln_bias == nullptr ? nullptr : ln_bias->data<U>());
    auto *ln_mean_data = ln_mean->mutable_data<U>(ctx.GetPlace());
    auto *ln_var_data = ln_var->mutable_data<U>(ctx.GetPlace());
    
    auto *dropout_mask_out = ctx.Output<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Output<Tensor>("BiasDropoutResidualOut");
    auto *dropout_mask_out_data =
        dropout_mask_out->mutable_data<uint8_t>(ctx.GetPlace());
    
    // auto *out_linear_bias = ctx.Input<Tensor>("OutLinearBias");
    // auto *out_linear_bias_data = out_linear_bias->data<T>();
    T *out_linear_bias_data = nullptr;
    auto *out_linear_out = ctx.Output<Tensor>("OutLinearOut");
    auto *out_linear_out_data = out_linear_out->mutable_data<T>(ctx.GetPlace());
    auto *final_out = ctx.Output<Tensor>("Y");
    auto *final_out_data = final_out->mutable_data<T>(ctx.GetPlace());

    int batch_size = input_x_dims[0];
    int max_seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];

    // mha
    const Tensor* mha_w = ctx.Input<Tensor>("W");
    const Tensor* mha_qo_slen = ctx.Input<Tensor>("QO_Seqlen"); 
    const Tensor* mha_kv_slen = ctx.Input<Tensor>("KV_Seqlen");

    float attn_dropout_rate = ctx.Attr<float>("attn_dropout_rate");
    int attn_heads = ctx.Attr<int>("attn_heads");  // must

    const Tensor*  attn_low_windows = ctx.Input<Tensor>("AttnLowWinHost"); 
    const Tensor*  attn_high_windows = ctx.Input<Tensor>("AttnHighWinHost"); 
    const Tensor*  attn_qo_seqlen = ctx.Input<Tensor>("QOSeqLenHost"); 
    const Tensor*  attn_kv_seqlen = ctx.Input<Tensor>("KVSeqLenHost"); 
     
    auto *reserve_space = ctx.Output<Tensor>("ReserveSpace");
    PADDLE_ENFORCE_NOT_NULL(
            reserve_space,
            platform::errors::NotFound(
                "The argument ReserveSpace of fused_attention_cudnn_fmha op is not found."));

    int num_head = attn_heads;
    int dim_head = dim_embed/num_head;

    int bsz_seq = batch_size * max_seq_len;
    // int hidden_size = num_head * dim_head;
    // int output_size = 3 * hidden_size;
    // int input_size = dim_embed;

    // int attn_vec_size = ctx.Attr<int>("attn_vec_size");
    // int attn_q_proj_size = ctx.Attr<int>("attn_q_proj_size");
    // int attn_k_proj_size = ctx.Attr<int>("attn_k_proj_size");
    // int attn_v_proj_size = ctx.Attr<int>("attn_v_proj_size");
    // int attn_o_proj_size = ctx.Attr<int>("attn_o_proj_size");
    // int attn_max_qo_seq_len = ctx.Attr<int>("attn_max_qo_seq_len");
    // int attn_max_kv_seq_len = ctx.Attr<int>("attn_max_kv_seq_len");
    // int attn_beam_size = ctx.Attr<int>("attn_beam_size");
    // double attn_sm_scaler =
    //     static_cast<double>(ctx.Attr<float>("attn_sm_scaler")); // must

    int attn_vec_size = dim_embed;
    int attn_q_proj_size = dim_head;
    int attn_k_proj_size = dim_head;
    int attn_v_proj_size = dim_head;
    int attn_o_proj_size = dim_embed;
    int attn_max_qo_seq_len = max_seq_len;
    int attn_max_kv_seq_len = max_seq_len;
    int attn_beam_size = 1;

   
    double attn_sm_scaler = 1.0/sqrt(dim_head); // 1/sqrt(dim_head)
    // double attn_sm_scaler = 1.0;

#if 0
    std::cout << "low_win = " << std::endl;
#if 0
    auto *attn_low_windows_data = attn_low_windows->data<int>();
    for (int i=0; i<max_seq_len; i++) {
        std::cout << attn_low_windows_data[i] << " "; 
    }
    std::cout << std::endl;
    std::cout << "high_win = " << std::endl;
    auto *attn_high_windows_data = attn_high_windows->data<int>();
    for (int i=0; i<max_seq_len; i++) {
        std::cout << attn_high_windows_data[i] << " "; 
    }
    std::cout << std::endl;
#endif
    std::cout << "low_windows dims = " << attn_low_windows->dims() << std::endl;
    std::cout << "high_windows dims = " << attn_high_windows->dims() << std::endl;
    std::cout << "seq_len_host dims = " << attn_qo_seqlen->dims() << std::endl;
#endif

   // memcpy:
#if 0
   Tensor attn_low_windows_temp;
   attn_low_windows_temp.Resize({max_seq_len});
   attn_low_windows_temp.mutable_data<int>(ctx.GetPlace());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_low_windows_temp.data<int>(),
      reinterpret_cast<const void*>(attn_low_windows_input->data<int>()),
      max_seq_len * sizeof(int), cudaMemcpyDeviceToHost));
  const Tensor* attn_low_windows = &attn_low_windows_temp;

  Tensor attn_high_windows_temp;
  attn_high_windows_temp.Resize({max_seq_len});
  attn_high_windows_temp.mutable_data<int>(ctx.GetPlace());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_high_windows_temp.data<int>(),
      reinterpret_cast<const void*>(attn_high_windows_input->data<int>()),
      max_seq_len * sizeof(int), cudaMemcpyDeviceToHost));
  const Tensor* attn_high_windows = &attn_high_windows_temp;

  Tensor attn_qo_seqlen_temp;
  attn_qo_seqlen_temp.Resize({batch_size});
  attn_qo_seqlen_temp.mutable_data<int>(ctx.GetPlace());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_qo_seqlen_temp.data<int>(),
      reinterpret_cast<const void*>(attn_qo_seqlen_input->data<int>()),
      batch_size * sizeof(int), cudaMemcpyDeviceToHost));
  const Tensor* attn_qo_seqlen = &attn_qo_seqlen_temp;

  Tensor attn_kv_seqlen_temp;
  attn_kv_seqlen_temp.Resize({batch_size});
  attn_kv_seqlen_temp.mutable_data<int>(ctx.GetPlace());
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_kv_seqlen_temp.data<int>(),
      reinterpret_cast<const void*>(attn_kv_seqlen_input->data<int>()),
      batch_size * sizeof(int), cudaMemcpyDeviceToHost));
  const Tensor* attn_kv_seqlen = &attn_kv_seqlen_temp;
#endif


#if 1
    auto layer_norm_compute = AttnLayerNorm<T>(ctx.cuda_device_context(),
                                               epsilon, bsz_seq, dim_embed);
    
    DropoutParam dropout_param2(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
            ctx.cuda_device_context(), bsz_seq, dim_embed, dropout_param2,
            epsilon);

    // compute
    if (pre_layer_norm) {
        auto *ln_out_data = ln_out->mutable_data<T>(ctx.GetPlace());

        layer_norm_compute.ComputeForward(x_data, ln_scale_data, ln_bias_data,
                                      ln_out_data, ln_mean_data, ln_var_data);
        // the input of cudnn fmha is ln_out.
        MHAFwKernel<T>(ctx.cuda_device_context(), has_bias,
                    ln_out, ln_out, ln_out, 
                    mha_w, mha_qo_slen, mha_kv_slen,
                    attn_dropout_rate, attn_heads, attn_sm_scaler,
                    attn_vec_size, attn_q_proj_size, attn_k_proj_size,
                    attn_v_proj_size, attn_o_proj_size,
                    attn_max_qo_seq_len, attn_max_kv_seq_len,
                    attn_beam_size, attn_low_windows,
                    attn_high_windows, out_linear_out,
                    attn_qo_seqlen, attn_kv_seqlen, reserve_space);
    } else {
        // the input of cudnn fmha is x
        MHAFwKernel<T>(ctx.cuda_device_context(), has_bias,
            input_x, input_x, input_x, 
            mha_w, mha_qo_slen, mha_kv_slen,
            attn_dropout_rate, attn_heads, attn_sm_scaler,
            attn_vec_size, attn_q_proj_size, attn_k_proj_size,
            attn_v_proj_size, attn_o_proj_size,
            attn_max_qo_seq_len, attn_max_kv_seq_len,
            attn_beam_size, attn_low_windows,
            attn_high_windows, out_linear_out,
            attn_qo_seqlen, attn_kv_seqlen, reserve_space);
    }
    // out = layernorm(residual + dropout(src + bias))
    if (pre_layer_norm) {
        // output = (residual + dropout(input + bias))
        fused_dropout_layernorm_helper.ResidualDropoutBias(
          ctx.cuda_device_context(), out_linear_out_data, x_data,
          out_linear_bias_data, final_out_data, dropout_mask_out_data);
    } else {
        auto *bias_dropout_residual_out_data =
            bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
            
        // output = layernorm(residual + dropout(input + bias))
        fused_dropout_layernorm_helper.LayernormResidualDropoutBias(
            ctx.cuda_device_context(), out_linear_out_data, x_data,
            out_linear_bias_data, ln_scale_data, ln_bias_data,
            bias_dropout_residual_out_data, dropout_mask_out_data, 
            final_out_data, ln_mean_data, ln_var_data);
    }
   
#endif

  }
};

template <typename T>
class FusedAttentionCuDNNFMHAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if CUDNN_VERSION >= 8000
    using U = LayerNormParamType<T>;

    const auto pre_layer_norm = ctx.Attr<bool>("pre_layer_norm");
    const float epsilon = ctx.Attr<float>("epsilon");

    // get inputs.
    auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto *d_y_data = d_y->data<T>();

    // fw input
    auto *input_x = ctx.Input<Tensor>("X");
    auto *ln_scale = ctx.Input<Tensor>("LnScale");

    auto *x_data = input_x->data<T>();
    auto *ln_scale_data = (ln_scale == nullptr ? nullptr : ln_scale->data<U>());

    // fw output
    auto *ln_mean = ctx.Input<Tensor>("LnMean");
    auto *ln_var = ctx.Input<Tensor>("LnVariance");
    auto *ln_out = ctx.Input<Tensor>("LnOut");
    auto *out_linear_out = ctx.Input<Tensor>("OutLinearOut");
    auto *dropout_mask_out = ctx.Input<Tensor>("DropoutMaskOut");
    auto *bias_dropout_residual_out =
        ctx.Input<Tensor>("BiasDropoutResidualOut");

    auto *ln_mean_data = ln_mean->data<U>();
    auto *ln_var_data = ln_var->data<U>();

    auto *out_linear_out_data = out_linear_out->data<T>();
    auto *dropout_mask_out_data = dropout_mask_out->data<uint8_t>();
    

    // bw output's grad
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_ln_out = ctx.Output<Tensor>(framework::GradVarName("LnOut"));
    auto *d_out_linear_out =
        ctx.Output<Tensor>(framework::GradVarName("OutLinearOut"));
    auto *d_bias_dropout_residual_out =
        ctx.Output<Tensor>(framework::GradVarName("BiasDropoutResidualOut"));

    auto *d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
   
    auto *d_out_linear_out_data =
        d_out_linear_out->mutable_data<T>(ctx.GetPlace());
   

    // bw parameter's grad
    auto *d_ln_scale = ctx.Output<Tensor>(framework::GradVarName("LnScale"));
    auto *d_ln_bias = ctx.Output<Tensor>(framework::GradVarName("LnBias"));
    // auto *d_out_linear_bias =
    //        ctx.Output<Tensor>(framework::GradVarName("OutLinearBias"));

    auto *d_ln_scale_data =
        (d_ln_scale == nullptr ? nullptr
                               : d_ln_scale->mutable_data<U>(ctx.GetPlace()));
    auto *d_ln_bias_data =
        (d_ln_bias == nullptr ? nullptr
                              : d_ln_bias->mutable_data<U>(ctx.GetPlace()));
    // auto *d_out_linear_bias_data =
    //        d_out_linear_bias->mutable_data<T>(ctx.GetPlace());        
    T *d_out_linear_bias_data = nullptr;         
    
    // std::vector<int> attn_low_windows =
    //     ctx.Attr<std::vector<int>>("attn_low_windows");
    // std::vector<int> attn_high_windows =
    //    ctx.Attr<std::vector<int>>("attn_high_windows");
    const Tensor*  attn_low_windows = ctx.Input<Tensor>("AttnLowWinHost"); 
    const Tensor*  attn_high_windows = ctx.Input<Tensor>("AttnHighWinHost"); 

    const Tensor* mha_w = ctx.Input<Tensor>("W");
    const Tensor* mha_qo_slen = ctx.Input<Tensor>("QO_Seqlen");
    const Tensor* mha_kv_slen = ctx.Input<Tensor>("KV_Seqlen");
    const auto *reserve_space = ctx.Input<Tensor>("ReserveSpace");

    Tensor* d_mha_w = ctx.Output<Tensor>(framework::GradVarName("W"));
    d_mha_w->mutable_data<T>(ctx.GetPlace());

    const auto input_x_dims = input_x->dims();

    int batch_size = input_x_dims[0];
    int max_seq_len = input_x_dims[1];
    int dim_embed = input_x_dims[2];
    
    int attn_heads = ctx.Attr<int>("attn_heads");
    int num_head = attn_heads;
    int dim_head = dim_embed/num_head;

    int bsz_seq = batch_size * max_seq_len;
    
    Tensor d_residual;
    d_residual.Resize(input_x_dims);
    T *d_residual_data = d_residual.mutable_data<T>(ctx.GetPlace());

    Tensor d_k;
    d_k.Resize(input_x_dims);
    T *d_k_data = d_k.mutable_data<T>(ctx.GetPlace());

    Tensor d_v;
    d_v.Resize(input_x_dims);
    T *d_v_data = d_v.mutable_data<T>(ctx.GetPlace());
    
#if 0
    std::cout << "low_win = " << std::endl;
    auto *attn_low_windows_data = attn_low_windows->data<int>();
    for (int i=0; i<max_seq_len; i++) {
        std::cout << attn_low_windows_data[i] << " "; 
    }
    std::cout << std::endl;
     std::cout << "high_win = " << std::endl;
    auto *attn_high_windows_data = attn_high_windows->data<int>();
    for (int i=0; i<max_seq_len; i++) {
        std::cout << attn_high_windows_data[i] << " "; 
    }
    std::cout << std::endl;
#endif

#if 0
    Tensor attn_low_windows_temp;
    attn_low_windows_temp.Resize({max_seq_len});
    attn_low_windows_temp.mutable_data<int>(ctx.GetPlace());
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        attn_low_windows_temp.data<int>(),
        reinterpret_cast<const void*>(attn_low_windows_input->data<int>()),
        max_seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    const Tensor* attn_low_windows = &attn_low_windows_temp;

    Tensor attn_high_windows_temp;
    attn_high_windows_temp.Resize({max_seq_len});
    attn_high_windows_temp.mutable_data<int>(ctx.GetPlace());
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        attn_high_windows_temp.data<int>(),
        reinterpret_cast<const void*>(attn_high_windows_input->data<int>()),
        max_seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    const Tensor* attn_high_windows = &attn_high_windows_temp;
#endif

    auto layer_norm_compute = AttnLayerNorm<T>(ctx.cuda_device_context(),
                                            epsilon, bsz_seq, dim_embed);

    DropoutParam dropout_param2(ctx, 0);
    FusedDropoutLayerNormHelper<T, uint8_t> fused_dropout_layernorm_helper(
                            ctx.cuda_device_context(), bsz_seq, dim_embed, 
                            dropout_param2, epsilon);                                       
    
    if (pre_layer_norm) {
        fused_dropout_layernorm_helper.ResidualDropoutBiasGrad(
            ctx.cuda_device_context(), d_y_data, dropout_mask_out_data,
            d_out_linear_out_data, d_residual_data, d_out_linear_bias_data);
    } else {
        auto *bias_dropout_residual_out_data = bias_dropout_residual_out->data<T>();
        auto *d_bias_dropout_residual_out_data =
            d_bias_dropout_residual_out->mutable_data<T>(ctx.GetPlace());
        fused_dropout_layernorm_helper.LayernormResidualDropoutBiasGrad(
            ctx.cuda_device_context(), d_y_data, bias_dropout_residual_out_data,
            dropout_mask_out_data, ln_scale_data, ln_mean_data, ln_var_data,
            d_bias_dropout_residual_out_data, d_ln_scale_data, d_ln_bias_data,
            d_out_linear_out_data, d_out_linear_bias_data, d_residual_data);
    }


#if 1
    if (pre_layer_norm) {
        auto *ln_out_data = ln_out->data<T>();
        auto *d_ln_out_data = 
            (d_ln_out == nullptr ? nullptr
                                : d_ln_out->mutable_data<T>(ctx.GetPlace()));

        MHAGradKernel<T>(ctx.cuda_device_context(),
            d_out_linear_out, input_x, input_x, input_x, 
            mha_w, mha_qo_slen, mha_kv_slen, 
            attn_low_windows, attn_high_windows, 
            d_ln_out, &d_k, &d_v, d_mha_w, reserve_space);
        
        std::vector<const Tensor *> ins;
        std::vector<Tensor *> outs;
        ins.emplace_back(d_ln_out);
        ins.emplace_back(&d_k);
        ins.emplace_back(&d_v);
        outs.emplace_back(d_ln_out);
        int elewise_add_axis = -1;

        LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
            ctx.cuda_device_context(), ins, &outs, TernaryAddFunctor<T>());

        layer_norm_compute.ComputeBackward(x_data, d_ln_out_data, ln_scale_data,
                                            ln_mean_data, ln_var_data, d_x_data,
                                            d_ln_scale_data, d_ln_bias_data);
        
    } else {
        MHAGradKernel<T>(ctx.cuda_device_context(),
            d_out_linear_out, input_x, input_x, input_x, 
            mha_w, mha_qo_slen, mha_kv_slen, 
            attn_low_windows, attn_high_windows, 
            d_x, &d_k, &d_v, d_mha_w, reserve_space);
        
        std::vector<const Tensor *> ins;
        std::vector<Tensor *> outs;
        ins.emplace_back(d_x);
        ins.emplace_back(&d_k);
        ins.emplace_back(&d_v);
        outs.emplace_back(d_x);
        int elewise_add_axis = -1;

        LaunchSameDimsElementwiseCudaKernel<ElementwiseType::kTernary, T, T>(
            ctx.cuda_device_context(), ins, &outs, TernaryAddFunctor<T>());

    }
    // gradient accumulation: d_x[] + d_residual[] = d_x[]
    std::vector<const Tensor *> ins;
    std::vector<Tensor *> outs;
    ins.emplace_back(&d_residual);
    ins.emplace_back(d_x);
    outs.emplace_back(d_x);
    int elewise_add_axis = -1;
    LaunchElementwiseCudaKernel<ElementwiseType::kBinary, T, T>(
                            ctx.cuda_device_context(), ins, &outs, 
                            elewise_add_axis, AddFunctor<T>());
#endif
    
#endif
    }
};


}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 8000
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(fused_attention_cudnn_fmha, ops::FusedAttentionCuDNNFMHAOpKernel<float>,
                        ops::FusedAttentionCuDNNFMHAOpKernel<double>,
                        ops::FusedAttentionCuDNNFMHAOpKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(fused_attention_cudnn_fmha_grad,
                        ops::FusedAttentionCuDNNFMHAGradKernel<float>,
                        ops::FusedAttentionCuDNNFMHAGradKernel<double>,
                        ops::FusedAttentionCuDNNFMHAGradKernel<plat::float16>);
#endif
