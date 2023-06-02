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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/cudnn_frontend.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/gpudnn/mha_cudnn_frontend.h"

#include "paddle/fluid/platform/device/gpu/cuda/cudnn_desc.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class FusedDotProductSelfAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_GE(cudnn_version,
                      8901,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8901, "
                          "but got %d.",
                          cudnn_version));
    // inputs
    auto *qkv = ctx.Input<Tensor>("QKV");
    auto *q_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenQ");
    auto *kv_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenKV");

    // attrs
    float scaling_factor = ctx.Attr<float>("scaling_factor");
    float dropout_probability = ctx.Attr<float>("attn_dropout_rate");
    int seed = ctx.Attr<int>("attn_dropout_seed");
    bool is_causal_masking = ctx.Attr<bool>("is_causal_masking");

    // outputs
    auto *out = ctx.Output<Tensor>("Out");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *softmax_out_ptr = dev_ctx.template Alloc<T>(
        softmax_out, softmax_out->numel() * sizeof(T));
    auto *out_ptr = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    // get handles
    auto handle = dev_ctx.cudnn_handle();

    auto tensor_dtype =
        platform::ToCudnnDataType(framework::TransToProtoVarType(qkv->dtype()));
    bool is_type_supported = (tensor_dtype == CUDNN_DATA_HALF ||
                              tensor_dtype == CUDNN_DATA_BFLOAT16);
    PADDLE_ENFORCE_EQ(is_type_supported,
                      true,
                      platform::errors::InvalidArgument(
                          "cuDNN FMHA Only supports FP16/BF16 currently"));
    auto mha_layout = MHA_Layout::QKV_INTERLEAVED;
    auto bias_type = MHA_Bias_Type::NO_BIAS;  // TODO(Shijie Wang): support bias
    std::vector<cudnn_frontend::Operation const *> all_ops;
    std::vector<cudnn_frontend::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // qkv dim: {b, s, 3, h, d};
    auto batch_size = qkv->dims()[0];
    auto seq_len = qkv->dims()[1];
    auto num_heads = qkv->dims()[3];
    auto head_size = qkv->dims()[4];

    // only support seqlen >= 64 and seqlen <= 512 and seqlen % 64 == 0
    // currently
    bool can_divide_by_64 = (seq_len % 64 == 0);
    bool is_seqlen_supported =
        (seq_len >= 64 && seq_len <= 512 && can_divide_by_64);
    PADDLE_ENFORCE_EQ(
        is_seqlen_supported,
        true,
        platform::errors::InvalidArgument(
            "cuDNN FMHA only supports sequence length >= 64 and "
            "sequence length <= 512 and sequence length % 64 == 0, "
            "but got sequence length: %d.",
            seq_len));

    T *qkv_dev_ptr = const_cast<T *>(qkv->data<T>());
    void *q_dev_ptr = reinterpret_cast<void *>(qkv_dev_ptr);
    void *k_dev_ptr =
        reinterpret_cast<void *>(qkv_dev_ptr + num_heads * head_size);
    void *v_dev_ptr =
        reinterpret_cast<void *>(qkv_dev_ptr + 2 * num_heads * head_size);
    void *q_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(q_actual_seqlen->data<int>()));
    void *kv_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(kv_actual_seqlen->data<int>()));
    void *out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(out->data<T>()));
    void *softmax_out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(softmax_out->data<T>()));
    void *bias_dev_ptr = nullptr;

    run_cudnn_fmha_fwd(batch_size,
                       num_heads,
                       seq_len,
                       seq_len,
                       head_size,
                       seed,
                       mha_layout,
                       scaling_factor,
                       dropout_probability,
                       bias_type,
                       is_causal_masking,
                       q_dev_ptr,
                       k_dev_ptr,
                       v_dev_ptr,
                       softmax_out_dev_ptr,
                       out_dev_ptr,
                       bias_dev_ptr,
                       q_actual_seqlen_dev_ptr,
                       kv_actual_seqlen_dev_ptr,
                       tensor_dtype,
                       handle);
  }
};

template <typename T>
class FusedDotProductCrossAttentionOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_GE(cudnn_version,
                      8901,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8901, "
                          "but got %d.",
                          cudnn_version));
    // inputs
    auto *q = ctx.Input<Tensor>("Q");
    auto *kv = ctx.Input<Tensor>("KV");
    auto *q_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenQ");
    auto *kv_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenKV");

    // attrs
    float scaling_factor = ctx.Attr<float>("scaling_factor");
    float dropout_probability = ctx.Attr<float>("attn_dropout_rate");
    int seed = ctx.Attr<int>("attn_dropout_seed");
    bool is_causal_masking = ctx.Attr<bool>("is_causal_masking");

    // outputs
    auto *out = ctx.Output<Tensor>("Out");
    auto *softmax_out = ctx.Output<Tensor>("SoftmaxOut");
    auto *softmax_out_ptr = dev_ctx.template Alloc<T>(
        softmax_out, softmax_out->numel() * sizeof(T));
    auto *out_ptr = dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    // get handles
    auto handle = dev_ctx.cudnn_handle();

    auto tensor_dtype =
        platform::ToCudnnDataType(framework::TransToProtoVarType(q->dtype()));
    bool is_type_supported = (tensor_dtype == CUDNN_DATA_HALF ||
                              tensor_dtype == CUDNN_DATA_BFLOAT16);
    PADDLE_ENFORCE_EQ(is_type_supported,
                      true,
                      platform::errors::InvalidArgument(
                          "cuDNN FMHA Only supports FP16/BF16 currently"));
    auto mha_layout = MHA_Layout::KV_INTERLEAVED;
    auto bias_type = MHA_Bias_Type::NO_BIAS;  // TODO(Shijie Wang): support bias
    std::vector<cudnn_frontend::Operation const *> all_ops;
    std::vector<cudnn_frontend::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // q dim: {b, s_q, h, d};
    // kv dim: {b, s_kv, 2, h, d};
    auto batch_size = q->dims()[0];
    auto q_seq_len = q->dims()[1];
    auto num_heads = q->dims()[2];
    auto head_size = q->dims()[3];
    auto kv_seq_len = kv->dims()[1];

    // only support seqlen >= 64 and seqlen <= 512 and seqlen % 64 == 0
    // currently
    bool can_divide_by_64 = (q_seq_len % 64 == 0 && kv_seq_len % 64 == 0);
    bool is_seqlen_supported =
        (q_seq_len >= 64 && q_seq_len <= 512 && kv_seq_len >= 64 &&
         kv_seq_len <= 512 && can_divide_by_64);
    PADDLE_ENFORCE_EQ(
        is_seqlen_supported,
        true,
        platform::errors::InvalidArgument(
            "cuDNN FMHA only supports sequence length >= 64 and "
            "sequence length <= 512 and sequence length % 64 == 0, "
            "but got sequence length: %d and %d.",
            q_seq_len,
            kv_seq_len));

    T *kv_dev_ptr = const_cast<T *>(kv->data<T>());
    void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q->data<T>()));
    void *k_dev_ptr = reinterpret_cast<void *>(kv_dev_ptr);
    void *v_dev_ptr =
        reinterpret_cast<void *>(kv_dev_ptr + num_heads * head_size);
    void *q_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(q_actual_seqlen->data<int>()));
    void *kv_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(kv_actual_seqlen->data<int>()));
    void *out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(out->data<T>()));
    void *softmax_out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(softmax_out->data<T>()));
    void *bias_dev_ptr = nullptr;

    run_cudnn_fmha_fwd(batch_size,
                       num_heads,
                       q_seq_len,
                       kv_seq_len,
                       head_size,
                       seed,
                       mha_layout,
                       scaling_factor,
                       dropout_probability,
                       bias_type,
                       is_causal_masking,
                       q_dev_ptr,
                       k_dev_ptr,
                       v_dev_ptr,
                       softmax_out_dev_ptr,
                       out_dev_ptr,
                       bias_dev_ptr,
                       q_actual_seqlen_dev_ptr,
                       kv_actual_seqlen_dev_ptr,
                       tensor_dtype,
                       handle);
  }
};

template <typename T>
class FusedDotProductSelfAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_GE(cudnn_version,
                      8901,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8901, "
                          "but got %d.",
                          cudnn_version));
    // inputs
    auto *qkv = ctx.Input<Tensor>("QKV");
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    auto *q_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenQ");
    auto *kv_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenKV");

    // attrs
    float scaling_factor = ctx.Attr<float>("scaling_factor");
    float dropout_probability = ctx.Attr<float>("attn_dropout_rate");
    bool is_causal_masking = ctx.Attr<bool>("is_causal_masking");

    // outputs
    auto *dqkv = ctx.Output<Tensor>(framework::GradVarName("QKV"));

    Tensor
        dSTensor;  // create a tmp dSTensor, we don't need this as op's output
    Tensor *dS = &dSTensor;
    dS->Resize(softmax_out->dims());
    auto *dqkv_ptr = dev_ctx.template Alloc<T>(dqkv, dqkv->numel() * sizeof(T));
    dev_ctx.template Alloc<T>(dS, dS->numel() * sizeof(T));

    // get handles
    auto handle = dev_ctx.cudnn_handle();

    auto tensor_dtype =
        platform::ToCudnnDataType(framework::TransToProtoVarType(qkv->dtype()));
    bool support_type = (tensor_dtype == CUDNN_DATA_HALF ||
                         tensor_dtype == CUDNN_DATA_BFLOAT16);
    PADDLE_ENFORCE_EQ(support_type,
                      true,
                      platform::errors::InvalidArgument(
                          "cuDNN FMHA Only supports FP16/BF16 currently"));
    auto mha_layout = MHA_Layout::QKV_INTERLEAVED;
    std::vector<cudnn_frontend::Operation const *> all_ops;
    std::vector<cudnn_frontend::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // qkv dim: {b, s, 3, h, d};
    auto batch_size = qkv->dims()[0];
    auto seq_len = qkv->dims()[1];
    auto num_heads = qkv->dims()[3];
    auto head_size = qkv->dims()[4];

    T *qkv_dev_ptr = const_cast<T *>(qkv->data<T>());
    T *dqkv_dev_ptr = dqkv->data<T>();
    void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(qkv_dev_ptr));
    void *k_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(qkv_dev_ptr + num_heads * head_size));
    void *v_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(qkv_dev_ptr + 2 * num_heads * head_size));
    void *dq_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dqkv_dev_ptr));
    void *dk_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(dqkv_dev_ptr + num_heads * head_size));
    void *dv_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(dqkv_dev_ptr + 2 * num_heads * head_size));
    void *do_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dO->data<T>()));
    void *ds_dev_ptr = reinterpret_cast<void *>(dS->data<T>());
    void *q_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(q_actual_seqlen->data<int>()));
    void *kv_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(kv_actual_seqlen->data<int>()));
    void *softmax_out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(softmax_out->data<T>()));

    run_cudnn_fmha_bwd(batch_size,
                       num_heads,
                       seq_len,
                       seq_len,
                       head_size,
                       mha_layout,
                       scaling_factor,
                       dropout_probability,
                       is_causal_masking,
                       q_dev_ptr,
                       k_dev_ptr,
                       v_dev_ptr,
                       softmax_out_dev_ptr,
                       dq_dev_ptr,
                       dk_dev_ptr,
                       dv_dev_ptr,
                       do_dev_ptr,
                       ds_dev_ptr,
                       q_actual_seqlen_dev_ptr,
                       kv_actual_seqlen_dev_ptr,
                       tensor_dtype,
                       handle);
  }
};

template <typename T>
class FusedDotProductCrossAttentionGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    PADDLE_ENFORCE_GE(dev_ctx.GetComputeCapability(),
                      80,
                      phi::errors::PreconditionNotMet(
                          "This op only supports Ampere and later devices, "
                          "but got compute capability: %d.",
                          dev_ctx.GetComputeCapability()));
    auto cudnn_version = platform::DnnVersion();
    PADDLE_ENFORCE_GE(cudnn_version,
                      8901,
                      phi::errors::PreconditionNotMet(
                          "This op only supports CUDNN version >= 8901, "
                          "but got %d.",
                          cudnn_version));
    // inputs
    auto *q = ctx.Input<Tensor>("Q");
    auto *kv = ctx.Input<Tensor>("KV");
    auto *dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *softmax_out = ctx.Input<Tensor>("SoftmaxOut");
    auto *q_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenQ");
    auto *kv_actual_seqlen = ctx.Input<Tensor>("ActualSeqlenKV");

    // attrs
    float scaling_factor = ctx.Attr<float>("scaling_factor");
    float dropout_probability = ctx.Attr<float>("attn_dropout_rate");
    bool is_causal_masking = ctx.Attr<bool>("is_causal_masking");

    // outputs
    auto *dq = ctx.Output<Tensor>(framework::GradVarName("Q"));
    auto *dkv = ctx.Output<Tensor>(framework::GradVarName("KV"));

    Tensor
        dSTensor;  // create a tmp dSTensor, we don't need this as op's output
    Tensor *dS = &dSTensor;
    dS->Resize(softmax_out->dims());
    auto *dq_ptr = dev_ctx.template Alloc<T>(dq, dq->numel() * sizeof(T));
    auto *dkv_ptr = dev_ctx.template Alloc<T>(dkv, dkv->numel() * sizeof(T));
    dev_ctx.template Alloc<T>(dS, dS->numel() * sizeof(T));

    // get handles
    auto handle = dev_ctx.cudnn_handle();

    auto tensor_dtype =
        platform::ToCudnnDataType(framework::TransToProtoVarType(q->dtype()));
    bool support_type = (tensor_dtype == CUDNN_DATA_HALF ||
                         tensor_dtype == CUDNN_DATA_BFLOAT16);
    PADDLE_ENFORCE_EQ(support_type,
                      true,
                      platform::errors::InvalidArgument(
                          "cuDNN FMHA Only supports FP16/BF16 currently"));
    auto mha_layout = MHA_Layout::KV_INTERLEAVED;
    std::vector<cudnn_frontend::Operation const *> all_ops;
    std::vector<cudnn_frontend::Operation> ops;
    std::set<std::pair<uint64_t, void *>> data_ptrs;

    // q dim: {b, s, h, d};
    // kv dim: {b, s, 2, h, d};
    auto batch_size = q->dims()[0];
    auto q_seq_len = q->dims()[1];
    auto num_heads = q->dims()[2];
    auto head_size = q->dims()[3];
    auto kv_seq_len = kv->dims()[1];

    T *kv_dev_ptr = const_cast<T *>(kv->data<T>());
    T *dkv_dev_ptr = dkv->data<T>();
    void *q_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(q->data<T>()));
    void *k_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(kv_dev_ptr));
    void *v_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(kv_dev_ptr + num_heads * head_size));
    void *dq_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dq->data<T>()));
    void *dk_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dkv_dev_ptr));
    void *dv_dev_ptr = reinterpret_cast<void *>(
        const_cast<T *>(dkv_dev_ptr + num_heads * head_size));
    void *do_dev_ptr = reinterpret_cast<void *>(const_cast<T *>(dO->data<T>()));
    void *ds_dev_ptr = reinterpret_cast<void *>(dS->data<T>());
    void *q_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(q_actual_seqlen->data<int>()));
    void *kv_actual_seqlen_dev_ptr = reinterpret_cast<void *>(
        const_cast<int *>(kv_actual_seqlen->data<int>()));
    void *softmax_out_dev_ptr =
        reinterpret_cast<void *>(const_cast<T *>(softmax_out->data<T>()));

    run_cudnn_fmha_bwd(batch_size,
                       num_heads,
                       q_seq_len,
                       kv_seq_len,
                       head_size,
                       mha_layout,
                       scaling_factor,
                       dropout_probability,
                       is_causal_masking,
                       q_dev_ptr,
                       k_dev_ptr,
                       v_dev_ptr,
                       softmax_out_dev_ptr,
                       dq_dev_ptr,
                       dk_dev_ptr,
                       dv_dev_ptr,
                       do_dev_ptr,
                       ds_dev_ptr,
                       q_actual_seqlen_dev_ptr,
                       kv_actual_seqlen_dev_ptr,
                       tensor_dtype,
                       handle);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    fused_dot_product_self_attention,
    ops::FusedDotProductSelfAttentionOpKernel<plat::float16>,
    ops::FusedDotProductSelfAttentionOpKernel<plat::bfloat16>)

REGISTER_OP_CUDA_KERNEL(
    fused_dot_product_cross_attention,
    ops::FusedDotProductCrossAttentionOpKernel<plat::float16>,
    ops::FusedDotProductCrossAttentionOpKernel<plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(
    fused_dot_product_self_attention_grad,
    ops::FusedDotProductSelfAttentionGradKernel<plat::float16>,
    ops::FusedDotProductSelfAttentionGradKernel<plat::bfloat16>);
REGISTER_OP_CUDA_KERNEL(
    fused_dot_product_cross_attention_grad,
    ops::FusedDotProductCrossAttentionGradKernel<plat::float16>,
    ops::FusedDotProductCrossAttentionGradKernel<plat::bfloat16>);
