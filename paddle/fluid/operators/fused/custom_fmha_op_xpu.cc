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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class CustomFMHAXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUType = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));
    // input
    const phi::DenseTensor* qkv = ctx.Input<phi::DenseTensor>("QKV");
    const phi::DenseTensor* cu_seq_len =
        ctx.Input<phi::DenseTensor>("CuSeqLen");
    const phi::DenseTensor* host_seq_len =
        ctx.Input<phi::DenseTensor>("HostSeqLen");
    // output
    phi::DenseTensor* s_out = ctx.Output<phi::DenseTensor>("SOut");
    phi::DenseTensor* dropout_mask =
        ctx.Output<phi::DenseTensor>("DropoutMask");
    phi::DenseTensor* dropout_out =
        ctx.Output<phi::DenseTensor>("DropoutOut");
    phi::DenseTensor* ctx_out = ctx.Output<phi::DenseTensor>("CtxOut");

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    const XPUType* qkv_ptr = reinterpret_cast<const XPUType*>(qkv->data<T>());
    // const int* host_seq_len_ptr =
    //     reinterpret_cast<const int*>(host_seq_len->data<int>());
    const int* cu_seq_len_ptr =
        reinterpret_cast<const int*>(cu_seq_len->data<int>());
    // printf("====> host_seq_len_ptr: %p\n", host_seq_len_ptr);
    // printf("====> cu_seq_len_ptr: %p\n", cu_seq_len_ptr);

    int tmp_size = host_seq_len->numel();
    int* tmp = reinterpret_cast<int*>(malloc(tmp_size * sizeof(int)));
    xpu_memcpy(tmp, cu_seq_len_ptr, tmp_size * sizeof(int), XPU_DEVICE_TO_HOST);
    // printf("host_seq_len len: %d\n", tmp_size);
    // for (int i = 0; i < tmp_size; i++) {
    //     printf("seq len: %d\n", tmp[i]);
    // }
    XPUType* s_out_ptr =
        reinterpret_cast<XPUType*>(s_out->mutable_data<T>(ctx.GetPlace()));
    XPUType* dropout_mask_ptr = reinterpret_cast<XPUType*>(
        dropout_mask->mutable_data<T>(ctx.GetPlace()));
    XPUType* dropout_out_ptr = reinterpret_cast<XPUType*>(
        dropout_out->mutable_data<T>(ctx.GetPlace()));
    XPUType* ctx_out_ptr =
        reinterpret_cast<XPUType*>(ctx_out->mutable_data<T>(ctx.GetPlace()));

    // qkv_shape(fp16) = [total_tokens, 3, num_heads, head_size]
    // seq_len_shape(int32) = [batch_size + 1]
    auto& qkv_shape = qkv->dims();
    auto& seq_len_shape = cu_seq_len->dims();
    int num_heads = qkv_shape[2];
    int head_size = qkv_shape[3];
    // batch_size = load size - 1
    int batch_size = seq_len_shape[0] - 1;
    // xpu::VectorParam<int> lod = {host_seq_len_ptr, batch_size + 1,
    // const_cast<int*>(cu_seq_len_ptr)};
    xpu::VectorParam<int> lod = {
        tmp, batch_size + 1, const_cast<int*>(cu_seq_len_ptr)};
    xpu::DropoutParam drop_p;
    drop_p.is_upscale_in_train = true;
    drop_p.is_fixed_seed = false;
    drop_p.dropout_rate = ctx.Attr<float>("dropout_rate");
    drop_p.seed_val = 0;
    bool is_test = ctx.Attr<bool>("is_test");

    int r = xpu::mha_fusion<XPUType, int16_t>(xpu_ctx,
                                              qkv_ptr,
                                              s_out_ptr,
                                              dropout_mask_ptr,
                                              dropout_out_ptr,
                                              ctx_out_ptr,
                                              batch_size,
                                              num_heads,
                                              head_size,
                                              lod,
                                              lod,
                                              drop_p,
                                              is_test);

    free(tmp);

    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_fusion");
  }
};

template <typename T>
class CustomFMHAGradXPUKernel : public framework::OpKernel<T> {
 public:
  using XPUType = typename XPUTypeTrait<T>::Type;

  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(ctx.GetPlace()),
        true,
        platform::errors::PreconditionNotMet("It must use XPUPlace."));
    // input
    const phi::DenseTensor* qkv = ctx.Input<phi::DenseTensor>("QKV");
    const phi::DenseTensor* cu_seq_len =
        ctx.Input<phi::DenseTensor>("CuSeqLen");
    const phi::DenseTensor* host_seq_len =
        ctx.Input<phi::DenseTensor>("HostSeqLen");
    const phi::DenseTensor* s_out = ctx.Input<phi::DenseTensor>("SOut");
    const phi::DenseTensor* dropout_mask =
        ctx.Input<phi::DenseTensor>("DropoutMask");
    const phi::DenseTensor* dropout_out =
        ctx.Input<phi::DenseTensor>("DropoutOut");
    const phi::DenseTensor* d_ctx_out = ctx.Input<phi::DenseTensor>("DCtxOut");
    // output
    phi::DenseTensor* d_qkv = ctx.Output<phi::DenseTensor>("DQKV");
    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);

    const XPUType* qkv_ptr = reinterpret_cast<const XPUType*>(qkv->data<T>());
    // const int* host_seq_len_ptr =
    //     reinterpret_cast<const int*>(host_seq_len->data<int>());
    const int* cu_seq_len_ptr =
        reinterpret_cast<const int*>(cu_seq_len->data<int>());
    const XPUType* s_out_ptr =
        reinterpret_cast<const XPUType*>(s_out->data<T>());
    const XPUType* dropout_mask_ptr =
        reinterpret_cast<const XPUType*>(dropout_mask->data<T>());
    const XPUType* dropout_out_ptr =
        reinterpret_cast<const XPUType*>(dropout_out->data<T>());
    const XPUType* d_ctx_out_ptr =
        reinterpret_cast<const XPUType*>(d_ctx_out->data<T>());

    XPUType* d_qkv_ptr =
        reinterpret_cast<XPUType*>(d_qkv->mutable_data<T>(ctx.GetPlace()));

    // printf("====> grad host_seq_len_ptr: %p\n", host_seq_len_ptr);
    // printf("====> grad cu_seq_len_ptr: %p\n", cu_seq_len_ptr);
    int tmp_size = host_seq_len->numel();
    int* tmp = reinterpret_cast<int*>(malloc(tmp_size * sizeof(int)));
    xpu_memcpy(tmp, cu_seq_len_ptr, tmp_size * sizeof(int), XPU_DEVICE_TO_HOST);
    // XPU_DEVICE_TO_HOST); for (int i = 0; i < tmp_size; i++) {
    //     printf("seq len: %d\n", tmp[i]);
    // }

    // qkv_shape(fp16) = [total_tokens, 3, num_heads, head_size]
    // seq_len_shape(int32) = [batch_size + 1]
    auto& qkv_shape = qkv->dims();
    auto& seq_len_shape = cu_seq_len->dims();
    int total_tokens = qkv_shape[0];
    int num_heads = qkv_shape[2];
    int head_size = qkv_shape[3];
    int batch_size = seq_len_shape[0] - 1;
    // xpu::VectorParam<int> lod = {host_seq_len_ptr, batch_size + 1,
    // const_cast<int*>(cu_seq_len_ptr)};tmp
    xpu::VectorParam<int> lod = {
        tmp, batch_size + 1, const_cast<int*>(cu_seq_len_ptr)};
    xpu::DropoutParam drop_p;
    drop_p.is_upscale_in_train = true;
    drop_p.is_fixed_seed = false;
    drop_p.dropout_rate = ctx.Attr<float>("dropout_rate");
    drop_p.seed_val = 0;
    bool is_test = ctx.Attr<bool>("is_test");

    int qkv_size = total_tokens * num_heads * head_size;
    XPUType* d_q_ptr = RAII_GUARD.alloc<XPUType>(qkv_size);
    PADDLE_ENFORCE_NOT_NULL(
        d_q_ptr, paddle::platform::errors::Fatal("XPU memory is not enough"));
    XPUType* d_k_ptr = RAII_GUARD.alloc<XPUType>(qkv_size);
    PADDLE_ENFORCE_NOT_NULL(
        d_k_ptr, paddle::platform::errors::Fatal("XPU memory is not enough"));
    XPUType* d_v_ptr = RAII_GUARD.alloc<XPUType>(qkv_size);
    PADDLE_ENFORCE_NOT_NULL(
        d_v_ptr, paddle::platform::errors::Fatal("XPU memory is not enough"));
    int r = xpu::mha_fusion_grad<XPUType, int16_t>(
        xpu_ctx,
        qkv_ptr,
        s_out_ptr,
        dropout_mask_ptr,
        dropout_out_ptr,
        const_cast<XPUType*>(d_ctx_out_ptr),
        d_q_ptr,
        d_k_ptr,
        d_v_ptr,
        batch_size,
        num_heads,
        head_size,
        lod,
        lod,
        drop_p,
        is_test);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "mha_fusion");
    free(tmp);
    std::vector<int64_t> dqkv_shape = {total_tokens, num_heads, head_size};
    r = xpu::concat<XPUType>(xpu_ctx,
                             {d_q_ptr, d_k_ptr, d_v_ptr},
                             d_qkv_ptr,
                             {dqkv_shape, dqkv_shape, dqkv_shape},
                             1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(custom_fmha,
                       ops::CustomFMHAXPUKernel<float>,
                       ops::CustomFMHAXPUKernel<plat::float16>);
REGISTER_OP_XPU_KERNEL(custom_fmha_grad,
                       ops::CustomFMHAGradXPUKernel<float>,
                       ops::CustomFMHAGradXPUKernel<plat::float16>);
#endif
