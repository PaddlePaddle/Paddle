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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

#include "paddle/fluid/operators/fused/xpu_fused_common_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedFeedForwardXPUKernel : public framework::OpKernel<T> {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

 public:
  void FFN(const phi::XPUContext& dev_ctx,
           const phi::DenseTensor* x,
           const phi::DenseTensor* linear1_weight,
           const phi::DenseTensor* linear1_bias,
           const phi::DenseTensor* linear2_weight,
           const phi::DenseTensor* linear2_bias,
           const phi::DenseTensor* ln_scale,
           const phi::DenseTensor* ln_bias,
           phi::DenseTensor* out,
           phi::DenseTensor* dropout1_mask,
           phi::DenseTensor* dropout2_mask,
           phi::DenseTensor* ln_mean,
           phi::DenseTensor* ln_variance,
           phi::DenseTensor* linear1_out,
           phi::DenseTensor* ln1_out,
           phi::DenseTensor* dropout1_out,
           phi::DenseTensor* dropout2_out,
           const int bsz_seq,
           const int d_model,
           const int dim_feedforward,
           const std::string& act_method,
           const bool pre_layer_norm,
           const float epsilon1,
           const float epsilon2,
           const XPUDropoutParam& dropout_param1,
           const XPUDropoutParam& dropout_param2,
           int ring_id) const {
    xpu::Context* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);

    int r = xpu::SUCCESS;

    const XPUTypeT* x_ptr = reinterpret_cast<const XPUTypeT*>(x->data<T>());
    const XPUTypeT* residual_ptr = x_ptr;
    const XPUTypeT* linear1_weight_ptr =
        reinterpret_cast<const XPUTypeT*>(linear1_weight->data<T>());
    const XPUTypeT* linear1_bias_ptr =
        reinterpret_cast<const XPUTypeT*>(linear1_bias->data<T>());
    const XPUTypeT* linear2_weight_ptr =
        reinterpret_cast<const XPUTypeT*>(linear2_weight->data<T>());
    const XPUTypeT* linear2_bias_ptr =
        reinterpret_cast<const XPUTypeT*>(linear2_bias->data<T>());

    const float* ln_scale_ptr = ln_scale->data<float>();

    const float* ln_bias_ptr = ln_bias->data<float>();

    // out
    XPUTypeT* out_ptr = reinterpret_cast<XPUTypeT*>(out->data<T>());
    XPUTypeT* linear1_out_ptr =
        reinterpret_cast<XPUTypeT*>(linear1_out->data<T>());
    XPUTypeT* dropout1_mask_ptr =
        reinterpret_cast<XPUTypeT*>(dropout1_mask->data<T>());
    XPUTypeT* dropout2_mask_ptr =
        reinterpret_cast<XPUTypeT*>(dropout2_mask->data<T>());
    float* ln_mean_ptr = ln_mean->data<float>();
    float* ln_variance_ptr = ln_variance->data<float>();

    XPUTypeT* dropout1_out_ptr =
        reinterpret_cast<XPUTypeT*>(dropout1_out->data<T>());
    XPUTypeT* dropout2_out_ptr =
        reinterpret_cast<XPUTypeT*>(dropout2_out->data<T>());

    size_t l3_total_size = xpu_ctx->_l3_mgr.get_size();
    XPUTypeT* linear2_before_tmp_ptr = NULL;  // dim_feedforward * bsz_seq
    XPUTypeT* linear2_after_tmp_ptr = NULL;   // d_model * bsz_seq
    if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T)) {
      XPUTypeT* l3_ptr =
          RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(l3_ptr);
      linear2_before_tmp_ptr = linear2_after_tmp_ptr = l3_ptr;
    } else if ((l3_total_size < dim_feedforward * bsz_seq * sizeof(T)) &&
               (l3_total_size >= d_model * bsz_seq * sizeof(T))) {
      XPUTypeT* l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(l3_ptr);
      linear2_after_tmp_ptr = l3_ptr;
      linear2_before_tmp_ptr =
          RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(linear2_before_tmp_ptr);

    } else {
      XPUTypeT* gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(gm_ptr);
      linear2_before_tmp_ptr = linear2_after_tmp_ptr = gm_ptr;
    }

    // layernorm
    if (pre_layer_norm) {
      XPUTypeT* ln1_out_ptr = reinterpret_cast<XPUTypeT*>(ln1_out->data<T>());
      r = xpu::layer_norm(xpu_ctx,
                          x_ptr,
                          ln1_out_ptr,
                          bsz_seq,
                          d_model,
                          epsilon1,
                          ln_scale_ptr,
                          ln_bias_ptr,
                          ln_mean_ptr,
                          ln_variance_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm ");
      x_ptr = ln1_out_ptr;
    }

    // fc
    phi::XpuFcInfo linear1_fc_info;
    linear1_fc_info.InitFcInfo(0,
                               bsz_seq,
                               dim_feedforward,
                               d_model,
                               false,
                               false,
                               nullptr,
                               nullptr,
                               nullptr);
    phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                     x_ptr,
                                     linear1_weight_ptr,
                                     linear2_before_tmp_ptr,
                                     linear1_fc_info,
                                     1.0f);

    // bias
    r = xpu::broadcast_add(xpu_ctx,
                           linear2_before_tmp_ptr,
                           linear1_bias_ptr,
                           linear1_out_ptr,
                           {bsz_seq, dim_feedforward},
                           {dim_feedforward});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

    // act
    if (act_method == "gelu") {
      r = xpu::gelu(xpu_ctx,
                    linear1_out_ptr,
                    linear2_before_tmp_ptr,
                    linear1_out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu");
    } else if (act_method == "relu") {
      r = xpu::relu(xpu_ctx,
                    linear1_out_ptr,
                    linear2_before_tmp_ptr,
                    linear1_out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently only supports gelu or relu activation functions!"));
    }

    // dropout1
    Dropout<XPUTypeT>(xpu_ctx,
                      linear2_before_tmp_ptr,
                      dropout1_mask_ptr,
                      dropout1_out_ptr,
                      dropout_param1,
                      dropout1_out->numel());

    // fc
    phi::XpuFcInfo linear2_fc_info;
    linear2_fc_info.InitFcInfo(0,
                               bsz_seq,
                               d_model,
                               dim_feedforward,
                               false,
                               false,
                               nullptr,
                               nullptr,
                               nullptr);
    phi::MatMulXPUFunction<XPUTypeT>(xpu_ctx,
                                     dropout1_out_ptr,
                                     linear2_weight_ptr,
                                     dropout2_out_ptr,
                                     linear2_fc_info,
                                     1.0f);

    // bias
    r = xpu::broadcast_add(xpu_ctx,
                           dropout2_out_ptr,
                           linear2_bias_ptr,
                           dropout2_out_ptr,
                           {bsz_seq, d_model},
                           {d_model});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

    // dropout2
    Dropout<XPUTypeT>(xpu_ctx,
                      dropout2_out_ptr,
                      dropout2_mask_ptr,
                      dropout2_out_ptr,
                      dropout_param2,
                      dropout2_out->numel());

    // residual_ptr + dropout_out
    XPUTypeT* residual_add_out_ptr = out_ptr;
    if (pre_layer_norm == false) {
      residual_add_out_ptr = dropout2_out_ptr;
    }
    r = xpu::broadcast_add(xpu_ctx,
                           residual_ptr,
                           dropout2_out_ptr,
                           residual_add_out_ptr,
                           {bsz_seq, d_model},
                           {bsz_seq, d_model});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");

    if (pre_layer_norm == false) {
      r = xpu::layer_norm(xpu_ctx,
                          residual_add_out_ptr,
                          out_ptr,
                          bsz_seq,
                          d_model,
                          epsilon2,
                          ln_scale_ptr,
                          ln_bias_ptr,
                          ln_mean_ptr,
                          ln_variance_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm");
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto place = context.GetPlace();

    auto* x = context.Input<phi::DenseTensor>("X");

    auto* linear1_weight = context.Input<phi::DenseTensor>("Linear1Weight");
    auto* linear1_bias = context.Input<phi::DenseTensor>("Linear1Bias");
    auto* linear2_weight = context.Input<phi::DenseTensor>("Linear2Weight");
    auto* linear2_bias = context.Input<phi::DenseTensor>("Linear2Bias");
    const bool pre_layer_norm = context.Attr<bool>("pre_layer_norm");

    const phi::DenseTensor* ln_scale = nullptr;
    const phi::DenseTensor* ln_bias = nullptr;
    phi::DenseTensor* ln_mean = nullptr;
    phi::DenseTensor* ln_variance = nullptr;
    phi::DenseTensor* ln1_out = nullptr;

    if (pre_layer_norm) {
      ln_scale = context.Input<phi::DenseTensor>("Ln1Scale");
      ln_bias = context.Input<phi::DenseTensor>("Ln1Bias");
      ln_mean = context.Output<phi::DenseTensor>("Ln1Mean");
      ln_variance = context.Output<phi::DenseTensor>("Ln1Variance");
      ln1_out = context.Output<phi::DenseTensor>("Ln1Out");
      ln1_out->mutable_data<T>(place);
    } else {
      ln_scale = context.Input<phi::DenseTensor>("Ln2Scale");
      ln_bias = context.Input<phi::DenseTensor>("Ln2Bias");
      ln_mean = context.Output<phi::DenseTensor>("Ln2Mean");
      ln_variance = context.Output<phi::DenseTensor>("Ln2Variance");
    }

    auto* out = context.Output<phi::DenseTensor>("Out");
    auto* dropout1_mask = context.Output<phi::DenseTensor>("Dropout1Mask");
    auto* dropout2_mask = context.Output<phi::DenseTensor>("Dropout2Mask");
    auto* linear1_out = context.Output<phi::DenseTensor>("Linear1Out");

    auto* dropout1_out = context.Output<phi::DenseTensor>("Dropout1Out");
    auto* dropout2_out = context.Output<phi::DenseTensor>("Dropout2Out");

    const std::string act_method = context.Attr<std::string>("act_method");

    const int ring_id = context.Attr<int>("ring_id");
    const float epsilon1 = context.Attr<float>("ln1_epsilon");
    const float epsilon2 = context.Attr<float>("ln2_epsilon");
    XPUDropoutParam dropout_param1;
    dropout_param1.initXPUDropoutParam(context, 1);
    XPUDropoutParam dropout_param2;
    dropout_param2.initXPUDropoutParam(context, 2);

    ln_mean->mutable_data<float>(place);
    ln_variance->mutable_data<float>(place);
    out->mutable_data<T>(place);
    dropout1_mask->mutable_data<T>(place);
    dropout2_mask->mutable_data<T>(place);
    dropout1_out->mutable_data<T>(place);
    dropout2_out->mutable_data<T>(place);
    linear1_out->mutable_data<T>(place);

    auto x_dim = x->dims();
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(x_dim), 0, false);

    auto dim = linear1_weight->dims();
    int d_model = dim[0];
    int dim_feedforward = dim[dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;

    auto& dev_ctx = context.template device_context<phi::XPUContext>();
    FFN(dev_ctx,
        x,
        linear1_weight,
        linear1_bias,
        linear2_weight,
        linear2_bias,
        ln_scale,
        ln_bias,
        out,
        dropout1_mask,
        dropout2_mask,
        ln_mean,
        ln_variance,
        linear1_out,
        ln1_out,
        dropout1_out,
        dropout2_out,
        bsz_seq,
        d_model,
        dim_feedforward,
        act_method,
        pre_layer_norm,
        epsilon1,
        epsilon2,
        dropout_param1,
        dropout_param2,
        ring_id);
  }
};

template <typename DeviceContext, typename T>
class FusedFeedForwardGradXPUKernel : public framework::OpKernel<T> {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

 public:
  void FFNGrad(const phi::XPUContext& dev_ctx,
               const phi::DenseTensor* d_out,
               const phi::DenseTensor* x,
               const phi::DenseTensor* dropout1_mask,
               const phi::DenseTensor* dropout2_mask,
               const phi::DenseTensor* linear1_out,
               const phi::DenseTensor* ln1_out,
               const phi::DenseTensor* dropout1_out,
               const phi::DenseTensor* dropout2_out,
               const phi::DenseTensor* linear1_weight,
               const phi::DenseTensor* linear2_weight,
               const phi::DenseTensor* ln_scale,
               const phi::DenseTensor* ln_mean,
               const phi::DenseTensor* ln_variance,
               phi::DenseTensor* d_x,
               phi::DenseTensor* d_linear1_weight,
               phi::DenseTensor* d_linear1_bias,
               phi::DenseTensor* d_linear2_weight,
               phi::DenseTensor* d_linear2_bias,
               phi::DenseTensor* d_ln_scale,
               phi::DenseTensor* d_ln_bias,
               const int bsz_seq,
               const int d_model,
               const int dim_feedforward,
               const XPUDropoutParam& dropout_param1,
               const XPUDropoutParam& dropout_param2,
               const std::string& act_method,
               const bool pre_layer_norm,
               const float epsilon,
               const int ring_id) const {
    xpu::Context* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    int r = xpu::SUCCESS;

    // inputs ptr
    const XPUTypeT* d_out_ptr =
        reinterpret_cast<const XPUTypeT*>(d_out->data<T>());
    const XPUTypeT* x_ptr = reinterpret_cast<const XPUTypeT*>(x->data<T>());
    const XPUTypeT* dropout1_mask_ptr =
        reinterpret_cast<const XPUTypeT*>(dropout1_mask->data<T>());
    const XPUTypeT* dropout2_mask_ptr =
        reinterpret_cast<const XPUTypeT*>(dropout2_mask->data<T>());
    const XPUTypeT* linear1_out_ptr =
        reinterpret_cast<const XPUTypeT*>(linear1_out->data<T>());
    const XPUTypeT* dropout1_out_ptr =
        reinterpret_cast<const XPUTypeT*>(dropout1_out->data<T>());
    const XPUTypeT* linear1_weight_ptr =
        reinterpret_cast<const XPUTypeT*>(linear1_weight->data<T>());
    const XPUTypeT* linear2_weight_ptr =
        reinterpret_cast<const XPUTypeT*>(linear2_weight->data<T>());
    const float* ln_scale_ptr = ln_scale->data<float>();

    const float* ln_mean_ptr = ln_mean->data<float>();
    const float* ln_variance_ptr = ln_variance->data<float>();
    // outputs ptr
    XPUTypeT* d_x_ptr = reinterpret_cast<XPUTypeT*>(d_x->data<T>());
    XPUTypeT* d_linear1_weight_ptr =
        reinterpret_cast<XPUTypeT*>(d_linear1_weight->data<T>());
    XPUTypeT* d_linear1_bias_ptr =
        reinterpret_cast<XPUTypeT*>(d_linear1_bias->data<T>());
    XPUTypeT* d_linear2_weight_ptr =
        reinterpret_cast<XPUTypeT*>(d_linear2_weight->data<T>());
    XPUTypeT* d_linear2_bias_ptr =
        reinterpret_cast<XPUTypeT*>(d_linear2_bias->data<T>());
    float* d_ln_scale_ptr = d_ln_scale->data<float>();
    float* d_ln_bias_ptr = d_ln_bias->data<float>();

    size_t l3_total_size = xpu_ctx->_l3_mgr.get_size();

    XPUTypeT* big_tmp_l3_ptr = NULL;    // dim_feedforward * bsz_seq
    XPUTypeT* small_tmp_l3_ptr = NULL;  // d_model * bsz_seq
    XPUTypeT* big_tmp_gm_ptr = NULL;    // dim_feedforward * bsz_seq
    XPUTypeT* small_tmp_gm_ptr = NULL;  // d_model * bsz_seq

    XPUTypeT* d_layernorm_out_ptr = NULL;  // dx9
    XPUTypeT* d_dropout2_out_ptr = NULL;   // dx7

    XPUTypeT* d_linear2_out_ptr = NULL;   // dx5
    XPUTypeT* d_dropout1_out_ptr = NULL;  // dx4
    XPUTypeT* d_act_out_ptr = NULL;       // dx3

    XPUTypeT* d_linear1_out_ptr = NULL;  // dx1

    const XPUTypeT* d_residual_ptr = d_out_ptr;

    if (l3_total_size >= (dim_feedforward * bsz_seq * sizeof(T) +
                          d_model * bsz_seq * sizeof(T))) {
      big_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_l3_ptr);
      small_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_l3_ptr);
      d_layernorm_out_ptr = small_tmp_l3_ptr;
      d_dropout2_out_ptr = small_tmp_l3_ptr;
      d_linear2_out_ptr = big_tmp_l3_ptr;
      d_dropout1_out_ptr = big_tmp_l3_ptr;
      d_act_out_ptr = big_tmp_l3_ptr;
      d_linear1_out_ptr = small_tmp_l3_ptr;
    } else if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T)) {
      big_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_l3_ptr);
      small_tmp_l3_ptr = big_tmp_l3_ptr;
      big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
      small_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(d_model * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_gm_ptr);

      d_layernorm_out_ptr = small_tmp_l3_ptr;
      d_dropout2_out_ptr = small_tmp_gm_ptr;
      d_linear2_out_ptr = big_tmp_l3_ptr;
      d_dropout1_out_ptr = big_tmp_l3_ptr;
      d_act_out_ptr = big_tmp_gm_ptr;
      d_linear1_out_ptr = small_tmp_l3_ptr;

    } else if (l3_total_size >= d_model * bsz_seq * sizeof(T)) {
      big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
      small_tmp_l3_ptr = RAII_GUARD.alloc_l3<XPUTypeT>(d_model * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_l3_ptr);

      d_layernorm_out_ptr = small_tmp_l3_ptr;
      d_dropout2_out_ptr = small_tmp_l3_ptr;
      d_linear2_out_ptr = big_tmp_gm_ptr;
      d_dropout1_out_ptr = big_tmp_gm_ptr;
      d_act_out_ptr = big_tmp_gm_ptr;
      d_linear1_out_ptr = small_tmp_l3_ptr;
    } else {
      big_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(dim_feedforward * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(big_tmp_gm_ptr);
      small_tmp_gm_ptr = RAII_GUARD.alloc<XPUTypeT>(d_model * bsz_seq);
      PADDLE_ENFORCE_XDNN_NOT_NULL(small_tmp_gm_ptr);
      d_layernorm_out_ptr = small_tmp_gm_ptr;
      d_dropout2_out_ptr = small_tmp_gm_ptr;
      d_linear2_out_ptr = big_tmp_gm_ptr;
      d_dropout1_out_ptr = big_tmp_gm_ptr;
      d_act_out_ptr = big_tmp_gm_ptr;
      d_linear1_out_ptr = small_tmp_gm_ptr;
    }

    if (pre_layer_norm == false) {
      const XPUTypeT* dropout2_out_ptr =
          reinterpret_cast<const XPUTypeT*>(dropout2_out->data<T>());
      r = xpu::layer_norm_grad(xpu_ctx,
                               dropout2_out_ptr,
                               d_out_ptr,
                               d_layernorm_out_ptr,
                               bsz_seq,
                               d_model,
                               epsilon,
                               ln_scale_ptr,
                               ln_mean_ptr,
                               ln_variance_ptr,
                               d_ln_scale_ptr,
                               d_ln_bias_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
      d_residual_ptr = d_layernorm_out_ptr;
    }
    DropoutGrad(xpu_ctx,
                d_residual_ptr,
                dropout2_mask_ptr,
                d_dropout2_out_ptr,
                dropout_param2,
                bsz_seq * d_model);
    // linear_grad2
    r = xpu::reduce_sum(xpu_ctx,
                        d_dropout2_out_ptr,
                        d_linear2_bias_ptr,
                        {bsz_seq, d_model},
                        {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

    phi::XpuFcInfo linear2_fc_info;
    linear2_fc_info.InitFcInfo(0,
                               bsz_seq,
                               d_model,
                               dim_feedforward,
                               false,
                               false,
                               nullptr,
                               nullptr,
                               nullptr);

    const XPUTypeT* a_1 = reinterpret_cast<const XPUTypeT*>(NULL);
    const XPUTypeT* b_1 = reinterpret_cast<const XPUTypeT*>(NULL);
    const XPUTypeT* a_2 = reinterpret_cast<const XPUTypeT*>(NULL);
    const XPUTypeT* b_2 = reinterpret_cast<const XPUTypeT*>(NULL);
    XPUTypeT* c_1 = d_linear2_out_ptr;
    XPUTypeT* c_2 = d_linear2_weight_ptr;
    phi::XpuFcInfo info_d_dropout1;
    phi::XpuFcInfo info_dw2;

    std::tuple<phi::XpuFcInfo,
               phi::XpuFcInfo,
               const XPUTypeT*,
               const XPUTypeT*,
               const XPUTypeT*,
               const XPUTypeT*>
        fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                        &RAII_GUARD,
                                        linear2_fc_info,
                                        false,
                                        false,
                                        dropout1_out_ptr,
                                        linear2_weight_ptr,
                                        d_dropout2_out_ptr);

    std::tie(info_d_dropout1, info_dw2, a_1, b_1, a_2, b_2) = fc_info;

    // if l3_total_size >= dim_feedforward * bsz_seq * sizeof(T), first transpos
    if (l3_total_size >= dim_feedforward * bsz_seq * sizeof(T) &&
        info_dw2.trans_x) {
      r = xpu::transpose<XPUTypeT>(xpu_ctx,
                                   dropout1_out_ptr,
                                   big_tmp_l3_ptr,
                                   {bsz_seq, dim_feedforward},
                                   {1, 0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      a_2 = big_tmp_l3_ptr;
      info_dw2.trans_x = !info_dw2.trans_x;
      info_dw2.stride_x = info_dw2.k;
    }

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_d_dropout1, 1.0f, true);

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_dw2, 1.0f, true);

    // dropout_grad1
    DropoutGrad(xpu_ctx,
                d_linear2_out_ptr,
                dropout1_mask_ptr,
                d_dropout1_out_ptr,
                dropout_param1,
                bsz_seq * dim_feedforward);

    // act_grad
    if (act_method == "gelu") {
      r = xpu::gelu_grad(xpu_ctx,
                         linear1_out_ptr,
                         linear1_out_ptr,
                         d_dropout1_out_ptr,
                         d_act_out_ptr,
                         bsz_seq * dim_feedforward);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu_grad");
    } else if (act_method == "relu") {
      r = xpu::relu_grad(xpu_ctx,
                         linear1_out_ptr,
                         linear1_out_ptr,
                         d_dropout1_out_ptr,
                         d_act_out_ptr,
                         bsz_seq * dim_feedforward);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently only supports gelu or relu activation functions!"));
    }

    // linear1_grad
    r = xpu::reduce_sum(xpu_ctx,
                        d_act_out_ptr,
                        d_linear1_bias_ptr,
                        {bsz_seq, dim_feedforward},
                        {0});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");

    phi::XpuFcInfo linear1_fc_info;
    linear1_fc_info.InitFcInfo(0,
                               bsz_seq,
                               dim_feedforward,
                               d_model,
                               false,
                               false,
                               nullptr,
                               nullptr,
                               nullptr);

    a_1 = reinterpret_cast<const XPUTypeT*>(NULL);
    b_1 = reinterpret_cast<const XPUTypeT*>(NULL);
    a_2 = reinterpret_cast<const XPUTypeT*>(NULL);
    b_2 = reinterpret_cast<const XPUTypeT*>(NULL);

    c_1 = (pre_layer_norm == true ? d_linear1_out_ptr : d_x_ptr);
    c_2 = d_linear1_weight_ptr;
    phi::XpuFcInfo info_dx;
    phi::XpuFcInfo info_dw1;

    const XPUTypeT* linear1_x_ptr =
        (pre_layer_norm == true
             ? reinterpret_cast<const XPUTypeT*>(ln1_out->data<T>())
             : x_ptr);

    if (l3_total_size >= d_model * bsz_seq * sizeof(T) && info_dw1.trans_x) {
      r = xpu::transpose<XPUTypeT>(
          xpu_ctx, linear1_x_ptr, small_tmp_l3_ptr, {bsz_seq, d_model}, {1, 0});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
      a_2 = small_tmp_l3_ptr;
      info_dw1.trans_x = !info_dw1.trans_x;
      info_dw1.stride_x = info_dw1.k;
    }

    fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                    &RAII_GUARD,
                                    linear1_fc_info,
                                    false,
                                    false,
                                    linear1_x_ptr,
                                    linear1_weight_ptr,
                                    d_act_out_ptr);

    std::tie(info_dx, info_dw1, a_1, b_1, a_2, b_2) = fc_info;

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f, true);

    phi::MatMulXPUFunction<XPUTypeT>(
        xpu_ctx, a_2, b_2, c_2, info_dw1, 1.0f, true);

    if (pre_layer_norm) {
      r = xpu::layer_norm_grad(xpu_ctx,
                               x_ptr,
                               c_1,
                               c_1,
                               bsz_seq,
                               d_model,
                               epsilon,
                               ln_scale_ptr,
                               ln_mean_ptr,
                               ln_variance_ptr,
                               d_ln_scale_ptr,
                               d_ln_bias_ptr);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "layer_norm_grad");
    }

    r = xpu::add(xpu_ctx, c_1, d_residual_ptr, d_x_ptr, d_model * bsz_seq);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto place = context.GetPlace();
    const bool pre_layer_norm = context.Attr<bool>("pre_layer_norm");
    // inputs
    auto* d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x = context.Input<phi::DenseTensor>("X");

    auto* dropout1_mask = context.Input<phi::DenseTensor>("Dropout1Mask");
    auto* dropout2_mask = context.Input<phi::DenseTensor>("Dropout2Mask");
    auto* linear1_out = context.Input<phi::DenseTensor>("Linear1Out");
    auto* ln1_out =
        pre_layer_norm ? context.Input<phi::DenseTensor>("Ln1Out") : nullptr;

    auto* dropout1_out = context.Input<phi::DenseTensor>("Dropout1Out");
    auto* dropout2_out = context.Input<phi::DenseTensor>("Dropout2Out");
    auto* linear1_weight = context.Input<phi::DenseTensor>("Linear1Weight");
    auto* linear2_weight = context.Input<phi::DenseTensor>("Linear2Weight");

    const phi::DenseTensor* ln_mean = nullptr;
    const phi::DenseTensor* ln_variance = nullptr;
    const phi::DenseTensor* ln_scale = nullptr;

    if (pre_layer_norm) {
      ln_mean = context.Input<phi::DenseTensor>("Ln1Mean");
      ln_variance = context.Input<phi::DenseTensor>("Ln1Variance");
      ln_scale = context.Input<phi::DenseTensor>("Ln1Scale");
    } else {
      ln_mean = context.Input<phi::DenseTensor>("Ln2Mean");
      ln_variance = context.Input<phi::DenseTensor>("Ln2Variance");
      ln_scale = context.Input<phi::DenseTensor>("Ln2Scale");
    }

    // output
    auto* d_x = context.Output<phi::DenseTensor>(framework::GradVarName("X"));

    phi::DenseTensor* d_ln_scale = nullptr;
    phi::DenseTensor* d_ln_bias = nullptr;

    if (pre_layer_norm) {
      d_ln_scale =
          context.Output<phi::DenseTensor>(framework::GradVarName("Ln1Scale"));
      d_ln_bias =
          context.Output<phi::DenseTensor>(framework::GradVarName("Ln1Bias"));
    } else {
      d_ln_scale =
          context.Output<phi::DenseTensor>(framework::GradVarName("Ln2Scale"));
      d_ln_bias =
          context.Output<phi::DenseTensor>(framework::GradVarName("Ln2Bias"));
    }

    auto* d_linear1_weight = context.Output<phi::DenseTensor>(
        framework::GradVarName("Linear1Weight"));
    auto* d_linear1_bias =
        context.Output<phi::DenseTensor>(framework::GradVarName("Linear1Bias"));
    auto* d_linear2_weight = context.Output<phi::DenseTensor>(
        framework::GradVarName("Linear2Weight"));
    auto* d_linear2_bias =
        context.Output<phi::DenseTensor>(framework::GradVarName("Linear2Bias"));

    float epsilon = 0.0f;
    if (pre_layer_norm) {
      epsilon = context.Attr<float>("ln1_epsilon");
    } else {
      epsilon = context.Attr<float>("ln2_epsilon");
    }

    const std::string act_method = context.Attr<std::string>("act_method");

    XPUDropoutParam dropout_param1(context, 1);
    XPUDropoutParam dropout_param2(context, 2);

    const int ring_id = context.Attr<int>("ring_id");

    d_x->mutable_data<T>(place);
    d_ln_scale->mutable_data<float>(place);
    d_ln_bias->mutable_data<float>(place);
    d_linear1_bias->mutable_data<T>(place);
    d_linear2_bias->mutable_data<T>(place);
    d_linear1_weight->mutable_data<T>(place);
    d_linear2_weight->mutable_data<T>(place);

    auto x_dim = x->dims();
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(x_dim), 0, false);

    auto linear1_weight_dim = linear1_weight->dims();
    int d_model = linear1_weight_dim[0];
    int dim_feedforward = linear1_weight_dim[linear1_weight_dim.size() - 1];
    int bsz_seq = mat_dim_x.batch_size_ * mat_dim_x.height_;
    auto& dev_ctx = context.template device_context<phi::XPUContext>();

    FFNGrad(dev_ctx,
            d_out,
            x,
            dropout1_mask,
            dropout2_mask,
            linear1_out,
            ln1_out,
            dropout1_out,
            dropout2_out,
            linear1_weight,
            linear2_weight,
            ln_scale,
            ln_mean,
            ln_variance,
            d_x,
            d_linear1_weight,
            d_linear1_bias,
            d_linear2_weight,
            d_linear2_bias,
            d_ln_scale,
            d_ln_bias,
            bsz_seq,
            d_model,
            dim_feedforward,
            dropout_param1,
            dropout_param2,
            act_method,
            pre_layer_norm,
            epsilon,
            ring_id);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    fused_feedforward,
    ops::FusedFeedForwardXPUKernel<phi::XPUContext, float>,
    ops::FusedFeedForwardXPUKernel<phi::XPUContext, paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    fused_feedforward_grad,
    ops::FusedFeedForwardGradXPUKernel<phi::XPUContext, float>,
    ops::FusedFeedForwardGradXPUKernel<phi::XPUContext,
                                       paddle::platform::float16>);

#endif
