/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class CustomFusedDenseXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  // using biasType= typename XPUTypeTrait<float>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor* bias = ctx.Input<phi::DenseTensor>("Bias");

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    phi::DenseTensor* reserve_space =
        ctx.Output<phi::DenseTensor>("ReserveSpace");

    bool transx = ctx.Attr<bool>("transx");
    bool transy = ctx.Attr<bool>("transy");

    std::string activation = ctx.Attr<std::string>("activation");
    VLOG(5) << "transx = " << transx << " , transy = " << transy
            << " , activation = " << activation;

    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), transx ? 1 : x->dims().size() - 1);

    // (M * K) * (K * N) for new api use
    // int64_t M = transx ? x_mat_dims[1] : x_mat_dims[0];
    // int64_t K = transy ? y->dims()[1] : y->dims()[0];
    // int64_t N = transy ? y->dims()[0] : y->dims()[1];

    // 调用新接口，这里先分开调用，等待qingpen的新接口
    int r = 0;
    xpu::Activation_t act = xpu::Activation_t::LINEAR;
    if (activation == "relu") {
      act = xpu::Activation_t::RELU;
    } else if (activation == "gelu") {
      act = xpu::Activation_t::GELU;
    }
    // fc + bias + act
    // 1. fc
    phi::XpuFcInfo fc_info;

    phi::GetFCInfo(x_mat_dims, y->dims(), transx, transy, &fc_info);
    VLOG(1) << "CustomFusedDenseXPUKernel 000";
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());
    XPUType* out_ptr =
        reinterpret_cast<XPUType*>(out->mutable_data<T>(ctx.GetPlace()));
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    // XPUType* fc_out_ptr = RAII_GUARD.alloc_l3_or_gm<XPUType>(out->numel());
    // phi::MatMulXPUFunction<XPUType>(
    // xpu_ctx, x_ptr, y_ptr, fc_out_ptr, fc_info, 1.0f);
    XPUType* bias_out_ptr = out_ptr;
    if (activation != "none" && reserve_space) {
      bias_out_ptr = reinterpret_cast<XPUType*>(
          reserve_space->mutable_data<T>(ctx.GetPlace()));
    }
    int m = fc_info.m;
    int n = fc_info.n;
    int k = fc_info.k;
    int ldx = fc_info.stride_x;
    int ldy = fc_info.stride_y;
    int ldout = fc_info.stride_out;
    bool trans_x = fc_info.trans_x;
    bool trans_y = fc_info.trans_y;
    float* max_x = fc_info.max_x;
    float* max_y = fc_info.max_y;
    float* max_out = fc_info.max_out;

    const XPUType* bias_ptr = reinterpret_cast<const XPUType*>(bias->data<T>());
    int nsdnn = xpu_ctx->nsdnn();
    XPUType* x_ptr_l3 = RAII_GUARD.alloc_l3_or_gm<XPUType>(m * k);
    float* max_x_ptr = RAII_GUARD.alloc<float>(nsdnn);
    // move x_ptr to l3
    int x_input_size = m * k;
    int bias_size = bias->numel();
    uint64_t l3_size = 63 * 1024 * 1024;
    bool copy_to_l3 = (l3_size >= x_input_size * sizeof(XPUType));
    bool copy_to_l3_bias = (l3_size >= (bias_size * sizeof(float) +
                                        x_input_size * sizeof(XPUType)));
    if ((bias_ptr == nullptr) || std::is_same<T, float>::value) {
      if (copy_to_l3 && (k == ldx)) {
        // r = xpu::findmax2d_copy_fusion<XPUType>(xpu_ctx, x_ptr, max_x_ptr,
        // x_ptr_l3, m, k, k, k); PADDLE_ENFORCE_XDNN_SUCCESS(r,
        // "findmax2d_copy_fusion");
        r = xpu::findmax_copy_fusion<XPUType>(
            xpu_ctx, x_ptr, max_x_ptr, x_ptr_l3, (m * k));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");

        r = xpu::fc_fusion<XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx,
            x_ptr_l3,
            y_ptr,
            bias_out_ptr,
            m,
            n,
            k,
            trans_x,
            trans_y,
            max_x_ptr,
            max_y,
            max_out,
            ldx,
            ldy,
            ldout,
            1.0f,
            0.0f,
            reinterpret_cast<const float*>(bias_ptr),
            xpu::Activation_t::LINEAR);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
      } else {
        r = xpu::fc_fusion<XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx,
            x_ptr,
            y_ptr,
            bias_out_ptr,
            m,
            n,
            k,
            trans_x,
            trans_y,
            max_x,
            max_y,
            max_out,
            ldx,
            ldy,
            ldout,
            1.0f,
            0.0f,
            reinterpret_cast<const float*>(bias_ptr),
            xpu::Activation_t::LINEAR);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
      }
    } else {
      float* bias_tmp_ptr = RAII_GUARD.alloc_l3_or_gm<float>(bias_size);
      r = xpu::cast<XPUType, float>(
          dev_ctx.x_context(), bias_ptr, bias_tmp_ptr, bias_size);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      if (copy_to_l3_bias && (k == ldx)) {
        // r = xpu::findmax2d_copy_fusion<XPUType>(xpu_ctx, x_ptr, max_x_ptr,
        // x_ptr_l3, m, k, k, k); PADDLE_ENFORCE_XDNN_SUCCESS(r,
        // "findmax2d_copy_fusion");
        r = xpu::findmax_copy_fusion<XPUType>(
            xpu_ctx, x_ptr, max_x_ptr, x_ptr_l3, (m * k));
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax_copy_fusion");

        r = xpu::fc_fusion<XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx,
            x_ptr_l3,
            y_ptr,
            bias_out_ptr,
            m,
            n,
            k,
            trans_x,
            trans_y,
            max_x_ptr,
            max_y,
            max_out,
            ldx,
            ldy,
            ldout,
            1.0f,
            0.0f,
            bias_tmp_ptr,
            xpu::Activation_t::LINEAR);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
      } else {
        r = xpu::fc_fusion<XPUType, XPUType, XPUType, int16_t>(
            xpu_ctx,
            x_ptr,
            y_ptr,
            bias_out_ptr,
            m,
            n,
            k,
            trans_x,
            trans_y,
            max_x,
            max_y,
            max_out,
            ldx,
            ldy,
            ldout,
            1.0f,
            0.0f,
            bias_tmp_ptr,
            xpu::Activation_t::LINEAR);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
      }
    }

    // r = xpu::broadcast_add(xpu_ctx,
    //                        fc_out_ptr,
    //                        bias_ptr,
    //                        bias_out_ptr,
    //                        {fc_info.m, fc_info.n},
    //                        {fc_info.n});
    // PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_add");
    // 3 act
    if (activation == "relu") {
      r = xpu::relu(xpu_ctx, bias_out_ptr, out_ptr, out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu");
    } else if (activation == "gelu") {
      r = xpu::gelu(xpu_ctx, bias_out_ptr, out_ptr, out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu");
    }
  }
};

template <typename DeviceContext, typename T>
class CustomFusedDenseXPUGradKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;
  // using dbiasType= typename XPUTypeTrait<float>::Type;
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int r = -1;
    bool transx = ctx.Attr<bool>("transx");
    bool transy = ctx.Attr<bool>("transy");
    // bool use_addto = ctx.Attr<bool>("use_addto");
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    const phi::DenseTensor* dout = ctx.Input<phi::DenseTensor>("DOut");
    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* y = ctx.Input<phi::DenseTensor>("Y");

    const phi::DenseTensor* reserve_space =
        ctx.Input<phi::DenseTensor>("ReserveSpace");

    phi::DenseTensor* dx = ctx.Output<phi::DenseTensor>("DX");
    phi::DenseTensor* dy = ctx.Output<phi::DenseTensor>("DY");
    phi::DenseTensor* dbias = ctx.Output<phi::DenseTensor>("DBias");

    auto tag_start = std::chrono::steady_clock::now();

    std::string activation = "none";
    if (ctx.HasAttr("activation")) {
      activation = ctx.Attr<std::string>("activation");
    } else if (ctx.HasAttr("activation_grad")) {
      activation = ctx.Attr<std::string>("activation_grad");
    }

    auto* xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout->data<T>());

    const XPUType* dout_fc_ptr = dout_ptr;
    const XPUType* x_ptr = reinterpret_cast<const XPUType*>(x->data<T>());
    const XPUType* y_ptr = reinterpret_cast<const XPUType*>(y->data<T>());

    // const XPUType*
    const XPUType* reserve_space_ptr =
        (reserve_space == NULL)
            ? (reinterpret_cast<const XPUType*>(NULL))
            : (reinterpret_cast<const XPUType*>(reserve_space->data<T>()));
    XPUType* d_act_input_ptr;
    if (activation != "none") {
      d_act_input_ptr = RAII_GUARD.alloc_l3_or_gm<XPUType>(dout->numel());
      dout_fc_ptr = d_act_input_ptr;
    }

    // 1. act_grad  2. fc_grad 3. dbias
    r = 0;
    if (activation == "relu") {
      r = xpu::relu_grad(xpu_ctx,
                         reserve_space_ptr,
                         reserve_space_ptr,
                         dout_ptr,
                         d_act_input_ptr,
                         dout->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "relu_grad");
    } else if (activation == "gelu") {
      r = xpu::gelu_grad(xpu_ctx,
                         reserve_space_ptr,
                         reserve_space_ptr,
                         dout_ptr,
                         d_act_input_ptr,
                         dout->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "gelu_grad");
    }

    auto x_mat_dims =
        phi::flatten_to_2d(x->dims(), transx ? 1 : x->dims().size() - 1);
    phi::XpuFcInfo info_forward;
    phi::GetFCInfo(x_mat_dims, y->dims(), transx, transy, &info_forward);

    // 2. fc_grad
    const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
    const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
    XPUType* c_1 =
        (dx == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dx->mutable_data<T>(ctx.GetPlace()));
    XPUType* c_2 =
        (dy == NULL)
            ? reinterpret_cast<XPUType*>(NULL)
            : reinterpret_cast<XPUType*>(dy->mutable_data<T>(ctx.GetPlace()));
    phi::XpuFcInfo info_dx;
    phi::XpuFcInfo info_dy;
    std::tuple<phi::XpuFcInfo,
               phi::XpuFcInfo,
               const XPUType*,
               const XPUType*,
               const XPUType*,
               const XPUType*>
        fc_info = phi::MatmulGradFcInfo(xpu_ctx,
                                        &RAII_GUARD,
                                        info_forward,
                                        transx,
                                        transy,
                                        x_ptr,
                                        y_ptr,
                                        dout_fc_ptr);
    std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;

    // const float beta_zero = 0.0f;
    // const float beta_one = 1.0f;
    // const float beta_dinput = (use_addto ? beta_one : beta_zero);

    if (dx) {
      phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_1, b_1, c_1, info_dx, 1.0f);
      // int dx_m = info_dx.m;
      // int dx_n = info_dx.n;
      // int dx_k = info_dx.k;
      // int dx_ldx = info_dx.stride_x;
      // int dx_ldy = info_dx.stride_y;
      // int dx_ldout = info_dx.stride_out;
      // bool dx_trans_x = info_dx.trans_x;
      // bool dx_trans_y = info_dx.trans_y;
      // float* dx_max_x = info_dx.max_x;
      // float* dx_max_y = info_dx.max_y;
      // float* dx_max_out = info_dx.max_out;

      // r = xpu::fc_fusion<XPUType, XPUType, XPUType, int16_t>(xpu_ctx, a_1,
      // b_1, c_1, dx_m, dx_n, dx_k, dx_trans_x,
      //     dx_trans_y, dx_max_x, dx_max_y, dx_max_out, dx_ldx, dx_ldy,
      //     dx_ldout, 1.0f, 0.0f, nullptr, xpu::Activation_t::LINEAR);
      // PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
    }
    if (dy) {
      phi::MatMulXPUFunction<XPUType>(xpu_ctx, a_2, b_2, c_2, info_dy, 1.0f);
      // 3. dbias
      if (dbias) {
        XPUType* dbias_ptr =
            reinterpret_cast<XPUType*>(dbias->mutable_data<T>(ctx.GetPlace()));
        r = xpu::reduce_sum(xpu_ctx,
                            dout_fc_ptr,
                            dbias_ptr,
                            {info_forward.m, info_forward.n},
                            {0});
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "reduce_sum");
      }

      // int m = info_forward.k;
      // int n = info_forward.n;
      // int k = info_forward.m;
      // int ldx = info_forward.stride_x;
      // int ldy = info_forward.n;
      // int ldout = info_forward.n;
      // bool trans_x = !info_forward.trans_x;
      // bool trans_y = info_forward.trans_y;
      // float* max_x = nullptr;
      // float* max_y = nullptr;
      // float* max_out = nullptr;
      // XPUType* dbias_ptr =
      // reinterpret_cast<XPUType*>(dbias->mutable_data<T>(ctx.GetPlace())); if
      // ((!dbias) || std::is_same<T, float>::value) {
      //   r = xpu::fc_dbias_fusion<XPUType, XPUType, XPUType, float,
      //   int16_t>(xpu_ctx, reinterpret_cast<const XPUType*>(b_2),
      //   reinterpret_cast<const XPUType*>(a_2),
      //   reinterpret_cast<XPUType*>(c_2), m, n, k, trans_x,
      //       trans_y, max_x, max_y, max_out, ldx, ldy, ldout, 1.0f, 0.0f,
      //       nullptr, reinterpret_cast<float*>(dbias_ptr));
      //   PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_dbias_fusion");
      // } else {
      //   float* dbias_tmp_ptr =
      //   RAII_GUARD.alloc_l3_or_gm<float>(dbias->numel()); r =
      //   xpu::fc_dbias_fusion<XPUType, XPUType, XPUType, float,
      //   int16_t>(xpu_ctx, reinterpret_cast<const XPUType*>(b_2),
      //   reinterpret_cast<const XPUType*>(a_2),
      //   reinterpret_cast<XPUType*>(c_2), m, n, k, trans_x,
      //       trans_y, max_x, max_y, max_out, ldx, ldy, ldout, 1.0f, 0.0f,
      //       nullptr, dbias_tmp_ptr);
      //   PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_dbias_fusion");
      //   r = xpu::cast<float, XPUType>(dev_ctx.x_context(),
      //                           dbias_tmp_ptr,
      //                           dbias_ptr,
      //                           dbias->numel());
      //   PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      // }
    }

    if (std::getenv("XPU_PADDLE_FC_PRINT") != nullptr) {
      auto tag_end = std::chrono::steady_clock::now();
      float time_use = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           tag_end - tag_start)
                           .count() /
                       1000000.0;
      printf("op: %s, %0.6f ms\n", "custom_fused_dense", time_use);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    custom_fused_dense,
    /*ops::CustomFusedDenseXPUKernel<phi::XPUContext, float>,*/
    ops::CustomFusedDenseXPUKernel<phi::XPUContext, paddle::platform::float16>);

REGISTER_OP_XPU_KERNEL(
    custom_fused_dense_grad,
    /*ops::CustomFusedDenseXPUGradKernel<phi::XPUContext, float>,*/
    ops::CustomFusedDenseXPUGradKernel<phi::XPUContext,
                                       paddle::platform::float16>);
