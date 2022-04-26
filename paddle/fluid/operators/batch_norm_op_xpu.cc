/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/batch_norm_op.h"
#include <iterator>
#include <vector>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class BatchNormXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    const auto is_test = ctx.Attr<bool>("is_test");
    const auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    const auto trainable_stats = ctx.Attr<bool>("trainable_statistics");
    bool test_mode = is_test && (!trainable_stats);

    bool global_stats = test_mode || use_global_stats;
    const auto &data_layout_str = ctx.Attr<std::string>("data_layout");
    const auto data_layout = framework::StringToDataLayout(data_layout_str);
    PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                      true,
                      platform::errors::InvalidArgument(
                          "The 'data_layout' attribute must be NCHW or NHWC. "
                          "But recevived 'data_layout' is [%s].",
                          data_layout_str));

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5"
            "But received: the size of input's dimensions is [%d]",
            x_dims.size()));

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");
    const auto *x_data = x->data<T>();
    const auto *scale_data = scale->data<float>();
    const auto *bias_data = bias->data<float>();

    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    // alloc memory
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    mean_out->mutable_data<float>(ctx.GetPlace());
    variance_out->mutable_data<float>(ctx.GetPlace());
    saved_mean->mutable_data<float>(ctx.GetPlace());
    saved_variance->mutable_data<float>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    bool is_nchw = data_layout_str == "NCHW";

    if (!global_stats) {
      auto *mean_out_data = mean_out->data<float>();
      auto *variance_out_data = variance_out->data<float>();
      auto *saved_mean_data = saved_mean->data<float>();
      auto *saved_variance_data = saved_variance->data<float>();

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        const auto *mom_tensor = ctx.Input<Tensor>("MomentumTensor");
        Tensor mom_cpu;
        paddle::framework::TensorCopySync(*mom_tensor, platform::CPUPlace(),
                                          &mom_cpu);
        momentum = mom_tensor->data<float>()[0];
      }

      int r = xpu::batch_norm<T>(dev_ctx.x_context(), x_data, y_data, N, C, H,
                                 W, epsilon, momentum, scale_data, bias_data,
                                 saved_mean_data, saved_variance_data,
                                 mean_out_data, variance_out_data, is_nchw);
      PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The batch_norm XPU API return wrong value[%d %s]",
                            r, XPUAPIErrorMsg[r]));
    } else {
      PADDLE_ENFORCE_EQ(
          data_layout_str == "NCHW", true,
          platform::errors::InvalidArgument(
              "The batch_norm_infer 'data_layout' attribute must be NCHW. "
              "But recevived 'data_layout' is [%s].",
              data_layout_str));
      const auto *mean = ctx.Input<Tensor>("Mean");
      const auto *variance = ctx.Input<Tensor>("Variance");
      const auto *mean_data = mean->data<float>();
      const auto *variance_data = variance->data<float>();
      int r = xpu::batch_norm_infer(dev_ctx.x_context(), x_data, y_data, N, C,
                                    H, W, epsilon, scale_data, bias_data,
                                    mean_data, variance_data, is_nchw);
      PADDLE_ENFORCE_EQ(
          r, xpu::Error_t::SUCCESS,
          platform::errors::External(
              "The batch_norm_infer XPU API return wrong value[%d %s]", r,
              XPUAPIErrorMsg[r]));
    }
  }
};

template <typename T>
static int calculate_inv_BN_Y(xpu::Context *ctx, T *x, const T *scale,
                              const T *bias, const T *mean, const T *variance,
                              const int N, const int C, const int M,
                              const T *y) {
  PADDLE_ENFORCE_EQ(x, y, platform::errors::InvalidArgument(
                              "X and Y should be inplaced in inplace mode"));
  std::vector<int> tensor_shape_vec({N, C, M});
  std::vector<int> array_shape_vec({1, C, 1});
  // y - bias
  int r1 =
      xpu::broadcast_sub<T>(ctx, bias, y, x, array_shape_vec, tensor_shape_vec);
  // (y - bias) / scale
  int r2 = xpu::broadcast_div<T>(ctx, scale, x, x, array_shape_vec,
                                 tensor_shape_vec);
  // (y - bias) / scale / variance
  int r3 = xpu::broadcast_div<T>(ctx, variance, x, x, array_shape_vec,
                                 tensor_shape_vec);
  // (y - bias) / scale / variance + mean
  int r4 =
      xpu::broadcast_add<T>(ctx, mean, x, x, array_shape_vec, tensor_shape_vec);

  return r1 + r2 + r3 + r4;
}

template <typename T>
static int calculate_inv_var(xpu::Context *ctx, const T *var, const T epsilon,
                             const int C, T *epsilon_data, T *inv_var) {
  int r1 = constant(ctx, epsilon_data, 1, epsilon);
  std::vector<int> tensor_shape_vec({C});
  std::vector<int> array_shape_vec({1});
  int r2 = xpu::broadcast_add<T>(ctx, epsilon_data, var, inv_var,
                                 array_shape_vec, tensor_shape_vec);
  int r3 = xpu::rsqrt<T>(ctx, inv_var, inv_var, C);
  return r1 + r2 + r3;
}

template <typename DeviceContext, typename T>
class BatchNormGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    const auto &data_layout_str = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool is_test = ctx.Attr<bool>("is_test");
    const float epsilon = ctx.Attr<float>("epsilon");
    const auto data_layout = framework::StringToDataLayout(data_layout_str);

    PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                      true,
                      platform::errors::InvalidArgument(
                          "The 'data_layout' attribute must be NCHW or NHWC. "
                          "But recevived 'data_layout' is [%s].",
                          data_layout_str));

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    use_global_stats = is_test || use_global_stats;

    // batch_norm with inplace as false will take X as grad input, which
    // is same as cuDNN batch_norm backward calculation, batch_norm
    // with inplace as true only take Y as input and X should be calculate
    // by inverse operation of batch_norm on Y
    const Tensor *x;
    bool is_inplace;
    if (ctx.HasInput("Y")) {
      x = ctx.Input<Tensor>("Y");
      is_inplace = true;
      // if the input of batch norm is stop_gradient, d_x is null.
      if (d_x) {
        PADDLE_ENFORCE_EQ(d_x, d_y,
                          platform::errors::InvalidArgument(
                              "X@GRAD and Y@GRAD not inplace in inplace mode"));
      }
    } else {
      x = ctx.Input<Tensor>("X");
      is_inplace = false;
      if (d_x) {
        PADDLE_ENFORCE_NE(
            d_x, d_y, platform::errors::InvalidArgument(
                          "X@GRAD and Y@GRAD inplaced in non-inplace mode"));
      }
    }

    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(
        x_dims.size() >= 2 && x_dims.size() <= 5, true,
        platform::errors::InvalidArgument(
            "The size of input's dimensions should be between 2 and 5"
            "But received: the size of input's dimensions is [%d]",
            x_dims.size()));

    int N, C, H, W, D;
    ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

    const auto *x_data = x->data<T>();
    const auto *d_y_data = d_y->data<T>();
    const auto *scale_data = scale->data<float>();

    // init output
    T *d_x_data = nullptr;
    T *d_bias_data = nullptr;
    T *d_scale_data = nullptr;
    if (d_x) {
      d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
    }
    if (d_scale && d_bias) {
      d_scale_data = d_scale->mutable_data<float>(ctx.GetPlace());
      d_bias_data = d_bias->mutable_data<float>(ctx.GetPlace());
    }

    PADDLE_ENFORCE_EQ(
        scale->dims().size(), 1UL,
        platform::errors::InvalidArgument(
            "The size of scale's dimensions must equal to 1. But received: "
            "the size of scale's dimensions is [%d], the dimensions of scale "
            "is [%s].",
            scale->dims().size(), scale->dims()));
    PADDLE_ENFORCE_EQ(
        scale->dims()[0], C,
        platform::errors::InvalidArgument(
            "The first dimension of scale must equal to Channels[%d]. But "
            "received: the first dimension of scale is [%d]",
            C, scale->dims()[0]));

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

    const auto *batch_mean = ctx.Input<Tensor>("SavedMean");
    const auto *batch_inv_std = ctx.Input<Tensor>("SavedVariance");
    const auto *global_mean = ctx.Input<Tensor>("Mean");
    const auto *global_var = ctx.Input<Tensor>("Variance");

    // TODO(guozibin): hadle the situation case of N * H * W = 1
    if (is_inplace) {
      float *global_inv_std_data = nullptr;
      if (use_global_stats) {
        global_inv_std_data =
            RAII_GUARD.alloc_l3_or_gm<float>(global_var->numel());
        float *epsilon_data = RAII_GUARD.alloc_l3_or_gm<float>(1);
        int r1 =
            calculate_inv_var(dev_ctx.x_context(), global_var->data<float>(),
                              epsilon, C, epsilon_data, global_inv_std_data);
        PADDLE_ENFORCE_EQ(r1, XPU_SUCCESS, platform::errors::External(
                                               "XPU API(batch_norm_grad "
                                               "calculate_inv_var function) "
                                               "return wrong value[%d %s]",
                                               r1, XPUAPIErrorMsg[r1]));
      }
      auto px = *x;
      auto *inv_std_data =
          use_global_stats ? global_inv_std_data : batch_inv_std->data<float>();
      auto mean_data = use_global_stats ? global_mean->data<float>()
                                        : batch_mean->data<float>();
      int r2 = calculate_inv_BN_Y(
          dev_ctx.x_context(), px.mutable_data<T>(ctx.GetPlace()),
          scale->data<float>(), bias->data<float>(), mean_data, inv_std_data, N,
          C, H * W, x->data<T>());
      PADDLE_ENFORCE_EQ(r2, XPU_SUCCESS, platform::errors::External(
                                             "XPU API(batch_norm_grad "
                                             "calculate_inv_BN_Y function) "
                                             "return wrong value[%d %s]",
                                             r2, XPUAPIErrorMsg[r2]));
    }

    int r3;
    bool is_nchw = data_layout_str == "NCHW";
    if (use_global_stats) {
      r3 = xpu::batch_norm_grad<T>(
          dev_ctx.x_context(), x_data, d_y_data, d_x_data, N, C, H, W,
          scale_data, nullptr, nullptr, d_scale_data, d_bias_data, is_nchw,
          global_mean->data<float>(), global_var->data<float>(), epsilon);
    } else {
      if (!d_x) {
        d_x_data = RAII_GUARD.alloc_l3_or_gm<T>(x->numel());
      }
      if (!d_scale) {
        d_scale_data = RAII_GUARD.alloc_l3_or_gm<float>(C);
      }
      if (!d_bias_data) {
        d_bias_data = RAII_GUARD.alloc_l3_or_gm<float>(C);
      }
      r3 = xpu::batch_norm_grad<T>(
          dev_ctx.x_context(), x_data, d_y_data, d_x_data, N, C, H, W,
          scale_data, batch_mean->data<float>(), batch_inv_std->data<float>(),
          d_scale_data, d_bias_data, is_nchw);
    }
    PADDLE_ENFORCE_EQ(r3, XPU_SUCCESS, platform::errors::External(
                                           "XPU API(batch_norm_grad) return "
                                           "wrong value[%d %s]",
                                           r3, XPUAPIErrorMsg[r3]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    batch_norm,
    ops::BatchNormXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    batch_norm_grad,
    ops::BatchNormGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
