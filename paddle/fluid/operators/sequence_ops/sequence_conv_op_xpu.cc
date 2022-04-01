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

#include "paddle/fluid/operators/sequence_ops/sequence_conv_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SequenceConvXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto filter = *context.Input<Tensor>("Filter");

    out->mutable_data<T>(context.GetPlace());

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(
        in->lod().empty(), false,
        platform::errors::InvalidArgument("Input(X) Tensor of SequenceConvOp "
                                          "does not contain LoD information."));
    PADDLE_ENFORCE_EQ(
        in->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "Only support input sequence with lod level equal to 1 at "
            "present. But received: lod level %u.",
            in->lod().size()));

    PADDLE_ENFORCE_EQ(
        padding_trainable, false,
        platform::errors::InvalidArgument("Only support padding_trainable "
                                          "equal false."));

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    PADDLE_ENFORCE_EQ(up_pad, 2, platform::errors::InvalidArgument(
                                     "Only support up_pad equal 2."));
    PADDLE_ENFORCE_EQ(down_pad, 2, platform::errors::InvalidArgument(
                                       "Only support down_pad equal 2."));

    auto xpu_context =
        context.template device_context<DeviceContext>().x_context();
    auto sequence_width = static_cast<int64_t>(in->dims()[1]);
    framework::DDim col_shape = {in->dims()[0],
                                 context_length * sequence_width};
    xpu::ctx_guard RAII_GUARD(xpu_context);
    int col_numel = col_shape[0] * col_shape[1];
    T* col_data = RAII_GUARD.alloc_l3_or_gm<T>(col_numel);
    PADDLE_ENFORCE_NOT_NULL(
        col_data, paddle::platform::errors::Fatal("XPU memory is not enough"));

    auto lod_level_0 = in->lod()[0];
    int lod_size = lod_level_0.size();
    // If batch size set to 256, the lod is {0, batch[0] - 0,
    // batch[1] - batch [0], ..., batch[255] - batch[254]},
    // so the lod_size will be 257.
    PADDLE_ENFORCE_LE(lod_size, 257, platform::errors::InvalidArgument(
                                         "Only support batch size <= 256."));

    std::vector<int> cpu_lodx(lod_size);
    for (int i = 0; i < lod_size; i++) {
      cpu_lodx[i] = lod_level_0[i];
    }
    xpu::VectorParam<int> lodx = {cpu_lodx.data(),
                                  static_cast<int>(cpu_lodx.size()), nullptr};

    int r = xpu::sequence_context_projection<T, int>(
        xpu_context, in->data<T>(), col_data, nullptr, lodx, sequence_width,
        context_start, context_length, context_stride, {2, 2});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sequence_context_projection");

    bool trans_a = false;
    bool trans_b = false;
    int m = col_shape[0];
    int k = col_shape[1];
    int k1 = filter.dims()[0];
    int n = filter.dims()[1];
    PADDLE_ENFORCE_EQ(k, k1,
                      platform::errors::InvalidArgument(
                          "The shape of FC in SequenceConvOp is invalid."
                          "The k of matrix A is %d, k1 of matrix B is %d."
                          "But expect k == k1",
                          k, k1));
    int lda = (!trans_a) ? k : m;
    int ldb = (!trans_b) ? n : k;
    int ldc = n;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    const T* data_a = col_data;
    const T* data_b = filter.data<T>();
    T* data_c = out->data<T>();

    r = xpu::fc_fusion<T, T, T, int32_t>(
        xpu_context, data_a, data_b, data_c, m, n, k, trans_a, trans_b, nullptr,
        nullptr, nullptr, lda, ldb, ldc, alpha, beta, nullptr,
        xpu::Activation_t::LINEAR);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
    if (xpu_context->xpu_stream != nullptr) {
      xpu_wait(xpu_context->xpu_stream);
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceConvGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* out_g = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* filter_g = context.Output<Tensor>(framework::GradVarName("Filter"));
    auto* in = context.Input<LoDTensor>("X");
    auto* filter = context.Input<Tensor>("Filter");

    int context_start = context.Attr<int>("contextStart");
    int context_length = context.Attr<int>("contextLength");
    int context_stride = context.Attr<int>("contextStride");
    bool padding_trainable = context.Attr<bool>("paddingTrainable");

    PADDLE_ENFORCE_EQ(
        in->lod().empty(), false,
        platform::errors::InvalidArgument("Input(X) Tensor of SequenceConvOp "
                                          "does not contain LoD information."));
    PADDLE_ENFORCE_EQ(
        in->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "Only support input sequence with lod level equal to 1 at "
            "present. But received: lod level %u.",
            in->lod().size()));

    PADDLE_ENFORCE_EQ(
        padding_trainable, false,
        platform::errors::InvalidArgument("Only support padding_trainable "
                                          "equal false."));

    int up_pad = std::max(0, -context_start);
    int down_pad = std::max(0, context_start + context_length - 1);
    PADDLE_ENFORCE_EQ(up_pad, 2, platform::errors::InvalidArgument(
                                     "Only support up_pad equal 2."));
    PADDLE_ENFORCE_EQ(down_pad, 2, platform::errors::InvalidArgument(
                                       "Only support down_pad equal 2."));

    auto lod_level_0 = in->lod()[0];
    int lod_size = lod_level_0.size();
    PADDLE_ENFORCE_LE(lod_size, 257, platform::errors::InvalidArgument(
                                         "Only support batch size <= 256."));

    std::vector<int> cpu_lodx(lod_size);
    for (int i = 0; i < lod_size; i++) {
      cpu_lodx[i] = lod_level_0[i];
    }
    xpu::VectorParam<int> lodx = {cpu_lodx.data(),
                                  static_cast<int>(cpu_lodx.size()), nullptr};

    auto xpu_context =
        context.template device_context<DeviceContext>().x_context();
    auto sequence_width = static_cast<int64_t>(in->dims()[1]);
    framework::DDim col_shape = {in->dims()[0],
                                 context_length * sequence_width};
    xpu::ctx_guard RAII_GUARD(xpu_context);
    int col_numel = col_shape[0] * col_shape[1];
    T* col_data = RAII_GUARD.alloc_l3_or_gm<T>(col_numel);
    PADDLE_ENFORCE_NOT_NULL(
        col_data, paddle::platform::errors::Fatal("XPU memory is not enough"));

    if (in_g || filter_g) {
      bool trans_a = false;
      bool trans_b = true;
      int m = out_g->dims()[0];
      int k = out_g->dims()[1];
      int n = filter->dims()[0];
      int k1 = filter->dims()[1];
      PADDLE_ENFORCE_EQ(k, k1,
                        platform::errors::InvalidArgument(
                            "The shape of FC in SequenceConvGradOp is invalid."
                            "The k of matrix A is %d, k1 of matrix B is %d."
                            "But expect k == k1",
                            k, k1));
      int lda = (!trans_a) ? k : m;
      int ldb = (!trans_b) ? n : k;
      int ldc = n;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);
      const T* data_a = out_g->data<T>();
      const T* data_b = filter->data<T>();
      T* data_c = col_data;

      int r = xpu::fc_fusion<T, T, T, int32_t>(
          xpu_context, data_a, data_b, data_c, m, n, k, trans_a, trans_b,
          nullptr, nullptr, nullptr, lda, ldb, ldc, alpha, beta, nullptr,
          xpu::Activation_t::LINEAR);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
    }

    if (in_g) {
      PADDLE_ENFORCE_LT(sequence_width, 512,
                        platform::errors::InvalidArgument(
                            "Only support sequence_width < 512."));

      in_g->mutable_data<T>(context.GetPlace());
      in_g->set_lod(in->lod());

      int r = xpu::sequence_context_projection_grad<T, int>(
          xpu_context, in_g->data<T>(), col_data, nullptr, lodx, sequence_width,
          context_start, context_length, context_stride, {2, 2});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "sequence_context_projection_grad");
    }

    if (filter_g) {
      filter_g->mutable_data<T>(context.GetPlace());

      int r = xpu::sequence_context_projection<T, int>(
          xpu_context, in->data<T>(), col_data, nullptr, lodx, sequence_width,
          context_start, context_length, context_stride, {2, 2});
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "sequence_context_projection");

      bool trans_a = true;
      bool trans_b = false;
      int k = col_shape[0];
      int m = col_shape[1];
      int k1 = out_g->dims()[0];
      int n = out_g->dims()[1];
      PADDLE_ENFORCE_EQ(k, k1,
                        platform::errors::InvalidArgument(
                            "The shape of FC in SequenceConvGradOp is invalid."
                            "The k of matrix A is %d, k1 of matrix B is %d."
                            "But expect k == k1",
                            k, k1));
      int lda = (!trans_a) ? k : m;
      int ldb = (!trans_b) ? n : k;
      int ldc = n;
      T alpha = static_cast<T>(1.0);
      T beta = static_cast<T>(0.0);
      const T* data_a = col_data;
      const T* data_b = out_g->data<T>();
      T* data_c = filter_g->data<T>();

      r = xpu::fc_fusion<T, T, T, int32_t>(
          xpu_context, data_a, data_b, data_c, m, n, k, trans_a, trans_b,
          nullptr, nullptr, nullptr, lda, ldb, ldc, alpha, beta, nullptr,
          xpu::Activation_t::LINEAR);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc_fusion");
      if (xpu_context->xpu_stream != nullptr) {
        xpu_wait(xpu_context->xpu_stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    sequence_conv,
    ops::SequenceConvXPUKernel<paddle::platform::XPUDeviceContext, float>);

REGISTER_OP_XPU_KERNEL(
    sequence_conv_grad,
    ops::SequenceConvGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
