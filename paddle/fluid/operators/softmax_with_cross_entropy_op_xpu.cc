/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/kernels/funcs/axis_utils.h"
#include "xpu/refactor/math.h"
#include "xpu/refactor/nn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SoftmaxWithCrossEntropyXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_xpu_place(context.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on XPU."));
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* softmax = context.Output<Tensor>("Softmax");
    Tensor* loss = context.Output<Tensor>("Loss");
    const int rank = logits->dims().size();
    const int axis = phi::funcs::CanonicalAxis(context.Attr<int>("axis"), rank);
    PADDLE_ENFORCE_EQ(axis, rank - 1, platform::errors::InvalidArgument(
                                          "axis should == rank - 1"));
    softmax->mutable_data<T>(context.GetPlace());
    loss->mutable_data<T>(context.GetPlace());
    const int n = phi::funcs::SizeToAxis(axis, logits->dims());
    const int d = phi::funcs::SizeFromAxis(axis, logits->dims());
    std::vector<int> logits_dims = phi::vectorize<int>(logits->dims());
    const bool soft_label = context.Attr<bool>("soft_label");

    auto logits_data = reinterpret_cast<const XPUType*>(logits->data<T>());
    auto softmax_data = reinterpret_cast<XPUType*>(softmax->data<T>());
    auto loss_data = reinterpret_cast<XPUType*>(loss->data<T>());

    // softmax
    auto& dev_ctx =
        context.template device_context<platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;
    if (platform::get_xpu_version(context.GetPlace().GetDeviceId()) ==
            phi::backends::xpu::XPUVersion::XPU2 &&
        soft_label) {
      auto labels_data = reinterpret_cast<const XPUType*>(labels->data<T>());
      r = xpu::soft_softmax_with_cross_entropy<XPUType>(
          dev_ctx.x_context(), logits_data, labels_data, softmax_data,
          loss_data, n, d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy");
      return;
    }

    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int len = logits->numel();
    T* clip_logits = RAII_GUARD.alloc_l3_or_gm<T>(len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(clip_logits);
    XPUType* clip_logits_data = reinterpret_cast<XPUType*>(clip_logits);

    float max_val = 1e20;
    float min_val = -1e20;
    if (std::is_same<T, platform::float16>::value) {
      max_val = 65504;
      min_val = -65504;
    }

    r = xpu::clip_v2<XPUType>(
        dev_ctx.x_context(), logits_data, clip_logits_data, len,
        static_cast<XPUType>(min_val), static_cast<XPUType>(max_val));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");

    r = xpu::softmax<XPUType>(dev_ctx.x_context(), clip_logits_data,
                              softmax_data, logits_dims, axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax");

    // cross_entropy
    if (soft_label) {
      auto labels_data = reinterpret_cast<const XPUType*>(labels->data<T>());
      r = xpu::soft_cross_entropy<XPUType>(dev_ctx.x_context(), softmax_data,
                                           labels_data, loss_data, n, d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_cross_entropy");
    } else {
      auto ignore_index = context.Attr<int>("ignore_index");
      Tensor labels_int32;
      labels_int32.mutable_data<int32_t>(context.GetPlace(),
                                         labels->numel() * sizeof(int32_t));
      r = xpu::cast_v2<int64_t, int32_t>(
          dev_ctx.x_context(), labels->data<int64_t>(),
          labels_int32.data<int32_t>(), labels->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "clip_v2");

      r = xpu::hard_cross_entropy<XPUType, int32_t>(
          dev_ctx.x_context(), softmax_data, labels_int32.data<int32_t>(),
          loss_data, nullptr, n, d, ignore_index);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_cross_entropy");
    }
  }
};

template <typename T>
class SoftmaxWithCrossEntropyGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Loss"));
    const Tensor* labels = context.Input<Tensor>("Label");
    Tensor* logit_grad =
        context.Output<Tensor>(framework::GradVarName("Logits"));

    logit_grad->mutable_data<T>(context.GetPlace());

    const Tensor* softmax = context.Input<Tensor>("Softmax");
    const bool use_softmax = context.Attr<bool>("use_softmax");

    const bool soft_label = context.Attr<bool>("soft_label");
    auto ignore_index = context.Attr<int>("ignore_index");

    const int rank = logit_grad->dims().size();
    const int axis = phi::funcs::CanonicalAxis(context.Attr<int>("axis"), rank);
    PADDLE_ENFORCE_EQ(axis, rank - 1, platform::errors::InvalidArgument(
                                          "axis should == rank - 1"));
    const int n = phi::funcs::SizeToAxis(axis, logit_grad->dims());
    const int d = phi::funcs::SizeFromAxis(axis, logit_grad->dims());

    auto& dev_ctx =
        context.template device_context<platform::XPUDeviceContext>();
    int r = XPU_SUCCESS;

    if (soft_label) {
      r = xpu::soft_softmax_with_cross_entropy_grad<XPUType>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(out_grad->data<T>()),
          reinterpret_cast<const XPUType*>(labels->data<T>()),
          reinterpret_cast<const XPUType*>(softmax->data<T>()),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()), use_softmax, n, d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "soft_softmax_with_cross_entropy_grad");
    } else {
      xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
      int* labels_int_ptr_l3 =
          RAII_GUARD.alloc_l3_or_gm<int32_t>(labels->numel());
      PADDLE_ENFORCE_XDNN_NOT_NULL(labels_int_ptr_l3);

      r = xpu::cast_v2<int64_t, int32_t>(dev_ctx.x_context(),
                                         labels->data<int64_t>(),
                                         labels_int_ptr_l3, labels->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");

      r = xpu::hard_softmax_with_cross_entropy_grad<XPUType, int>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(out_grad->data<T>()),
          labels_int_ptr_l3,
          reinterpret_cast<const XPUType*>(softmax->data<T>()),
          reinterpret_cast<XPUType*>(logit_grad->data<T>()), ignore_index,
          use_softmax, n, d);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "hard_softmax_with_cross_entropy_grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    softmax_with_cross_entropy, ops::SoftmaxWithCrossEntropyXPUKernel<float>,
    ops::SoftmaxWithCrossEntropyXPUKernel<paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    softmax_with_cross_entropy_grad,
    ops::SoftmaxWithCrossEntropyGradXPUKernel<float>,
    ops::SoftmaxWithCrossEntropyGradXPUKernel<paddle::platform::float16>);
#endif
