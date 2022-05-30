/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unordered_map>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

using framework::Tensor;

xpu::Pooling_t XPUPoolingType(const std::string& pooltype, bool exclusive,
                              bool is_test) {
  if (pooltype == "max") {
    return xpu::Pooling_t::MAX_WITHOUT_INDEX;
  } else if (pooltype == "avg") {
    if (exclusive) {
      return xpu::Pooling_t::AVG_WITHOUT_PAD;
    } else {
      return xpu::Pooling_t::AVG_WITH_PAD;
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Pool op only supports 2D and 3D input."));
  }
}

template <typename DeviceContext, typename T>
class PoolXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    PADDLE_ENFORCE_EQ(
        ksize.size(), 2,
        platform::errors::InvalidArgument(
            "The Pool2d XPU OP only support 2 dimension pooling!"));
    PADDLE_ENFORCE_EQ(!adaptive || (ksize[0] * ksize[1] == 1), true,
                      platform::errors::InvalidArgument(
                          "The Pool2d XPU OP does not support (adaptive == "
                          "true && output_size != 1)"));
    int* index_data = nullptr;
    bool global_pooling = context.Attr<bool>("global_pooling") ||
                          (adaptive && (ksize[0] * ksize[1] == 1));
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }
    const int n = in_x->dims()[0];
    const int c = in_x->dims()[1];
    const int in_h = in_x->dims()[2];
    const int in_w = in_x->dims()[3];
    auto input = reinterpret_cast<const XPUType*>(in_x->data<T>());
    out->mutable_data<T>(context.GetPlace());
    auto output = reinterpret_cast<XPUType*>(out->data<T>());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (pooling_type == "max") {
      r = xpu::max_pool2d<XPUType>(dev_ctx.x_context(), input, output,
                                   index_data, n, c, in_h, in_w, ksize, strides,
                                   paddings, true);
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool2d<XPUType>(dev_ctx.x_context(), input, output, n, c,
                                   in_h, in_w, ksize, strides, paddings,
                                   !exclusive, true);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The pool2d XPU API return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class PoolGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    const int* index_data = nullptr;
    PADDLE_ENFORCE_EQ(ksize.size(), 2, platform::errors::InvalidArgument(
                                           "The Pool2d XPU OP only support 2 "
                                           "dimension pooling!, but received "
                                           "%d-dimension pool kernel size",
                                           ksize.size()));
    PADDLE_ENFORCE_EQ(!adaptive || (ksize[0] * ksize[1] == 1), true,
                      platform::errors::InvalidArgument(
                          "The Pool2d XPU OP does not support (adaptive == "
                          "true && output_size != 1)"));
    bool global_pooling = context.Attr<bool>("global_pooling") ||
                          (adaptive && (ksize[0] * ksize[1] == 1));
    if (global_pooling) {
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x->dims()[i + 2]);
      }
    }
    if (!in_x_grad) {
      return;
    }
    const int n = in_x->dims()[0];
    const int c = in_x->dims()[1];
    const int in_h = in_x->dims()[2];
    const int in_w = in_x->dims()[3];
    auto input = reinterpret_cast<const XPUType*>(in_x->data<T>());
    auto output = reinterpret_cast<const XPUType*>(out->data<T>());
    auto output_grad = reinterpret_cast<const XPUType*>(out_grad->data<T>());
    in_x_grad->mutable_data<T>(context.GetPlace());
    auto input_grad = reinterpret_cast<XPUType*>(in_x_grad->data<T>());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (pooling_type == "max") {
      r = xpu::max_pool2d_grad<XPUType>(
          dev_ctx.x_context(), input, output, index_data, output_grad,
          input_grad, n, c, in_h, in_w, ksize, strides, paddings, true);
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool2d_grad<XPUType>(
          dev_ctx.x_context(), input, output, output_grad, input_grad, n, c,
          in_h, in_w, ksize, strides, paddings, !exclusive, true);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The Pool2dGrad XPU OP return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    pool2d, ops::PoolXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::PoolXPUKernel<paddle::platform::XPUDeviceContext,
                       paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    pool2d_grad,
    ops::PoolGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::PoolGradXPUKernel<paddle::platform::XPUDeviceContext,
                           paddle::platform::float16>);

#endif
