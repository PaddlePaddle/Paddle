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
#include "paddle/phi/kernels/funcs/pooling.h"

#ifdef PADDLE_WITH_XPU
namespace paddle {
namespace operators {

using framework::Tensor;

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
    bool ceil_mode = context.Attr<bool>("ceil_mode");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    PADDLE_ENFORCE_EQ(
        ksize.size(),
        2,
        platform::errors::InvalidArgument(
            "The Pool2d XPU OP only support 2 dimension pooling!"));

    std::string data_format = context.Attr<std::string>("data_format");
    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        platform::errors::InvalidArgument("The Pool2d XPU OP only support"
                                          "data_format is 'NCHW', but received "
                                          "%s",
                                          data_format));

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

    const int out_h = out->dims()[2];
    const int out_w = out->dims()[3];

    framework::DDim data_dims;

    data_dims = phi::slice_ddim(in_x->dims(), 2, in_x->dims().size());
    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);

    if (ceil_mode) {
      int in_h_ceil = (out_h - 1) * strides[0] + ksize[0] - 2 * paddings[0];
      int in_w_ceil = (out_w - 1) * strides[1] + ksize[1] - 2 * paddings[2];

      paddings[1] += (in_h_ceil - in_h);
      paddings[3] += (in_w_ceil - in_w);
    }

    auto input = reinterpret_cast<const XPUType*>(in_x->data<T>());
    out->mutable_data<T>(context.GetPlace());
    auto output = reinterpret_cast<XPUType*>(out->data<T>());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (!adaptive) {
      if (pooling_type == "max") {
        r = xpu::max_pool2d<XPUType>(dev_ctx.x_context(),
                                     input,
                                     output,
                                     index_data,
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     ksize,
                                     strides,
                                     paddings,
                                     true);
      } else if (pooling_type == "avg") {
        r = xpu::avg_pool2d<XPUType>(dev_ctx.x_context(),
                                     input,
                                     output,
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     ksize,
                                     strides,
                                     paddings,
                                     !exclusive,
                                     true);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported pooling type for kunlun ", pooling_type));
      }
    } else {
      if (pooling_type == "max") {
        r = xpu::adaptive_max_pool2d<XPUType>(dev_ctx.x_context(),
                                              input,
                                              output,
                                              index_data,
                                              n,
                                              c,
                                              in_h,
                                              in_w,
                                              out_h,
                                              out_w,
                                              true);
      } else if (pooling_type == "avg") {
        r = xpu::adaptive_avg_pool2d<XPUType>(dev_ctx.x_context(),
                                              input,
                                              output,
                                              n,
                                              c,
                                              in_h,
                                              in_w,
                                              out_h,
                                              out_w,
                                              true);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported pooling type for kunlun ", pooling_type));
      }
    }
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The pool2d XPU API return wrong value[%d %s]",
                          r,
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
    bool ceil_mode = context.Attr<bool>("ceil_mode");

    std::string data_format = context.Attr<std::string>("data_format");
    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        platform::errors::InvalidArgument("The Pool2d_grad XPU OP only support"
                                          "data_format is 'NCHW', but received "
                                          "%s",
                                          data_format));

    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const int* index_data = nullptr;
    PADDLE_ENFORCE_EQ(
        ksize.size(),
        2,
        platform::errors::InvalidArgument("The Pool2d XPU OP only support 2 "
                                          "dimension pooling!, but received "
                                          "%d-dimension pool kernel size",
                                          ksize.size()));
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

    const int out_h = out->dims()[2];
    const int out_w = out->dims()[3];

    framework::DDim data_dims;

    data_dims = phi::slice_ddim(in_x->dims(), 2, in_x->dims().size());
    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);
    if (ceil_mode) {
      int in_h_ceil = (out_h - 1) * strides[0] + ksize[0] - 2 * paddings[0];
      int in_w_ceil = (out_w - 1) * strides[1] + ksize[1] - 2 * paddings[2];

      paddings[1] += (in_h_ceil - in_h);
      paddings[3] += (in_w_ceil - in_w);
    }

    auto input = reinterpret_cast<const XPUType*>(in_x->data<T>());
    auto output = reinterpret_cast<const XPUType*>(out->data<T>());
    auto output_grad = reinterpret_cast<const XPUType*>(out_grad->data<T>());
    in_x_grad->mutable_data<T>(context.GetPlace());
    auto input_grad = reinterpret_cast<XPUType*>(in_x_grad->data<T>());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (adaptive) {
      // floor for stride
      strides = {in_h / out_h, in_w / out_w};
      int kh = in_h - (out_h - 1) * strides[0];
      int kw = in_w - (out_w - 1) * strides[1];
      ksize = {kh, kw};
      paddings = {0, 0, 0, 0};
    }

    if (pooling_type == "max") {
      // TODO(zhanghuan05) to bind max_pool2d_grad_indices xpu api
      r = xpu::max_pool2d_grad<XPUType>(dev_ctx.x_context(),
                                        input,
                                        output,
                                        index_data,
                                        output_grad,
                                        input_grad,
                                        n,
                                        c,
                                        in_h,
                                        in_w,
                                        ksize,
                                        strides,
                                        paddings,
                                        true);
    } else if (pooling_type == "avg") {
      r = xpu::avg_pool2d_grad<XPUType>(dev_ctx.x_context(),
                                        input,
                                        output,
                                        output_grad,
                                        input_grad,
                                        n,
                                        c,
                                        in_h,
                                        in_w,
                                        ksize,
                                        strides,
                                        paddings,
                                        !exclusive,
                                        true);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported pooling type for kunlun ", pooling_type));
    }
    PADDLE_ENFORCE_EQ(r,
                      xpu::Error_t::SUCCESS,
                      platform::errors::External(
                          "The Pool2dGrad XPU OP return wrong value[%d %s]",
                          r,
                          XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    pool2d,
    ops::PoolXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::PoolXPUKernel<paddle::platform::XPUDeviceContext,
                       paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    pool2d_grad,
    ops::PoolGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::PoolGradXPUKernel<paddle::platform::XPUDeviceContext,
                           paddle::platform::float16>);

#endif
