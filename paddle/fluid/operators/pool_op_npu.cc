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
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/fill_constant_op.h"

#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUPoolOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const Tensor *in_x = ctx.Input<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::string data_format = ctx.Attr<std::string>("data_format");

    bool global_pooling = ctx.Attr<bool>("global_pooling");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    auto in_x_dims = in_x->dims();
    auto out_dims = out->dims();
    framework::DDim data_dims;
    framework::DDim out_data_dims;
    Tensor in_x_tensor, out_tensor;
    in_x_tensor.ShareDataWith(*in_x);
    out_tensor.ShareDataWith(*out);
    std::vector<int> ksize_vec(4, 1);
    std::vector<int> strides_vec(4, 1);
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = framework::slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      in_x_tensor.set_layout(DataLayout::kNHWC);
      out_tensor.set_layout(DataLayout::kNHWC);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = framework::slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    PADDLE_ENFORCE_LT(
        std::max(paddings[0], paddings[1]), ksize[0],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
            ksize[0], std::max(paddings[0], paddings[1])));
    PADDLE_ENFORCE_LT(
        std::max(paddings[2], paddings[3]), ksize[1],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
            ksize[1], std::max(paddings[2], paddings[3])));
    if (adaptive) {
      PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0], 0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, the H and W of input must "
                            "be divisible by the output, "
                            "but x dims is %s, out dims is %s",
                            data_dims, out_data_dims));
      PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1], 0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, the H and W of input must "
                            "be divisible by the output,, "
                            "but x dims is %s, out dims is %s",
                            data_dims, out_data_dims));
      if (channel_last) {
        strides_vec[1] = data_dims[0] / out_data_dims[0];
        strides_vec[2] = data_dims[1] / out_data_dims[1];
        ksize_vec[1] = strides_vec[1];
        ksize_vec[2] = strides_vec[2];
      } else {
        strides_vec[2] = data_dims[0] / out_data_dims[0];
        strides_vec[3] = data_dims[1] / out_data_dims[1];
        ksize_vec[2] = strides_vec[2];
        ksize_vec[3] = strides_vec[3];
      }
    }

    std::string pooling_mode = "AvgPoolV2";
    if (pooling_type == "max") {
      PADDLE_ENFORCE_EQ(
          exclusive, true,
          platform::errors::InvalidArgument(
              "MaxPool only support exclusive=false, but got true"));
      pooling_mode = "MaxPoolV3";
    }
    const auto &runner =
        NpuOpRunner(pooling_mode, {in_x_tensor}, {out_tensor},
                    {{"ksize", ksize_vec},
                     {"strides", strides_vec},
                     {"padding_mode", std::string("CALCULATED")},
                     {"pads", paddings},
                     {"data_format", data_format},
                     {"global_pooling", global_pooling},
                     {"ceil_mode", ceil_mode},
                     {"exclusive", exclusive}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
};

template <typename T>
class NPUPoolGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    const Tensor *in_x = ctx.Input<Tensor>("X");
    const Tensor *out = ctx.Input<Tensor>("Out");
    const Tensor *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor *in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    in_x_grad->mutable_data<T>(ctx.GetPlace());

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");
    std::vector<int> ksize = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    bool ceil_mode = ctx.Attr<bool>("ceil_mode");
    bool exclusive = ctx.Attr<bool>("exclusive");
    bool adaptive = ctx.Attr<bool>("adaptive");
    std::string data_format = ctx.Attr<std::string>("data_format");
    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    const bool channel_last = data_format == "NHWC";

    // update paddings
    auto in_x_dims = in_x->dims();
    auto out_dims = out->dims();
    framework::DDim data_dims;
    framework::DDim out_data_dims;
    std::vector<int> ksize_vec(4, 1);
    std::vector<int> strides_vec(4, 1);
    Tensor in_x_tensor, out_tensor, out_grad_tensor, in_x_grad_tensor;
    in_x_tensor.ShareDataWith(*in_x);
    out_tensor.ShareDataWith(*out);
    out_grad_tensor.ShareDataWith(*out_grad);
    in_x_grad_tensor.ShareDataWith(*in_x_grad);
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = framework::slice_ddim(out_dims, 1, out_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      in_x_tensor.set_layout(DataLayout::kNHWC);
      out_tensor.set_layout(DataLayout::kNHWC);
      out_grad_tensor.set_layout(DataLayout::kNHWC);
      in_x_grad_tensor.set_layout(DataLayout::kNHWC);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = framework::slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (global_pooling) {
      adaptive = true;
    }
    PADDLE_ENFORCE_LT(
        std::max(paddings[0], paddings[1]), ksize[0],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
            ksize[0], std::max(paddings[0], paddings[1])));
    PADDLE_ENFORCE_LT(
        std::max(paddings[2], paddings[3]), ksize[1],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
            ksize[1], std::max(paddings[2], paddings[3])));

    if (adaptive) {
      PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0], 0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, H and W must be divisible, "
                            "but input dims is %s, output dims is %s",
                            data_dims, out_data_dims));
      PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1], 0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, H and W must be divisible, "
                            "but input dims is %s, output dims is %s",
                            data_dims, out_data_dims));
      if (channel_last) {
        strides_vec[1] = data_dims[0] / out_data_dims[0];
        strides_vec[2] = data_dims[1] / out_data_dims[1];
        ksize_vec[1] = strides_vec[1];
        ksize_vec[2] = strides_vec[2];
      } else {
        strides_vec[2] = data_dims[0] / out_data_dims[0];
        strides_vec[3] = data_dims[1] / out_data_dims[1];
        ksize_vec[2] = strides_vec[2];
        ksize_vec[3] = strides_vec[3];
      }
    }

    if (pooling_type == "max") {
      if (global_pooling) {
        for (auto &s : strides_vec) {
          s = 1;
        }
        PADDLE_ENFORCE_LT(std::max(data_dims[0], data_dims[1]), 255,
                          platform::errors::InvalidArgument(
                              "MaxPoolV3Grad H, W must be less than 255 when "
                              "global_pooling = True, but got %s",
                              data_dims));
        global_pooling = false;
      }
      const auto &runner = NpuOpRunner(
          "MaxPoolV3Grad", {in_x_tensor, out_tensor, out_grad_tensor},
          {in_x_grad_tensor}, {{"ksize", ksize_vec},
                               {"strides", strides_vec},
                               {"padding_mode", std::string("CALCULATED")},
                               {"pads", paddings},
                               {"data_format", data_format},
                               {"global_pooling", global_pooling},
                               {"ceil_mode", ceil_mode},
                               {"exclusive", exclusive}});  // 0: floor, 1: ceil
      runner.Run(dev_ctx.stream());
    } else if (pooling_type == "avg") {
      auto cpu_dev_ctx = platform::CPUDeviceContext(platform::CPUPlace());
      Tensor cpu_in_x, cpu_out, cpu_in_x_grad, cpu_out_grad;
      cpu_in_x.mutable_data<T>(in_x->dims(), cpu_dev_ctx.GetPlace());
      cpu_in_x_grad.mutable_data<T>(in_x_grad->dims(), cpu_dev_ctx.GetPlace());
      cpu_out.mutable_data<T>(out->dims(), cpu_dev_ctx.GetPlace());
      cpu_out_grad.mutable_data<T>(out_grad->dims(), cpu_dev_ctx.GetPlace());

      framework::TensorCopy(*in_x, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_in_x);
      framework::TensorCopy(*out, cpu_dev_ctx.GetPlace(), dev_ctx, &cpu_out);
      framework::TensorCopy(*out_grad, cpu_dev_ctx.GetPlace(), dev_ctx,
                            &cpu_out_grad);
      math::SetConstant<platform::CPUDeviceContext, T> set_constant;
      set_constant(cpu_dev_ctx, &cpu_in_x_grad, static_cast<T>(0));
      dev_ctx.Wait();

      paddle::operators::math::Pool2dGradFunctor<
          platform::CPUDeviceContext, paddle::operators::math::AvgPoolGrad<T>,
          T>
          pool2d_backward;
      paddle::operators::math::AvgPoolGrad<T> pool_process;
      pool2d_backward(cpu_dev_ctx, cpu_in_x, cpu_out, cpu_out_grad, ksize,
                      strides, paddings, data_format, exclusive, adaptive,
                      &cpu_in_x_grad, pool_process);
      framework::TensorCopy(cpu_in_x_grad, dev_ctx.GetPlace(), dev_ctx,
                            in_x_grad);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(pool2d, ops::NPUPoolOpKernel<float>);
REGISTER_OP_NPU_KERNEL(pool2d_grad, ops::NPUPoolGradOpKernel<float>);
