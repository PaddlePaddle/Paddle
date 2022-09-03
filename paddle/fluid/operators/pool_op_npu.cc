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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/funcs/pooling.h"

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
      data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      in_x_tensor.set_layout(DataLayout::kNHWC);
      out_tensor.set_layout(DataLayout::kNHWC);
    } else {
      data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);
#if (CANN_VERSION_CODE < 512000)
    PADDLE_ENFORCE_LT(
        std::max(paddings[0], paddings[1]),
        ksize[0],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
            ksize[0],
            std::max(paddings[0], paddings[1])));
    PADDLE_ENFORCE_LT(
        std::max(paddings[2], paddings[3]),
        ksize[1],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
            ksize[1],
            std::max(paddings[2], paddings[3])));
#endif
    if (adaptive) {
      std::string pooling_mode = "AdaptiveAvgPool2d";
      if (pooling_type == "max") {
        pooling_mode = "AdaptiveMaxPool2d";
      }

      // AdaptiveAvgPool2d only support NCHW
      Tensor transformed_input, transformed_output;
      if (pooling_type == "avg" && channel_last) {
        transformed_input.mutable_data<T>(
            phi::make_dim(
                in_x_dims[0], in_x_dims[3], in_x_dims[1], in_x_dims[2]),
            ctx.GetPlace());
        transformed_output.mutable_data<T>(
            phi::make_dim(out_dims[0], out_dims[3], out_dims[1], out_dims[2]),
            ctx.GetPlace());

        const auto &trans_runner =
            NpuOpRunner("TransData",
                        {in_x_tensor},
                        {transformed_input},
                        {{"src_format", std::string("NHWC")},
                         {"dst_format", std::string("NCHW")}});
        trans_runner.Run(dev_ctx.stream());
      } else {
        transformed_input.ShareDataWith(in_x_tensor);
        transformed_output.ShareDataWith(out_tensor);
      }

      const auto &runner =
          NpuOpRunner(pooling_mode,
                      {transformed_input},
                      {transformed_output},
                      {{"output_size", phi::vectorize<int>(out_data_dims)}});
      runner.Run(dev_ctx.stream());

      if (pooling_type == "avg" && channel_last) {
        const auto &trans_runner =
            NpuOpRunner("TransData",
                        {transformed_output},
                        {out_tensor},
                        {{"src_format", std::string("NCHW")},
                         {"dst_format", std::string("NHWC")}});
        trans_runner.Run(dev_ctx.stream());
      }
    } else {
      std::string pooling_mode = "AvgPoolV2";
      if (pooling_type == "max") {
        PADDLE_ENFORCE_EQ(
            exclusive,
            true,
            platform::errors::InvalidArgument(
                "MaxPool only support exclusive=false, but got true"));
        pooling_mode = "MaxPoolV3";
      }

      const auto &runner =
          NpuOpRunner(pooling_mode,
                      {in_x_tensor},
                      {out_tensor},
                      {{"ksize", ksize_vec},
                       {"strides", strides_vec},
                       {"padding_mode", std::string("CALCULATED")},
                       {"pads", paddings},
                       {"data_format", data_format},
                       {"global_pooling", global_pooling},
                       {"ceil_mode", ceil_mode},
                       {"exclusive", exclusive}});
      runner.Run(dev_ctx.stream());
    }
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
      data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
      out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
      ksize_vec[1] = ksize[0];
      ksize_vec[2] = ksize[1];
      strides_vec[1] = strides[0];
      strides_vec[2] = strides[1];
      in_x_tensor.set_layout(DataLayout::kNHWC);
      out_tensor.set_layout(DataLayout::kNHWC);
      out_grad_tensor.set_layout(DataLayout::kNHWC);
      in_x_grad_tensor.set_layout(DataLayout::kNHWC);
    } else {
      data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
      out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
      ksize_vec[2] = ksize[0];
      ksize_vec[3] = ksize[1];
      strides_vec[2] = strides[0];
      strides_vec[3] = strides[1];
    }
    phi::funcs::UpdatePadding(&paddings,
                              global_pooling,
                              adaptive,
                              padding_algorithm,
                              data_dims,
                              strides,
                              ksize);
#if (CANN_VERSION_CODE < 512000)
    PADDLE_ENFORCE_LT(
        std::max(paddings[0], paddings[1]),
        ksize[0],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
            ksize[0],
            std::max(paddings[0], paddings[1])));
    PADDLE_ENFORCE_LT(
        std::max(paddings[2], paddings[3]),
        ksize[1],
        platform::errors::InvalidArgument(
            "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
            ksize[1],
            std::max(paddings[2], paddings[3])));
#endif
    if (adaptive || (global_pooling && pooling_type == "max")) {
      PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0],
                        0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, H and W must be divisible, "
                            "but input dims is %s, output dims is %s",
                            data_dims,
                            out_data_dims));
      PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1],
                        0,
                        platform::errors::InvalidArgument(
                            "When adaptive = True, H and W must be divisible, "
                            "but input dims is %s, output dims is %s",
                            data_dims,
                            out_data_dims));
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

    NPUAttributeMap attrs = {{"ksize", ksize_vec},
                             {"strides", strides_vec},
                             {"padding_mode", std::string("CALCULATED")},
                             {"pads", paddings},
                             {"data_format", data_format},
                             {"global_pooling", global_pooling},
                             {"ceil_mode", ceil_mode},
                             {"exclusive", exclusive}};

    if (pooling_type == "max") {
      if (global_pooling) {
        for (auto &s : strides_vec) {
          s = 1;
        }
        PADDLE_ENFORCE_LT(std::max(data_dims[0], data_dims[1]),
                          255,
                          platform::errors::InvalidArgument(
                              "MaxPoolGrad H, W must be less than 255 when "
                              "global_pooling = True, but got %s",
                              data_dims));
        attrs["global_pooling"] = false;
      }

      const auto &runner =
          NpuOpRunner("MaxPoolV3Grad",
                      {in_x_tensor, out_tensor, out_grad_tensor},
                      {in_x_grad_tensor},
                      attrs);  // 0: floor, 1: ceil
      runner.Run(dev_ctx.stream());
    } else if (pooling_type == "avg") {
      PADDLE_ENFORCE(strides[0] == strides[1],
                     platform::errors::InvalidArgument(
                         "AvgPoolGrad dose not support Asymmetric strides. but "
                         "strides = (%d, %d)",
                         strides[0],
                         strides[1]));

      NpuOpRunner runner;
      runner.SetType("AvgPoolV2Grad");
      runner.AddInput(phi::vectorize<int>(in_x->dims()));
      runner.AddInput(out_grad_tensor);
      runner.AddOutput(in_x_grad_tensor);
      runner.AddAttrs(attrs);
      runner.Run(dev_ctx.stream());
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(pool2d,
                       ops::NPUPoolOpKernel<float>,
                       ops::NPUPoolOpKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(pool2d_grad,
                       ops::NPUPoolGradOpKernel<float>,
                       ops::NPUPoolGradOpKernel<plat::float16>);
