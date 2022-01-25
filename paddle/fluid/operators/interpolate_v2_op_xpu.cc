/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/interpolate_op.h"

#ifdef PADDLE_WITH_XPU

namespace paddle {
namespace operators {

using framework::Tensor;
using DataLayout = framework::DataLayout;

inline std::vector<int> get_new_shape_xpu(
    const std::vector<const Tensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(), framework::make_ddim({1}),
        platform::errors::InvalidArgument("shape of dim tensor should be [1]"));
    framework::Tensor temp;
    paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
    vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor_xpu(
    const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  framework::Tensor cpu_starts_tensor;
  paddle::framework::TensorCopySync(*new_data_tensor, platform::CPUPlace(),
                                    &cpu_starts_tensor);
  auto* new_data = cpu_starts_tensor.data<T>();
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <typename T>
class InterpolateV2XPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    PADDLE_ENFORCE_EQ(
        input_dims.size(), 4,
        platform::errors::External("XPU Interpolate kernel only support 2d"));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      // have size tensor
      auto new_size = get_new_shape_xpu(list_new_size_tensor);
      out_h = new_size[0];
      out_w = new_size[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor_xpu<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0, true,
            platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                              "should be greater than 0."));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0 && scale_h > 0, true,
              platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                                "should be greater than 0."));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
      auto out_size = ctx.Input<Tensor>("OutSize");
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_h = out_size_data[0];
        out_w = out_size_data[1];
      }
    }
    PADDLE_ENFORCE_GT(out_h, 0, platform::errors::InvalidArgument(
                                    "out_h in Attr(out_shape) of "
                                    "Op(interpolate) "
                                    "should be greater than 0."));
    PADDLE_ENFORCE_GT(out_w, 0, platform::errors::InvalidArgument(
                                    "out_w in Attr(out_shape) of "
                                    "Op(interpolate) "
                                    "should be greater than 0."));
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {n, c, out_h, out_w};
    } else {
      dim_out = {n, out_h, out_w, c};
    }
    output->mutable_data<T>(dim_out, ctx.GetPlace());

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*input, ctx.GetPlace(), output);
      return;
    }
    bool nearest = "nearest" == interp_method;
    int trans_mode = (align_corners) ? (0) : ((align_mode == 0) ? (1) : (2));
    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    if (nearest) {
      PADDLE_ENFORCE_EQ((data_layout == DataLayout::kNCHW), true,
                        platform::errors::InvalidArgument(
                            "XPU nearest is only support NCHW"));
    }
    int r = xpu::interpolate2d<T>(dev_ctx.x_context(), input->data<T>(),
                                  output->data<T>(), n, c, in_h, in_w, out_h,
                                  out_w, nearest, trans_mode,
                                  (data_layout == DataLayout::kNCHW));
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External("XPU interpolate2d kernel "
                                                 "return wrong value[%d %s]",
                                                 r, XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class InterpolateV2GradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto output_grad_dims = output_grad->dims();

    PADDLE_ENFORCE_EQ(output_grad_dims.size(), 4,
                      platform::errors::External(
                          "XPU Interpolategrad kernel only support 2d"));

    auto* input = ctx.Input<Tensor>("X");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    auto list_new_size_tensor = ctx.MultiInput<framework::Tensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      // have size tensor
      auto new_size = get_new_shape_xpu(list_new_size_tensor);
      out_h = new_size[0];
      out_w = new_size[1];
    } else {
      auto scale_tensor = ctx.Input<Tensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor_xpu<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0, true,
            platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                              "should be greater than 0."));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0 && scale_h > 0, true,
              platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                                "should be greater than 0."));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
      auto out_size = ctx.Input<Tensor>("OutSize");
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_h = out_size_data[0];
        out_w = out_size_data[1];
      }
    }

    framework::DDim dim_grad;
    if (data_layout == DataLayout::kNCHW) {
      dim_grad = {n, c, in_h, in_w};
    } else {
      dim_grad = {n, in_h, in_w, c};
    }
    input_grad->mutable_data<T>(dim_grad, ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    int r = XPU_SUCCESS;
    r = xpu::constant<T>(dev_ctx.x_context(), input_grad->data<T>(),
                         input_grad->numel(), static_cast<T>(0.0));
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU constant in interpolate2d_grad kernel return "
                          "wrong value[%d %s]",
                          r, XPUAPIErrorMsg[r]));

    if (in_h == out_h && in_w == out_w) {
      framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
      return;
    }

    bool nearest = "nearest" == interp_method;
    int trans_mode = (align_corners) ? (0) : ((align_mode == 0) ? (1) : (2));

    if (nearest) {
      trans_mode = (align_corners) ? (0) : (2);
    }

    r = xpu::interpolate2d_grad<T>(dev_ctx.x_context(), output_grad->data<T>(),
                                   input_grad->data<T>(), n, c, in_h, in_w,
                                   out_h, out_w, nearest, trans_mode,
                                   (data_layout == DataLayout::kNCHW));
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU interpolate2d_grad kernel return "
                                   "wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(bilinear_interp_v2, ops::InterpolateV2XPUKernel<float>);
REGISTER_OP_XPU_KERNEL(nearest_interp_v2, ops::InterpolateV2XPUKernel<float>);

REGISTER_OP_XPU_KERNEL(bilinear_interp_v2_grad,
                       ops::InterpolateV2GradXPUKernel<float>);
REGISTER_OP_XPU_KERNEL(nearest_interp_v2_grad,
                       ops::InterpolateV2GradXPUKernel<float>);
#endif
