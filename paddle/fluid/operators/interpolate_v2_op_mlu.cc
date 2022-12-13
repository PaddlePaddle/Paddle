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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/interpolate_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

inline std::vector<int> get_new_shape_mlu(
    const std::vector<const phi::DenseTensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        phi::make_ddim({1}),
        platform::errors::InvalidArgument("shape of dim tensor should be [1]"));
    phi::DenseTensor temp;
    paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(), &temp);
    vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
  }

  return vec_new_shape;
}

template <typename T>
class InterpolateV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<MLUDeviceContext>();
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");

    auto input_dims = input->dims();
    PADDLE_ENFORCE_GE(
        input_dims.size(),
        4,
        platform::errors::External("MLU Interpolate kernel supports input "
                                   "range greater or equal than 4."));
    PADDLE_ENFORCE_LE(
        input_dims.size(),
        5,
        platform::errors::External("MLU Interpolate kernel supports input "
                                   "range less or equal than 5. "));

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");
    int align_center = align_corners ? 0 : (align_mode == 1 ? 0 : 1);

    int out_d = ctx.Attr<int>("out_d");
    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_d = -1;
    float scale_h = -1;
    float scale_w = -1;

    auto list_new_size_tensor = ctx.MultiInput<phi::DenseTensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      // have size tensor
      auto new_size = get_new_shape_mlu(list_new_size_tensor);
      if (new_size.size() <= 2) {
        // default NCHW
        out_h = new_size[0];
        out_w = new_size[1];
      } else {
        // rank of input is 5, HCDHW
        out_d = new_size[0];
        out_h = new_size[1];
        out_w = new_size[2];
      }
    } else {
      auto scale_tensor = ctx.Input<phi::DenseTensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        std::vector<float> scale_data;
        scale_data = GetDataFromTensor<float>(scale_tensor);

        if (scale_data.size() > 1 && scale_data.size() <= 2) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else if (scale_data.size() > 2) {
          scale_d = scale_data[0];
          scale_h = scale_data[1];
          scale_w = scale_data[2];
        } else {
          scale_d = scale_data[0];
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0,
            true,
            platform::errors::InvalidArgument("scale of Op(interpolate) "
                                              "should be greater than 0."));
      } else {
        if (scale.size() > 1 && scale.size() <= 2) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0 && scale_h > 0,
              true,
              platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                                "should be greater than 0."));
        } else if (scale.size() > 2) {
          scale_d = scale[0];
          scale_h = scale[1];
          scale_w = scale[2];
          PADDLE_ENFORCE_EQ(
              scale_d > 0 && scale_w > 0 && scale_h > 0,
              true,
              platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                                "should be greater than 0."));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }

      if (scale_d > 0.) {
        out_d = static_cast<int>(in_d * scale_d);
      }
      auto out_size = ctx.Input<phi::DenseTensor>("OutSize");
      if (out_size != nullptr) {
        std::vector<int32_t> out_size_data;
        out_size_data = GetDataFromTensor<int>(out_size);
        if (out_size_data.size() <= 2) {
          out_h = out_size_data[0];
          out_w = out_size_data[1];
        } else {
          out_d = out_size_data[0];
          out_h = out_size_data[1];
          out_w = out_size_data[2];
        }
      }
    }
    PADDLE_ENFORCE_GT(
        out_h,
        0,
        platform::errors::InvalidArgument("out_h in Attr(out_shape) of "
                                          "Op(interpolate) "
                                          "should be greater than 0."));
    PADDLE_ENFORCE_GT(
        out_w,
        0,
        platform::errors::InvalidArgument("out_w in Attr(out_shape) of "
                                          "Op(interpolate) "
                                          "should be greater than 0."));

    // do transpose according to cnnl's constraints
    // cnnlInterp_v2 only accepts NHWC when mode is CNNL_INTERP_BILINEAR and
    // CNNL_INTERP_NEAREST,
    framework::DDim dim_in, dim_in_trans, dim_out, dim_out_trans;
    phi::DenseTensor transformed_input, transformed_output;
    bool need_transpose = input_dims.size() != 2;
    if (input_dims.size() == 4) {
      // need to do transpose if layout is kNCHW
      need_transpose &= data_layout == DataLayout::kNCHW;
      if (need_transpose) {
        // if need_transpose, do the following
        // 1. transpose input NCHW -> NHWC
        // 2. interpolation in(NHWC) -> out(NHWC)
        // 3. transpose output NHWC -> HCHW
        // dim_in = {n, c, in_h, in_w};
        dim_in_trans = {n, in_h, in_w, c};
        dim_out = {n, c, out_h, out_w};
        dim_out_trans = {n, out_h, out_w, c};
        output->mutable_data<T>(dim_out, ctx.GetPlace());

        if (in_h == out_h && in_w == out_w) {
          framework::TensorCopy(*input, ctx.GetPlace(), output);
          return;
        }
        // do transpose on input tensor, then do interpolation
        MLUCnnlTensorDesc input_desc(
            *input, CNNL_LAYOUT_NCHW, ToCnnlDataType(input->dtype()));

        transformed_input =
            ctx.AllocateTmpTensor<T, MLUDeviceContext>(dim_in_trans, dev_ctx);
        transformed_output =
            ctx.AllocateTmpTensor<T, MLUDeviceContext>(dim_out_trans, dev_ctx);

        MLUCnnlTensorDesc input_reshaped_desc(
            transformed_input,
            CNNL_LAYOUT_NHWC,
            ToCnnlDataType(transformed_input.dtype()));
        const std::vector<int> perm = {0, 2, 3, 1};
        MLUCnnl::Transpose(ctx,
                           perm,
                           input_dims.size(),
                           input_desc.get(),
                           GetBasePtr(input),
                           input_reshaped_desc.get(),
                           GetBasePtr(&transformed_input));
      } else {
        // if no need_transpose, do the following
        // 1. interpolation in(NHWC) -> out(NHWC)
        // dim_in = {n, in_h, in_w, c};
        dim_out = {n, out_h, out_w, c};
        output->mutable_data<T>(dim_out, ctx.GetPlace());

        if (in_h == out_h && in_w == out_w) {
          framework::TensorCopy(*input, ctx.GetPlace(), output);
          return;
        }
        transformed_input = *input;
        transformed_output = *output;
      }

      MLUCnnlTensorDesc input_desc(transformed_input,
                                   CNNL_LAYOUT_NHWC,
                                   ToCnnlDataType(transformed_input.dtype()));
      MLUCnnlTensorDesc output_desc(transformed_output,
                                    CNNL_LAYOUT_NHWC,
                                    ToCnnlDataType(transformed_output.dtype()));
      MLUCnnl::Interp(ctx,
                      GetMLUCnnlInterpMode(interp_method),
                      align_corners,
                      align_center,
                      input_desc.get(),
                      GetBasePtr(&transformed_input),
                      output_desc.get(),
                      GetBasePtr(&transformed_output));

      if (need_transpose) {
        // if need_transpose, reshape output back to NCHW
        const std::vector<int> perm = {0, 3, 1, 2};
        MLUCnnlTensorDesc output_reshape_desc(
            *output, CNNL_LAYOUT_NCHW, ToCnnlDataType(output->dtype()));
        MLUCnnl::Transpose(ctx,
                           perm,
                           dim_out_trans.size(),
                           output_desc.get(),
                           GetBasePtr(&transformed_output),
                           output_reshape_desc.get(),
                           GetBasePtr(output));
      }
    } else {
      PADDLE_ENFORCE_EQ(
          interp_method,
          "trilinear",
          platform::errors::External("MLU Interpolate kernel only supports 5D "
                                     "data in trilinear mode."));

      // need to do transpose if layout is kNCDHW
      need_transpose &= data_layout == DataLayout::kNCHW;
      if (need_transpose) {
        // if need_transpose, do the following
        // 1. transpose input NCDHW -> NDHWC
        // 2. interpolation in(NDHWC) -> out(NDHWC)
        // 3. transpose output NDHWC -> HCDHW
        // dim_in = {n, c, in_d, in_h, in_w};
        dim_in_trans = {n, in_d, in_h, in_w, c};
        dim_out = {n, c, out_d, out_h, out_w};
        dim_out_trans = {n, out_d, out_h, out_w, c};
        output->mutable_data<T>(dim_out, ctx.GetPlace());

        if (in_h == out_h && in_w == out_w && in_d == out_d) {
          framework::TensorCopy(*input, ctx.GetPlace(), output);
          return;
        }
        // do transpose on input tensor (HCDHW -> NDHWC), then do interpolation
        MLUCnnlTensorDesc input_desc(
            *input, CNNL_LAYOUT_NCDHW, ToCnnlDataType(input->dtype()));

        transformed_input =
            ctx.AllocateTmpTensor<T, MLUDeviceContext>(dim_in_trans, dev_ctx);
        transformed_output =
            ctx.AllocateTmpTensor<T, MLUDeviceContext>(dim_out_trans, dev_ctx);

        MLUCnnlTensorDesc input_reshaped_desc(
            transformed_input,
            CNNL_LAYOUT_NDHWC,
            ToCnnlDataType(transformed_input.dtype()));
        const std::vector<int> perm = {0, 2, 3, 4, 1};
        MLUCnnl::Transpose(ctx,
                           perm,
                           input_dims.size(),
                           input_desc.get(),
                           GetBasePtr(input),
                           input_reshaped_desc.get(),
                           GetBasePtr(&transformed_input));
      } else {
        // if no need_transpose, do the following
        // 1. interpolation in(NDHWC) -> out(NDHWC)
        // dim_in = {n, in_d, in_h, in_w, c};
        dim_out = {n, out_d, out_h, out_w, c};
        output->mutable_data<T>(dim_out, ctx.GetPlace());

        if (in_h == out_h && in_w == out_w && in_d == out_d) {
          framework::TensorCopy(*input, ctx.GetPlace(), output);
          return;
        }
        transformed_input = *input;
        transformed_output = *output;
      }

      MLUCnnlTensorDesc input_desc(transformed_input,
                                   CNNL_LAYOUT_NDHWC,
                                   ToCnnlDataType(transformed_input.dtype()));
      MLUCnnlTensorDesc output_desc(transformed_output,
                                    CNNL_LAYOUT_NDHWC,
                                    ToCnnlDataType(transformed_output.dtype()));
      // use trilinear mode in HCDHW layout
      MLUCnnl::Interp(ctx,
                      GetMLUCnnlInterpMode(interp_method),
                      align_corners,
                      align_center,
                      input_desc.get(),
                      GetBasePtr(&transformed_input),
                      output_desc.get(),
                      GetBasePtr(&transformed_output));

      if (need_transpose) {
        // if need_transpose, reshape output back (NDHWC -> NCDHW)
        const std::vector<int> perm = {0, 4, 1, 2, 3};
        MLUCnnlTensorDesc output_reshape_desc(
            *output, CNNL_LAYOUT_NCDHW, ToCnnlDataType(output->dtype()));
        MLUCnnl::Transpose(ctx,
                           perm,
                           dim_out_trans.size(),
                           output_desc.get(),
                           GetBasePtr(&transformed_output),
                           output_reshape_desc.get(),
                           GetBasePtr(output));
      }
    }
  }
};

template <typename T>
class InterpolateV2GradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<MLUDeviceContext>();
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto output_grad_dims = output_grad->dims();

    PADDLE_ENFORCE_EQ(output_grad_dims.size(),
                      4,
                      platform::errors::External(
                          "XPU Interpolategrad kernel only support 2d"));

    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto input_dims = input->dims();
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
    int n, c, in_d, in_h, in_w;
    ExtractNCDWH(input->dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

    auto interp_method = ctx.Attr<std::string>("interp_method");
    bool align_corners = ctx.Attr<bool>("align_corners");
    int align_mode = ctx.Attr<int>("align_mode");
    int align_center = align_corners ? 0 : (align_mode == 0 ? 0 : 1);
    align_center = 0;

    int out_h = ctx.Attr<int>("out_h");
    int out_w = ctx.Attr<int>("out_w");
    float scale_h = -1;
    float scale_w = -1;

    auto list_new_size_tensor = ctx.MultiInput<phi::DenseTensor>("SizeTensor");
    if (list_new_size_tensor.size() > 0) {
      // have size tensor
      auto new_size = get_new_shape_mlu(list_new_size_tensor);
      out_h = new_size[0];
      out_w = new_size[1];
    } else {
      auto scale_tensor = ctx.Input<phi::DenseTensor>("Scale");
      auto scale = ctx.Attr<std::vector<float>>("scale");
      if (scale_tensor != nullptr) {
        std::vector<float> scale_data;
        scale_data = GetDataFromTensor<float>(scale_tensor);
        if (scale_data.size() > 1) {
          scale_h = scale_data[0];
          scale_w = scale_data[1];
        } else {
          scale_h = scale_data[0];
          scale_w = scale_data[0];
        }
        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0,
            true,
            platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                              "should be greater than 0."));
      } else {
        if (scale.size() > 1) {
          scale_h = scale[0];
          scale_w = scale[1];

          PADDLE_ENFORCE_EQ(
              scale_w > 0 && scale_h > 0,
              true,
              platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                                "should be greater than 0."));
        }
      }
      if (scale_h > 0. && scale_w > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
      auto out_size = ctx.Input<phi::DenseTensor>("OutSize");
      if (out_size != nullptr) {
        std::vector<int32_t> out_size_data;
        out_size_data = GetDataFromTensor<int>(out_size);
        out_h = out_size_data[0];
        out_w = out_size_data[1];
      }
    }

    framework::DDim dim_grad;
    framework::DDim dim_out_grad, dim_out_trans_grad, dim_in_grad,
        dim_in_trans_grad;
    phi::DenseTensor transformed_output_grad, transformed_input_grad;
    bool need_transpose =
        input_dims.size() != 2 && data_layout == DataLayout::kNCHW;

    if (need_transpose) {
      // if need_transpose, do the following
      // 1. transpose output_grad NCHW -> NHWC
      // 2. InterpBackward output_grad(NHWC) -> input_grad(NHWC)
      // 3. transpose input_grad NHWC -> HCHW
      // dim_out_grad = {n, c, out_h, out_w};
      dim_out_trans_grad = {n, out_h, out_w, c};
      dim_in_grad = {n, c, in_h, in_w};
      dim_in_trans_grad = {n, in_h, in_w, c};
      input_grad->mutable_data<T>(dim_in_grad, ctx.GetPlace());

      if (in_h == out_h && in_w == out_w) {
        framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
        return;
      }
      // do transpose on input tensor, then do interpolation
      MLUCnnlTensorDesc input_desc(
          *output_grad, CNNL_LAYOUT_NCHW, ToCnnlDataType(output_grad->dtype()));

      transformed_output_grad = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          dim_out_trans_grad, dev_ctx);
      transformed_input_grad = ctx.AllocateTmpTensor<T, MLUDeviceContext>(
          dim_in_trans_grad, dev_ctx);

      MLUCnnlTensorDesc input_reshaped_desc(
          transformed_output_grad,
          CNNL_LAYOUT_NHWC,
          ToCnnlDataType(transformed_output_grad.dtype()));
      const std::vector<int> perm = {0, 2, 3, 1};
      MLUCnnl::Transpose(ctx,
                         perm,
                         input_dims.size(),
                         input_desc.get(),
                         GetBasePtr(output_grad),
                         input_reshaped_desc.get(),
                         GetBasePtr(&transformed_output_grad));
    } else {
      // if no need_transpose, do the following
      // 1. InterpBackward output_grad(NHWC) -> input_grad(NHWC)
      dim_in_grad = {n, in_h, in_w, c};
      input_grad->mutable_data<T>(dim_in_grad, ctx.GetPlace());

      if (in_h == out_h && in_w == out_w) {
        framework::TensorCopy(*output_grad, ctx.GetPlace(), input_grad);
        return;
      }
      transformed_output_grad = *output_grad;
      transformed_input_grad = *input_grad;
    }

    MLUCnnlTensorDesc input_desc(
        transformed_output_grad,
        CNNL_LAYOUT_NHWC,
        ToCnnlDataType(transformed_output_grad.dtype()));
    MLUCnnlTensorDesc output_desc(
        transformed_input_grad,
        CNNL_LAYOUT_NHWC,
        ToCnnlDataType(transformed_input_grad.dtype()));
    MLUCnnl::InterpBackward(ctx,
                            GetMLUCnnlInterpBackwardMode(interp_method),
                            align_corners,
                            align_center,
                            input_desc.get(),
                            GetBasePtr(&transformed_output_grad),
                            output_desc.get(),
                            GetBasePtr(&transformed_input_grad));

    if (need_transpose) {
      const std::vector<int> perm = {0, 3, 1, 2};
      MLUCnnlTensorDesc output_reshape_desc(
          *input_grad, CNNL_LAYOUT_NCHW, ToCnnlDataType(input_grad->dtype()));
      MLUCnnl::Transpose(ctx,
                         perm,
                         dim_in_trans_grad.size(),
                         output_desc.get(),
                         GetBasePtr(&transformed_input_grad),
                         output_reshape_desc.get(),
                         GetBasePtr(input_grad));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(bilinear_interp_v2,
                       ops::InterpolateV2MLUKernel<float>,
                       ops::InterpolateV2MLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(nearest_interp_v2,
                       ops::InterpolateV2MLUKernel<float>,
                       ops::InterpolateV2MLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(nearest_interp_v2_grad,
                       ops::InterpolateV2GradMLUKernel<float>,
                       ops::InterpolateV2GradMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(bilinear_interp_v2_grad,
                       ops::InterpolateV2GradMLUKernel<float>,
                       ops::InterpolateV2GradMLUKernel<plat::float16>);
