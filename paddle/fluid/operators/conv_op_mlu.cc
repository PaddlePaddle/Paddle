// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DataLayout = framework::DataLayout;

template <typename T>
class MLUConvOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("Input");
    auto* filter = ctx.Input<Tensor>("Filter");
    auto* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    Tensor input_tensor(input->type());
    Tensor output_tensor(output->type());
    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    if (channel_last) {
      input_tensor.ShareDataWith(*input);
      output_tensor.ShareDataWith(*output);
    } else {
      // transpose input from NCHW to NHWC
      TransposeFromMLUTensor<T>(ctx, perm_to_nhwc, input, &input_tensor,
                                true /*need_reshape_or_alloc*/);
      auto output_dims = output->dims();
      output_tensor.mutable_data<T>(
          {output_dims[0], output_dims[2], output_dims[3], output_dims[1]},
          ctx.GetPlace());
    }
    input_tensor.set_layout(DataLayout::kNHWC);
    output_tensor.set_layout(DataLayout::kNHWC);

    // transpose filter from MCHW to MHWC
    Tensor trans_filter(filter->type());
    TransposeFromMLUTensor<T>(ctx, perm_to_nhwc, filter, &trans_filter,
                              true /*need_reshape_or_alloc*/);

    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(
        input_tensor, data_layout,
        ToCnnlDataType(framework::TransToProtoVarType(input_tensor.dtype())));
    MLUCnnlTensorDesc filter_desc(trans_filter, data_layout,
                                  ToCnnlDataType(trans_filter.type()));
    MLUCnnlTensorDesc output_desc(
        output_tensor, data_layout,
        ToCnnlDataType(framework::TransToProtoVarType(output_tensor.dtype())));

    MLUCnnlConvolutionDesc conv_desc(in_dims_size, paddings.data(),
                                     strides.data(), dilations.data(), groups,
                                     ToCnnlDataType<T>());

    MLUCnnl::ConvolutionForward(
        ctx, conv_desc.get(), nullptr /*alpha*/, nullptr /*beta*/,
        nullptr /*bias_desc*/, nullptr /*bias_ptr*/, input_desc.get(),
        GetBasePtr(&input_tensor), filter_desc.get(), GetBasePtr(&trans_filter),
        output_desc.get(), GetBasePtr(&output_tensor));

    if (!channel_last) {
      // transpose ouput from NHWC to NCHW
      const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
      TransposeFromMLUTensor<T>(ctx, perm_to_nchw, &output_tensor, output,
                                false /*need_reshape_or_alloc*/);
    }
  }
};

template <typename T>
class MLUConvGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<Tensor>("Input");
    auto filter = ctx.Input<Tensor>("Filter");
    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = data_format == "NHWC";

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    Tensor input_tensor(input->type());
    Tensor output_grad_tensor(output_grad->type());
    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    if (channel_last) {
      input_tensor.ShareDataWith(*input);
      output_grad_tensor.ShareDataWith(*output_grad);
    } else {
      // transpose input and output_grad from NCHW to NHWC
      TransposeFromMLUTensor<T>(ctx, perm_to_nhwc, input, &input_tensor,
                                true /*need_reshape_or_alloc*/);
      TransposeFromMLUTensor<T>(ctx, perm_to_nhwc, output_grad,
                                &output_grad_tensor,
                                true /*need_reshape_or_alloc*/);
    }
    input_tensor.set_layout(DataLayout::kNHWC);
    output_grad_tensor.set_layout(DataLayout::kNHWC);

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());

      auto filter_grad_dims = filter_grad->dims();
      Tensor temp_filter_grad(filter_grad->type());
      temp_filter_grad.mutable_data<T>(
          {filter_grad_dims[0], filter_grad_dims[2], filter_grad_dims[3],
           filter_grad_dims[1]},
          ctx.GetPlace());

      cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
      cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
      MLUCnnlTensorDesc input_desc(input_tensor, data_layout, tensor_dtype);
      MLUCnnlTensorDesc out_grad_desc(output_grad_tensor, data_layout,
                                      tensor_dtype);
      MLUCnnlTensorDesc temp_filter_grad_desc(temp_filter_grad, data_layout,
                                              tensor_dtype);

      MLUCnnlConvolutionDesc conv_desc(in_dims_size, paddings.data(),
                                       strides.data(), dilations.data(), groups,
                                       tensor_dtype);

      MLUCnnl::ConvBackpropFilter(
          ctx, conv_desc.get(), input_desc.get(), GetBasePtr(&input_tensor),
          out_grad_desc.get(), GetBasePtr(&output_grad_tensor),
          temp_filter_grad_desc.get(), GetBasePtr(&temp_filter_grad));

      // transpose filter_grad from MHWC to MCHW
      TransposeFromMLUTensor<T>(ctx, perm_to_nchw, &temp_filter_grad,
                                filter_grad, false /*need_reshape_or_alloc*/);
    }
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());

      Tensor input_grad_tensor(input_grad->type());
      if (channel_last) {
        input_grad_tensor.ShareDataWith(*input_grad);
      } else {
        auto input_grad_dims = input_grad->dims();
        input_grad_tensor.mutable_data<T>(
            {input_grad_dims[0], input_grad_dims[2], input_grad_dims[3],
             input_grad_dims[1]},
            ctx.GetPlace());
      }
      input_grad_tensor.set_layout(DataLayout::kNHWC);

      // transpose filter from MCHW to MHWC
      Tensor trans_filter(filter->type());
      TransposeFromMLUTensor<T>(ctx, perm_to_nhwc, filter, &trans_filter,
                                true /*need_reshape_or_alloc*/);

      cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
      cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
      MLUCnnlTensorDesc filter_desc(trans_filter, data_layout, tensor_dtype);
      MLUCnnlTensorDesc out_grad_desc(output_grad_tensor, data_layout,
                                      tensor_dtype);
      MLUCnnlTensorDesc in_grad_desc(input_grad_tensor, data_layout,
                                     tensor_dtype);

      MLUCnnlConvolutionDesc conv_desc(in_dims_size, paddings.data(),
                                       strides.data(), dilations.data(), groups,
                                       tensor_dtype);

      MLUCnnl::ConvBackpropInput(
          ctx, conv_desc.get(), filter_desc.get(), GetBasePtr(&trans_filter),
          out_grad_desc.get(), GetBasePtr(&output_grad_tensor),
          in_grad_desc.get(), GetBasePtr(&input_grad_tensor));

      if (!channel_last) {
        // transpose input_grad from NHWC to NCHW
        TransposeFromMLUTensor<T>(ctx, perm_to_nchw, &input_grad_tensor,
                                  input_grad, false /*need_reshape_or_alloc*/);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(conv2d, ops::MLUConvOpKernel<float>,
                       ops::MLUConvOpKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(conv2d_grad, ops::MLUConvGradOpKernel<float>,
                       ops::MLUConvGradOpKernel<plat::float16>);
