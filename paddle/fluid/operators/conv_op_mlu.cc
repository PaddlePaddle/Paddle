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

    PADDLE_ENFORCE_EQ(
        data_format == "NCHW", false,
        platform::errors::InvalidArgument(
            ("MLU only support data_format is NHWC in conv op, but now %s",
             data_format)));

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();
    auto filter_dims_size = filter_dims.size();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    Tensor input_tensor, output_tensor;
    input_tensor.ShareDataWith(*input);
    output_tensor.ShareDataWith(*output);
    input_tensor.set_layout(DataLayout::kNHWC);
    output_tensor.set_layout(DataLayout::kNHWC);

    // transpose filter from MCHW to MHWC
    Tensor trans_filter(filter->type());
    const std::vector<int> trans_perm = {0, 2, 3, 1};
    trans_filter.mutable_data<T>(
        {
            filter->dims()[trans_perm[0]], filter->dims()[trans_perm[1]],
            filter->dims()[trans_perm[2]], filter->dims()[trans_perm[3]],
        },
        ctx.GetPlace());

    MLUCnnlTensorDesc in_filter_desc(*filter, CNNL_LAYOUT_ARRAY,
                                     ToCnnlDataType(filter->type()));
    MLUCnnlTensorDesc out_filter_desc(trans_filter, CNNL_LAYOUT_ARRAY,
                                      ToCnnlDataType(trans_filter.type()));

    MLUCnnl::Transpose(ctx, trans_perm, filter_dims_size, in_filter_desc.get(),
                       GetBasePtr(filter), out_filter_desc.get(),
                       GetBasePtr(&trans_filter));

    MLUCnnlConvolutionDesc conv_desc(in_dims_size, paddings.data(),
                                     strides.data(), dilations.data(), groups,
                                     ToCnnlDataType<T>());
    MLUCnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_NHWC,
                                 ToCnnlDataType(input_tensor.type()));
    MLUCnnlTensorDesc filter_desc(trans_filter, CNNL_LAYOUT_NHWC,
                                  ToCnnlDataType(trans_filter.type()));
    MLUCnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_NHWC,
                                  ToCnnlDataType(output_tensor.type()));

    MLUCnnl::ConvolutionForward(
        ctx, conv_desc.get(), nullptr /*alpha*/, nullptr /*beta*/,
        nullptr /*bias_desc*/, nullptr /*bias_ptr*/, input_desc.get(),
        GetBasePtr(&input_tensor), filter_desc.get(), GetBasePtr(&trans_filter),
        output_desc.get(), GetBasePtr(&output_tensor));
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
    PADDLE_ENFORCE_EQ(
        data_format == "NCHW", false,
        platform::errors::InvalidArgument(
            ("MLU only support data_format is NHWC in conv op, but now %s",
             data_format)));

    // update padding and dilation
    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();
    auto filter_dims_size = filter_dims.size();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    in_data_dims = framework::slice_ddim(in_dims, 1, in_dims.size() - 1);
    filter_data_dims = framework::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    Tensor input_tensor, output_grad_tensor;
    input_tensor.ShareDataWith(*input);
    output_grad_tensor.ShareDataWith(*output_grad);
    input_tensor.set_layout(DataLayout::kNHWC);
    output_grad_tensor.set_layout(DataLayout::kNHWC);

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());

      Tensor temp_filter_grad(filter_grad->type());
      const std::vector<int> trans_perm = {0, 2, 3, 1};
      temp_filter_grad.mutable_data<T>(
          {
              filter->dims()[trans_perm[0]], filter->dims()[trans_perm[1]],
              filter->dims()[trans_perm[2]], filter->dims()[trans_perm[3]],
          },
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

      // transpose filter from MHWC to MCHW
      MLUCnnlTensorDesc trans_filter_desc(temp_filter_grad, CNNL_LAYOUT_ARRAY,
                                          tensor_dtype);
      MLUCnnlTensorDesc filter_grad_desc(*filter_grad, CNNL_LAYOUT_ARRAY,
                                         tensor_dtype);

      MLUCnnl::Transpose(ctx, {0, 3, 1, 2} /*trans_perm*/, filter_dims_size,
                         trans_filter_desc.get(), GetBasePtr(&temp_filter_grad),
                         filter_grad_desc.get(), GetBasePtr(filter_grad));
    }
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());

      Tensor input_grad_tensor;
      input_grad_tensor.ShareDataWith(*input_grad);
      input_grad_tensor.set_layout(DataLayout::kNHWC);

      // transpose filter from MCHW to MHWC
      Tensor trans_filter(filter->type());
      const std::vector<int> trans_perm = {0, 2, 3, 1};
      trans_filter.mutable_data<T>(
          {
              filter->dims()[trans_perm[0]], filter->dims()[trans_perm[1]],
              filter->dims()[trans_perm[2]], filter->dims()[trans_perm[3]],
          },
          ctx.GetPlace());

      cnnlDataType_t tensor_dtype = ToCnnlDataType<T>();
      MLUCnnlTensorDesc in_filter_desc(*filter, CNNL_LAYOUT_ARRAY,
                                       tensor_dtype);
      MLUCnnlTensorDesc trans_filter_desc(trans_filter, CNNL_LAYOUT_ARRAY,
                                          tensor_dtype);

      MLUCnnl::Transpose(ctx, trans_perm, filter_dims_size,
                         in_filter_desc.get(), GetBasePtr(filter),
                         trans_filter_desc.get(), GetBasePtr(&trans_filter));

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
