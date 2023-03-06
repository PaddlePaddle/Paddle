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
#include "paddle/fluid/operators/conv_transpose_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

template <typename T>
class Conv2DTransposeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const phi::DenseTensor* input = ctx.Input<phi::DenseTensor>("Input");
    const phi::DenseTensor* filter = ctx.Input<phi::DenseTensor>("Filter");
    phi::DenseTensor* output = ctx.Output<phi::DenseTensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());
    std::vector<int> output_padding =
        ctx.Attr<std::vector<int>>("output_padding");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    int groups = ctx.Attr<int>("groups");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");

    // check dimension
    const bool channel_last = data_format == "NHWC";

    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();
    framework::DDim in_data_dims;
    framework::DDim filter_data_dims;

    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    filter_data_dims = phi::slice_ddim(filter_dims, 2, in_dims.size());

    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    phi::UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    phi::DenseTensor input_tensor(input->type());
    phi::DenseTensor output_tensor(output->type());
    input_tensor.set_layout(DataLayout::kNHWC);
    output_tensor.set_layout(DataLayout::kNHWC);
    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};

    if (channel_last) {
      input_tensor.ShareDataWith(*input);
      output_tensor.ShareDataWith(*output);
    } else {
      // transpose input from NCHW to NHWC
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nhwc,
                                input,
                                &input_tensor,
                                true /*need_reshape_or_alloc*/);
      auto output_dims = output->dims();
      output_tensor.mutable_data<T>(
          {output_dims[0], output_dims[2], output_dims[3], output_dims[1]},
          ctx.GetPlace());
    }

    // transpose filter from MCHW to MHWC
    phi::DenseTensor trans_filter(filter->type());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    // construct MLU attr
    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(
        input_tensor, data_layout, ToCnnlDataType(input_tensor.dtype()));
    MLUCnnlTensorDesc filter_desc(
        trans_filter, data_layout, ToCnnlDataType(trans_filter.type()));
    MLUCnnlTensorDesc output_desc(
        output_tensor, data_layout, ToCnnlDataType(output_tensor.dtype()));
    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     ToCnnlDataType<T>());

    MLUCnnl::ConvBackpropInput(ctx,
                               conv_desc.get(),
                               filter_desc.get(),
                               GetBasePtr(&trans_filter),
                               input_desc.get(),
                               GetBasePtr(&input_tensor),
                               output_desc.get(),
                               GetBasePtr(&output_tensor));

    if (!channel_last) {
      // transpose output from NHWC to NCHW
      const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nchw,
                                &output_tensor,
                                output,
                                false /*need_reshape_or_alloc*/);
    }
  }
};

template <typename T>
class Conv2DTransposeGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const phi::DenseTensor* input = ctx.Input<phi::DenseTensor>("Input");
    const phi::DenseTensor* filter = ctx.Input<phi::DenseTensor>("Filter");
    const phi::DenseTensor* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Output"));
    phi::DenseTensor* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    phi::DenseTensor* filter_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Filter"));

    if ((!input_grad) && (!filter_grad)) return;

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const int groups = ctx.Attr<int>("groups");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");
    const phi::DataLayout data_layout = phi::StringToDataLayout(data_format);

    auto in_dims = input->dims();
    auto filter_dims = filter->dims();
    auto in_dims_size = in_dims.size();

    const bool channel_last = (data_layout == phi::DataLayout::kNHWC);

    framework::DDim in_data_dims;
    if (channel_last) {
      in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
    } else {
      in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
    }
    framework::DDim filter_data_dims =
        phi::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
    phi::UpdatePaddingAndDilation(
        &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

    phi::DenseTensor input_tensor(input->type());
    phi::DenseTensor output_grad_tensor(output_grad->type());
    output_grad_tensor.set_layout(DataLayout::kNHWC);

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    if (channel_last) {
      input_tensor.ShareDataWith(*input);
      output_grad_tensor.ShareDataWith(*output_grad);
    } else {
      // transpose input from NCHW to NHWC
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nhwc,
                                input,
                                &input_tensor,
                                true /*need_reshape_or_alloc*/);
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nhwc,
                                output_grad,
                                &output_grad_tensor,
                                true /*need_reshape_or_alloc*/);
    }

    // transpose filter from MCHW to MHWC
    phi::DenseTensor trans_filter(filter->type());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    // MLU descs
    cnnlTensorLayout_t data_layout_mlu = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(
        input_tensor, data_layout_mlu, ToCnnlDataType(input_tensor.dtype()));
    MLUCnnlTensorDesc trans_filter_desc(
        trans_filter, data_layout_mlu, ToCnnlDataType(trans_filter.type()));
    MLUCnnlTensorDesc output_grad_desc(
        output_grad_tensor,
        data_layout_mlu,
        ToCnnlDataType(output_grad_tensor.dtype()));
    MLUCnnlConvolutionDesc conv_desc(in_dims_size,
                                     paddings.data(),
                                     strides.data(),
                                     dilations.data(),
                                     groups,
                                     ToCnnlDataType<T>());

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      phi::DenseTensor filter_grad_tensor(filter_grad->type());
      // filter_grad always MCHW
      // filter_grad_tensor always MHWC
      auto filter_grad_dims = filter_grad->dims();
      filter_grad_tensor.mutable_data<T>({filter_grad_dims[0],
                                          filter_grad_dims[2],
                                          filter_grad_dims[3],
                                          filter_grad_dims[1]},
                                         ctx.GetPlace());
      //}
      filter_grad_tensor.set_layout(DataLayout::kNHWC);

      MLUCnnlTensorDesc filter_grad_desc(
          filter_grad_tensor,
          data_layout_mlu,
          ToCnnlDataType(filter_grad_tensor.dtype()));

      MLUCnnl::ConvBackpropFilter(ctx,
                                  conv_desc.get(),
                                  output_grad_desc.get(),
                                  GetBasePtr(output_grad),
                                  input_desc.get(),
                                  GetBasePtr(&input_tensor),
                                  filter_grad_desc.get(),
                                  GetBasePtr(&filter_grad_tensor));
      // transpose output from MHWC to MCHW
      const std::vector<int> perm_to_mchw = {0, 3, 1, 2};
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_mchw,
                                &filter_grad_tensor,
                                filter_grad,
                                false /*need_reshape_or_alloc*/);
    }

    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      phi::DenseTensor input_grad_tensor(input_grad->type());
      input_tensor.set_layout(DataLayout::kNHWC);

      if (channel_last) {
        input_grad_tensor.ShareDataWith(*input_grad);
      } else {
        auto input_grad_dims = input_grad->dims();
        input_grad_tensor.mutable_data<T>({input_grad_dims[0],
                                           input_grad_dims[2],
                                           input_grad_dims[3],
                                           input_grad_dims[1]},
                                          ctx.GetPlace());
      }

      MLUCnnlTensorDesc input_grad_desc(
          input_grad_tensor,
          data_layout_mlu,
          ToCnnlDataType(input_grad_tensor.dtype()));

      MLUCnnl::ConvolutionForward(ctx,
                                  conv_desc.get(),
                                  nullptr /*alpha*/,
                                  nullptr /*beta*/,
                                  nullptr /*bias_desc*/,
                                  nullptr /*bias_ptr*/,
                                  output_grad_desc.get(),
                                  GetBasePtr(&output_grad_tensor),
                                  trans_filter_desc.get(),
                                  GetBasePtr(&trans_filter),
                                  input_grad_desc.get(),
                                  GetBasePtr(&input_grad_tensor));
      if (!channel_last) {
        // transpose output from NHWC to NCHW
        const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
        TransposeFromMLUTensor<T>(ctx,
                                  perm_to_nchw,
                                  &input_grad_tensor,
                                  input_grad,
                                  false /*need_reshape_or_alloc*/);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(conv2d_transpose,
                       ops::Conv2DTransposeMLUKernel<float>,
                       ops::Conv2DTransposeMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(conv2d_transpose_grad,
                       ops::Conv2DTransposeGradMLUKernel<float>,
                       ops::Conv2DTransposeGradMLUKernel<plat::float16>);
