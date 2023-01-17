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
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class DeformableConvMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* offset = ctx.Input<phi::DenseTensor>("Offset");
    auto* mask = ctx.Input<phi::DenseTensor>("Mask");
    auto* filter = ctx.Input<phi::DenseTensor>("Filter");
    auto* output = ctx.Output<phi::DenseTensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    // TODO(fwg): Remove this check when cnnl fix the bug that groups > 1.
    PADDLE_ENFORCE_EQ(
        groups == 1,
        true,
        platform::errors::InvalidArgument(
            "MLU deformable_conv kernel only support groups == 1, but get %d.",
            groups));

    // transform paddings from {h, w} to {top, bottom, left, right}.
    const std::vector<int> trans_paddings{
        paddings[0], paddings[0], paddings[1], paddings[1]};
    MLUCnnlDCNDesc dcn_desc(input->dims().size(),
                            trans_paddings.data(),
                            strides.data(),
                            dilations.data(),
                            deformable_groups,
                            groups,
                            im2col_step);

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    Tensor trans_input(input->dtype());
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, input, &trans_input, true /*need_reshape_or_alloc*/);

    Tensor trans_offset(offset->dtype());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              offset,
                              &trans_offset,
                              true /*need_reshape_or_alloc*/);

    Tensor trans_mask(mask->dtype());
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, mask, &trans_mask, true /*need_reshape_or_alloc*/);

    Tensor trans_filter(filter->dtype());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    Tensor tmp_output(output->dtype());
    auto output_dims = output->dims();
    tmp_output.mutable_data<T>(
        {output_dims[0], output_dims[2], output_dims[3], output_dims[1]},
        ctx.GetPlace());

    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc input_desc(
        trans_input, data_layout, ToCnnlDataType(trans_input.dtype()));
    MLUCnnlTensorDesc offset_desc(
        trans_offset, data_layout, ToCnnlDataType(trans_offset.dtype()));
    MLUCnnlTensorDesc mask_desc(
        trans_mask, data_layout, ToCnnlDataType(trans_mask.dtype()));
    MLUCnnlTensorDesc filter_desc(
        trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));
    MLUCnnlTensorDesc output_desc(
        tmp_output, data_layout, ToCnnlDataType(tmp_output.dtype()));
    MLUCnnl::DCNForward(ctx,
                        dcn_desc.get(),
                        input_desc.get(),
                        GetBasePtr(&trans_input),
                        offset_desc.get(),
                        GetBasePtr(&trans_offset),
                        mask_desc.get(),
                        GetBasePtr(&trans_mask),
                        filter_desc.get(),
                        GetBasePtr(&trans_filter),
                        nullptr,
                        nullptr,
                        output_desc.get(),
                        GetBasePtr(&tmp_output));

    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nchw,
                              &tmp_output,
                              output,
                              false /*need_reshape_or_alloc*/);
  }
};

template <typename T>
class DeformableConvGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const phi::DenseTensor* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Output"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    auto* filter_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Filter"));
    auto* offset_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Offset"));
    auto* mask_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Mask"));

    const phi::DenseTensor* input = ctx.Input<phi::DenseTensor>("Input");
    auto* offset = ctx.Input<phi::DenseTensor>("Offset");
    auto* mask = ctx.Input<phi::DenseTensor>("Mask");
    auto* filter = ctx.Input<phi::DenseTensor>("Filter");

    int groups = ctx.Attr<int>("groups");
    int deformable_groups = ctx.Attr<int>("deformable_groups");
    int im2col_step = ctx.Attr<int>("im2col_step");
    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    // TODO(fwg): Remove this check when cnnl fix the bug that groups > 1.
    PADDLE_ENFORCE_EQ(groups == 1,
                      true,
                      platform::errors::InvalidArgument(
                          "MLU deformable_conv_grad kernel only support groups "
                          "== 1, but get %d.",
                          groups));

    // transform paddings from {h, w} to {top, bottom, left, right}.
    const std::vector<int> trans_paddings{
        paddings[0], paddings[0], paddings[1], paddings[1]};
    MLUCnnlDCNDesc dcn_desc(input->dims().size(),
                            trans_paddings.data(),
                            strides.data(),
                            dilations.data(),
                            deformable_groups,
                            groups,
                            im2col_step);

    Tensor tmp_input_grad;
    auto input_dims = input->dims();
    tmp_input_grad.mutable_data<T>(
        {input_dims[0], input_dims[2], input_dims[3], input_dims[1]},
        ctx.GetPlace());

    Tensor tmp_filter_grad;
    auto filter_dims = filter->dims();
    tmp_filter_grad.mutable_data<T>(
        {filter_dims[0], filter_dims[2], filter_dims[3], filter_dims[1]},
        ctx.GetPlace());

    Tensor tmp_offset_grad;
    auto offset_dims = offset->dims();
    tmp_offset_grad.mutable_data<T>(
        {offset_dims[0], offset_dims[2], offset_dims[3], offset_dims[1]},
        ctx.GetPlace());

    Tensor tmp_mask_grad;
    auto mask_dims = mask->dims();
    tmp_mask_grad.mutable_data<T>(
        {mask_dims[0], mask_dims[2], mask_dims[3], mask_dims[1]},
        ctx.GetPlace());

    const std::vector<int> perm_to_nhwc = {0, 2, 3, 1};
    Tensor trans_output_grad(output_grad->dtype());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              output_grad,
                              &trans_output_grad,
                              true /*need_reshape_or_alloc*/);

    Tensor trans_input(input->dtype());
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, input, &trans_input, true /*need_reshape_or_alloc*/);

    Tensor trans_offset(offset->dtype());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              offset,
                              &trans_offset,
                              true /*need_reshape_or_alloc*/);

    Tensor trans_mask(mask->dtype());
    TransposeFromMLUTensor<T>(
        ctx, perm_to_nhwc, mask, &trans_mask, true /*need_reshape_or_alloc*/);

    Tensor trans_filter(filter->dtype());
    TransposeFromMLUTensor<T>(ctx,
                              perm_to_nhwc,
                              filter,
                              &trans_filter,
                              true /*need_reshape_or_alloc*/);

    cnnlTensorLayout_t data_layout = CNNL_LAYOUT_NHWC;
    MLUCnnlTensorDesc output_grad_desc(
        trans_output_grad,
        data_layout,
        ToCnnlDataType(trans_output_grad.dtype()));
    MLUCnnlTensorDesc input_desc(
        trans_input, data_layout, ToCnnlDataType(trans_input.dtype()));
    MLUCnnlTensorDesc offset_desc(
        trans_offset, data_layout, ToCnnlDataType(trans_offset.dtype()));
    MLUCnnlTensorDesc mask_desc(
        trans_mask, data_layout, ToCnnlDataType(trans_mask.dtype()));
    MLUCnnlTensorDesc filter_desc(
        trans_filter, data_layout, ToCnnlDataType(trans_filter.dtype()));

    MLUCnnl::DCNBackwardData(ctx,
                             dcn_desc.get(),
                             input_desc.get(),
                             GetBasePtr(&trans_input),
                             offset_desc.get(),
                             GetBasePtr(&trans_offset),
                             mask_desc.get(),
                             GetBasePtr(&trans_mask),
                             filter_desc.get(),
                             GetBasePtr(&trans_filter),
                             output_grad_desc.get(),
                             GetBasePtr(&trans_output_grad),
                             input_desc.get(),
                             GetBasePtr(&tmp_input_grad),
                             offset_desc.get(),
                             GetBasePtr(&tmp_offset_grad),
                             mask_desc.get(),
                             GetBasePtr(&tmp_mask_grad));

    MLUCnnl::DCNBackwardWeight(ctx,
                               dcn_desc.get(),
                               input_desc.get(),
                               GetBasePtr(&trans_input),
                               offset_desc.get(),
                               GetBasePtr(&trans_offset),
                               mask_desc.get(),
                               GetBasePtr(&trans_mask),
                               output_grad_desc.get(),
                               GetBasePtr(&trans_output_grad),
                               filter_desc.get(),
                               GetBasePtr(&tmp_filter_grad),
                               nullptr,
                               nullptr);

    const std::vector<int> perm_to_nchw = {0, 3, 1, 2};
    if (input_grad) {
      input_grad->mutable_data<T>(ctx.GetPlace());
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nchw,
                                &tmp_input_grad,
                                input_grad,
                                false /*need_reshape_or_alloc*/);
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(ctx.GetPlace());
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nchw,
                                &tmp_filter_grad,
                                filter_grad,
                                false /*need_reshape_or_alloc*/);
    }

    if (offset_grad) {
      offset_grad->mutable_data<T>(ctx.GetPlace());
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nchw,
                                &tmp_offset_grad,
                                offset_grad,
                                false /*need_reshape_or_alloc*/);
    }

    if (mask_grad) {
      mask_grad->mutable_data<T>(ctx.GetPlace());
      TransposeFromMLUTensor<T>(ctx,
                                perm_to_nchw,
                                &tmp_mask_grad,
                                mask_grad,
                                false /*need_reshape_or_alloc*/);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(deformable_conv, ops::DeformableConvMLUKernel<float>);
REGISTER_OP_MLU_KERNEL(deformable_conv_grad,
                       ops::DeformableConvGradMLUKernel<float>);
