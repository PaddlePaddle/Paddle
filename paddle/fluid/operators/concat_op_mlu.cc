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

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ConcatMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");
    auto ins_size = ins.size();
    bool need_resize_out_dims = false;
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<phi::DenseTensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
      need_resize_out_dims = true;
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    if (need_resize_out_dims) {
      const size_t n = ins.size();
      std::vector<framework::DDim> ins_dims(n);
      for (size_t i = 0; i < n; i++) {
        ins_dims[i] = ins[i]->dims();
      }

      framework::DDim out_dims =
          phi::funcs::ComputeAndCheckShape(true, ins_dims, axis);
      out->Resize(out_dims);
    }
    const int axis_t = axis;
    const int ins_size_t = ins_size;
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    // mlu should do sth
    // init ins tensors
    std::vector<const void*> inputs;
    std::vector<MLUCnnlTensorDesc> input_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    for (size_t i = 0; i < ins_size; i++) {
      input_descs.emplace_back(MLUCnnlTensorDesc(
          *ins[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(ins[i]->dtype())));
      desc_vector.push_back(input_descs.back().get());
      inputs.push_back(GetBasePtr(ins[i]));
    }
    // init out tensors
    MLUCnnlTensorDesc output_desc(
        *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));

    // MLU should do sth
    MLUCnnl::Concat(ctx,
                    ins_size_t,
                    axis_t,
                    desc_vector.data(),
                    inputs.data(),
                    output_desc.get(),
                    GetBasePtr(out));
  }
};

template <typename T>
class ConcatGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<phi::DenseTensor>("X");
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    auto outs = ctx.MultiOutput<phi::DenseTensor>(framework::GradVarName("X"));
    auto axis = ctx.Attr<int>("axis");
    int split_num = ins.size();

    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));

    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<phi::DenseTensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
    }

    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    PADDLE_ENFORCE_GE(axis,
                      0,
                      platform::errors::InvalidArgument(
                          "concat_grad: axis should be larger than or "
                          "equal to 0, but received axis is %d.",
                          axis));
    PADDLE_ENFORCE_LT(
        axis,
        out_grad->dims().size(),
        platform::errors::InvalidArgument(
            "concat_grad: axis should be less than ins[0]->dims()!"
            "But received axis is %d, while ins[0]->dims()"
            "size is %d.",
            axis,
            out_grad->dims().size()));
    // get output tensor that the name is not kEmptyVarName
    std::vector<void*> outputs_vec;
    std::vector<phi::DenseTensor> tmp_outputs_vec;
    std::vector<MLUCnnlTensorDesc> output_descs;
    std::vector<cnnlTensorDescriptor_t> descs_vec;
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        output_descs.emplace_back(MLUCnnlTensorDesc(*outs[j]));
        outputs_vec.push_back(GetBasePtr(outs[j]));
      } else {
        phi::DenseTensor tmp_tensor;
        tmp_tensor.mutable_data<T>(ins[j]->dims(), ctx.GetPlace());
        tmp_outputs_vec.push_back(tmp_tensor);
        output_descs.emplace_back(MLUCnnlTensorDesc(*ins[j]));
        outputs_vec.push_back(GetBasePtr(&(tmp_outputs_vec.back())));
      }
      descs_vec.push_back(output_descs.back().get());
    }

    MLUCnnlTensorDesc out_grad_desc(*out_grad);
    MLUCnnl::Split(ctx,
                   static_cast<int>(split_num),
                   static_cast<int>(axis),
                   out_grad_desc.get(),
                   GetBasePtr(out_grad),
                   descs_vec.data(),
                   outputs_vec.data());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(concat,
                       ops::ConcatMLUKernel<float>,
                       ops::ConcatMLUKernel<paddle::platform::float16>,
                       ops::ConcatMLUKernel<int64_t>,
                       ops::ConcatMLUKernel<bool>,
                       ops::ConcatMLUKernel<int>,
                       ops::ConcatMLUKernel<uint8_t>);
REGISTER_OP_MLU_KERNEL(concat_grad,
                       ops::ConcatGradMLUKernel<float>,
                       ops::ConcatGradMLUKernel<paddle::platform::float16>,
                       ops::ConcatGradMLUKernel<int64_t>,
                       ops::ConcatGradMLUKernel<bool>,
                       ops::ConcatGradMLUKernel<int>,
                       ops::ConcatGradMLUKernel<uint8_t>);
