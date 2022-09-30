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

template <typename T>
class ArgsortMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");
    const auto& place = ctx.GetPlace();

    const auto& sorted = true;
    const bool descending = ctx.Attr<bool>("descending");

    // axis < 0, cacluate the real axis
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    if (axis < 0) {
      const auto& in_dims = input->dims();
      axis += in_dims.size();
    }

    auto in_dims = input->dims();
    size_t k = in_dims[axis];

    output->mutable_data<T>(place);
    indices->mutable_data<int64_t>(place);

    // cnnl only support int32/int16 type of indices
    phi::DenseTensor indices_int32(framework::TransToPhiDataType(VT::INT32));
    indices_int32.Resize(indices->dims());
    indices_int32.mutable_data<int32_t>(place);

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc values_output_desc(*output);
    MLUCnnlTensorDesc indices_int32_desc(indices_int32);
    MLUCnnl::TopK(ctx,
                  k,
                  axis,
                  descending,
                  sorted,
                  input_desc.get(),
                  GetBasePtr(input),
                  values_output_desc.get(),
                  GetBasePtr(output),
                  indices_int32_desc.get(),
                  GetBasePtr(&indices_int32));

    // cast indices type to int64
    MLUCnnlTensorDesc cast_output_desc(*indices);
    cnnlCastDataType_t cast_type = GetCastDataType(VT::INT32, VT::INT64);
    MLUCnnl::Cast(ctx,
                  cast_type,
                  indices_int32_desc.get(),
                  GetBasePtr(&indices_int32),
                  cast_output_desc.get(),
                  GetBasePtr(indices));
  }
};

template <typename T>
class ArgsortGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* indices = ctx.Input<phi::DenseTensor>("Indices");
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    int axis = ctx.Attr<int>("axis");
    dx->mutable_data<T>(ctx.GetPlace());

    auto in_dims = indices->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    if (dout->numel() == 0) return;

    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc indices_desc(*indices);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnl::ScatterFunctor(ctx,
                            dx_desc.get(),
                            GetBasePtr(dx),
                            dout_desc.get(),
                            GetBasePtr(dout),
                            indices_desc.get(),
                            GetBasePtr(indices),
                            axis);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(argsort,
                       ops::ArgsortMLUKernel<paddle::platform::float16>,
                       ops::ArgsortMLUKernel<float>,
                       ops::ArgsortMLUKernel<int8_t>,
                       ops::ArgsortMLUKernel<uint8_t>,
                       ops::ArgsortMLUKernel<int16_t>,
                       ops::ArgsortMLUKernel<int>);

REGISTER_OP_MLU_KERNEL(argsort_grad,
                       ops::ArgsortGradMLUKernel<paddle::platform::float16>,
                       ops::ArgsortGradMLUKernel<float>,
                       ops::ArgsortGradMLUKernel<int8_t>,
                       ops::ArgsortGradMLUKernel<uint8_t>,
                       ops::ArgsortGradMLUKernel<int16_t>,
                       ops::ArgsortGradMLUKernel<int>);
