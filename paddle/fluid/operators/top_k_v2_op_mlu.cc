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
class TopkV2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("X");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto* indices = ctx.Output<framework::LoDTensor>("Indices");
    const auto& place = ctx.GetPlace();

    const auto& sorted = static_cast<bool>(ctx.Attr<bool>("sorted"));
    const auto& largest = static_cast<bool>(ctx.Attr<bool>("largest"));

    // axis < 0, cacluate the real axis
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    if (axis < 0) {
      const auto& in_dims = input->dims();
      axis += in_dims.size();
    }

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      auto k_t_ptr = static_cast<const void*>(k_t->data<int>());
      auto size = k_t->numel() * sizeof(int);
      memory::Copy(platform::CPUPlace(), reinterpret_cast<void*>(&k),
                   k_t->place(), k_t_ptr, size, nullptr);
      framework::DDim output_dims = output->dims();
      // accroding to axis to set K value in the dim
      output_dims[axis] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    output->mutable_data<T>(place);
    indices->mutable_data<int64_t>(place);

    // cnnl only support int32/int16 type of indices
    framework::Tensor indices_int32(framework::TransToPhiDataType(VT::INT32));
    indices_int32.Resize(indices->dims());
    indices_int32.mutable_data<int32_t>(place);

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc values_output_desc(*output);
    MLUCnnlTensorDesc indices_int32_desc(indices_int32);
    MLUCnnl::TopK(ctx, k, axis, largest, sorted, input_desc.get(),
                  GetBasePtr(input), values_output_desc.get(),
                  GetBasePtr(output), indices_int32_desc.get(),
                  GetBasePtr(&indices_int32));

    // cast indices type to int64
    MLUCnnlTensorDesc cast_output_desc(*indices);
    cnnlCastDataType_t cast_type = GetCastDataType(VT::INT32, VT::INT64);
    MLUCnnl::Cast(ctx, cast_type, indices_int32_desc.get(),
                  GetBasePtr(&indices_int32), cast_output_desc.get(),
                  GetBasePtr(indices));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(top_k_v2, ops::TopkV2MLUKernel<float>,
                       ops::TopkV2MLUKernel<paddle::platform::float16>);
