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
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");
    auto ins_size = ins.size();
    bool need_resize_out_dims = false;
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
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
    MLUCnnlTensorDesc output_desc(*out, CNNL_LAYOUT_ARRAY,
                                  ToCnnlDataType(out->dtype()));

    // MLU should do sth
    MLUCnnl::Concat(ctx, ins_size_t, axis_t, desc_vector.data(), inputs.data(),
                    output_desc.get(), GetBasePtr(out));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(concat, ops::ConcatMLUKernel<float>,
                       ops::ConcatMLUKernel<paddle::platform::float16>,
                       ops::ConcatMLUKernel<int64_t>,
                       ops::ConcatMLUKernel<bool>, ops::ConcatMLUKernel<int>,
                       ops::ConcatMLUKernel<uint8_t>);
