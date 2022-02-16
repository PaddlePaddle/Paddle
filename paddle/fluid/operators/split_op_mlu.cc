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

#include "paddle/fluid/operators/split_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SplitMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // init parameter
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int num = ctx.Attr<int>("num");
    std::vector<int> sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");
    auto in_dims = in->dims();
    auto out_size = outs.size();
    auto num_tensor = num == 0 ? out_size : num;

    bool need_resize_outs_dims = false;
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor(axis_tensor)[0];
      need_resize_outs_dims = true;
    }
    auto sections_tensor_list =
        ctx.MultiInput<framework::Tensor>("SectionsTensorList");
    if (sections_tensor_list.size() > 0) {
      sections = GetDataFromTensorList(sections_tensor_list);
      need_resize_outs_dims = true;
    }
    if (need_resize_outs_dims) {
      std::vector<framework::DDim> outs_dims =
          UpdateOutsDims(true, true, in_dims, num, sections, axis, out_size);
      for (size_t j = 0; j < outs.size(); ++j) {
        outs[j]->Resize(outs_dims[j]);
      }
    }

    // init out tensors
    std::vector<void*> vct_tensor;
    std::vector<MLUCnnlTensorDesc> output_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    for (size_t i = 0; i < outs.size(); i++) {
      outs[i]->mutable_data<T>(ctx.GetPlace());
      output_descs.emplace_back(MLUCnnlTensorDesc(
          *outs[i], CNNL_LAYOUT_ARRAY,
          ToCnnlDataType(framework::TransToProtoVarType(outs[i]->dtype()))));
      desc_vector.push_back(output_descs.back().get());
      vct_tensor.push_back(GetBasePtr(outs[i]));
    }
    // init in tensors
    MLUCnnlTensorDesc input_desc(
        *in, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(in->dtype())));

    // MLU should do sth
    MLUCnnl::Split(ctx, num_tensor, axis, input_desc.get(), GetBasePtr(in),
                   desc_vector.data(), vct_tensor.data());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(split, ops::SplitMLUKernel<float>,
                       ops::SplitMLUKernel<int64_t>, ops::SplitMLUKernel<int>,
                       ops::SplitMLUKernel<bool>,
                       ops::SplitMLUKernel<plat::float16>);
