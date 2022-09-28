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

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using SelectedRows = phi::SelectedRows;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SumMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      // init
      auto *out = out_var->GetMutable<framework::LoDTensor>();
      auto ins = ctx.MultiInput<phi::DenseTensor>("X");
      out->mutable_data<T>(ctx.GetPlace());
      auto place = ctx.GetPlace();
      int ins_size = static_cast<int>(ins.size());
      if (ins_size == 1) {
        framework::TensorCopy(*ins[0], place, out);
        return;
      }

      // MLU shoul do sth
      std::vector<const void *> inputs;
      std::vector<MLUCnnlTensorDesc> input_descs;
      std::vector<cnnlTensorDescriptor_t> desc_vector;
      for (int i = 0; i < ins_size; i++) {
        input_descs.emplace_back(MLUCnnlTensorDesc(
            *ins[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(ins[i]->dtype())));
        desc_vector.push_back(input_descs.back().get());
        inputs.push_back(GetBasePtr(ins[i]));
      }
      // init out tensors
      MLUCnnlTensorDesc output_desc(
          *out, CNNL_LAYOUT_ARRAY, ToCnnlDataType(out->dtype()));
      uint32_t ins_size_t = static_cast<uint32_t>(ins_size);
      MLUCnnl::AddN(ctx,
                    ins_size_t,
                    desc_vector.data(),
                    inputs.data(),
                    output_desc.get(),
                    GetBasePtr(out));

    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) must be Tensor or But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    sum,
    ops::SumMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::SumMLUKernel<paddle::platform::MLUDeviceContext,
                      paddle::platform::float16>);
