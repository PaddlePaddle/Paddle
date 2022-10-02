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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class CastMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    auto src_type = static_cast<VT::Type>(ctx.Attr<int>("in_dtype"));
    auto dst_type = static_cast<VT::Type>(ctx.Attr<int>("out_dtype"));
    auto place = ctx.GetPlace();

    if (src_type == dst_type) {
      auto& dev_ctx = ctx.template device_context<platform::MLUDeviceContext>();
      output->mutable_data<T>(place);
      framework::TensorCopy(*input, place, dev_ctx, output);
      return;
    }

    PADDLE_ENFORCE_EQ(MLUSupportsCast(src_type, dst_type),
                      true,
                      platform::errors::InvalidArgument(
                          "MLU not support cast [%d] to [%d]",
                          framework::DataTypeToString(src_type),
                          framework::DataTypeToString(dst_type)));

    output->mutable_data(place, framework::TransToPhiDataType(dst_type));

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);
    cnnlCastDataType_t cast_type = GetCastDataType(src_type, dst_type);

    MLUCnnl::Cast(ctx,
                  cast_type,
                  input_desc.get(),
                  GetBasePtr(input),
                  output_desc.get(),
                  GetBasePtr(output));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(cast,
                       ops::CastMLUKernel<float>,
                       ops::CastMLUKernel<int>,
                       ops::CastMLUKernel<int16_t>,
                       ops::CastMLUKernel<uint8_t>,
                       ops::CastMLUKernel<bool>,
                       ops::CastMLUKernel<int64_t>,
                       ops::CastMLUKernel<paddle::platform::float16>);
