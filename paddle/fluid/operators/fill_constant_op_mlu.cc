/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class FillConstantMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
   
    auto str_value = ctx.Attr<std::string>("str_value");
    auto float_value = ctx.Attr<float>("value");

    auto *out_var = ctx.Output<framework::Tensor>("Out");

    T value;
    if (str_value.empty()) {
      value = static_cast<T>(float_value);
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = static_cast<T>(std::numeric_limits<double>::infinity());
      } else if (str_value == "-inf") {
        value = static_cast<T>(-std::numeric_limits<double>::infinity());
      } else if (str_value == "nan") {
        value = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
      } else {
        std::stringstream convert_stream(str_value);
        if (std::is_same<int64_t, T>::value) {
          int64_t tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        } else {
          double tmp_value;
          convert_stream >> tmp_value;
          value = static_cast<T>(tmp_value);
        }
      }
    }

    auto shape = GetShape(ctx);

    out_var->mutable_data<T>(shape, ctx.GetPlace());
    //out_var->mutable_data<T>(ctx.GetPlace());
    //MLU should do sth
    MLUCnnlTensorDesc output_desc(*out_var, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(out_var->type()));
    MLUCnnl::Fill(ctx, value,output_desc.get(),reinterpret_cast<void*>(out_var->data<T>()));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    fill_constant, paddle::operators::FillConstantMLUKernel<float>,
    paddle::operators::FillConstantMLUKernel<bool>,
    paddle::operators::FillConstantMLUKernel<int>);

