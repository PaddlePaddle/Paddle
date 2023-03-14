// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceMinMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<phi::DenseTensor>("X");
    auto* output = context.Output<phi::DenseTensor>("Out");
    int out_dtype = context.Attr<int>("out_dtype");
    bool reduce_all = context.Attr<bool>("reduce_all");
    auto dims = context.Attr<std::vector<int>>("dim");
    auto input_dims = input->dims();
    const auto& input_dim_size = input->dims().size();
    std::vector<int> reduce_dims;
    if (reduce_all) {
      for (int i = 0; i < input_dims.size(); i++) {
        reduce_dims.push_back(static_cast<int>(i));
      }
    } else {
      for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
          reduce_dims.push_back(dims[i] + input_dim_size);
        } else {
          reduce_dims.push_back(dims[i]);
        }
      }
    }

    auto place = context.GetPlace();
    phi::DenseTensor cast_out(input->type());
    cast_out.Resize(output->dims());
    cast_out.mutable_data<T>(place);

    auto cast_out_dtype = framework::TransToProtoVarType(input->dtype());

    if (out_dtype != -1) {
      cast_out_dtype = static_cast<framework::proto::VarType::Type>(out_dtype);
    }
    if (framework::TransToProtoVarType(input->type()) != cast_out_dtype) {
      if (cast_out_dtype == framework::proto::VarType::FP32) {
        output->mutable_data<float>(place);
      } else if (cast_out_dtype == framework::proto::VarType::FP16) {
        output->mutable_data<paddle::platform::float16>(place);
      } else if (cast_out_dtype == framework::proto::VarType::INT32) {
        output->mutable_data<int32_t>(place);
      }
    } else {
      output->ShareDataWith(cast_out);
    }

    MLUCnnlTensorDesc input_desc(
        *input, CNNL_LAYOUT_ARRAY, ToCnnlDataType(input->dtype()));
    MLUCnnlTensorDesc output_desc(
        *output, CNNL_LAYOUT_ARRAY, ToCnnlDataType(output->dtype()));

    MLUCnnlReduceDesc reduction_desc(reduce_dims,
                                     CNNL_REDUCE_MIN,
                                     ToCnnlDataType<T>(),
                                     CNNL_NOT_PROPAGATE_NAN,
                                     CNNL_REDUCE_NO_INDICES,
                                     CNNL_32BIT_INDICES);

    MLUCnnl::Reduce(context,
                    true /*need_workspace*/,
                    reduction_desc.get(),
                    nullptr,
                    input_desc.get(),
                    GetBasePtr(input),
                    0 /*indices_size*/,
                    nullptr,
                    nullptr,
                    output_desc.get(),
                    GetBasePtr(output));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(reduce_min,
                       ops::ReduceMinMLUKernel<float>,
                       ops::ReduceMinMLUKernel<plat::float16>,
                       ops::ReduceMinMLUKernel<int>);
