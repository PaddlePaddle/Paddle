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

#include "paddle/fluid/operators/reduce_ops/reduce_mean_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

template <typename T>
class ReduceMeanMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<T>(context.GetPlace());

    bool reduce_all = context.Attr<bool>("reduce_all");
    auto dims = context.Attr<std::vector<int>>("dim");
    auto input_dims = framework::vectorize(input->dims());
    const auto& input_dim_size = input->dims().size();
    std::vector<int> reduce_dims;
    if (reduce_all) {
      for (size_t i = 0; i < input_dims.size(); i++) {
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

    MLUCnnlTensorDesc input_desc(
        *input, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(input->dtype())));
    MLUCnnlTensorDesc output_desc(
        *output, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(output->dtype())));

    MLUCnnlReduceDesc reduction_desc(
        reduce_dims, CNNL_REDUCE_AVG, ToCnnlDataType<T>(),
        CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

    MLUCnnl::Reduce(context, true /*need_workspace*/, reduction_desc.get(),
                    nullptr, input_desc.get(), GetBasePtr(input),
                    0 /*indices_size*/, nullptr, nullptr, output_desc.get(),
                    GetBasePtr(output));
  }
};

template <typename T>
class ReduceMeanGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = context.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(context.GetPlace());

    bool reduce_all = context.Attr<bool>("reduce_all");
    auto reduce_dims = context.Attr<std::vector<int>>("dim");
    auto input_dims = framework::vectorize(input->dims());

    int reduce_numel = 1;
    if (reduce_all) {
      reduce_dims.clear();
      for (size_t d = 0; d < input_dims.size(); ++d) {
        reduce_dims.push_back(static_cast<int>(d));
      }
    }
    for (auto& d : reduce_dims) {
      if (d < 0) {
        d = d + input_dims.size();
      }
      reduce_numel *= input_dims[d];
    }

    Tensor tmp_output_grad(
        framework::TransToProtoVarType(output_grad->dtype()));
    auto tmp_output_dims = input_dims;
    for (auto d : reduce_dims) {
      tmp_output_dims[d] = 1;
    }
    tmp_output_grad.ShareDataWith(*output_grad);
    tmp_output_grad.Resize(framework::make_ddim(tmp_output_dims));

    MLUCnnlTensorDesc output_grad_desc(
        tmp_output_grad, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(
            framework::TransToProtoVarType(tmp_output_grad.dtype())));
    MLUCnnlTensorDesc input_grad_desc(
        *input_grad, CNNL_LAYOUT_ARRAY,
        ToCnnlDataType(framework::TransToProtoVarType(input_grad->dtype())));

    auto value = static_cast<T>(1.0 / static_cast<float>(reduce_numel));
    MLUCnnl::Fill(context, value, input_grad_desc.get(),
                  GetBasePtr(input_grad));

    MLUCnnlOpTensorDesc op_tensor_desc(CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN);

    MLUCnnl::OpTensor(context, op_tensor_desc.get(), output_grad_desc.get(),
                      GetBasePtr(&tmp_output_grad), input_grad_desc.get(),
                      GetBasePtr(input_grad), input_grad_desc.get(),
                      GetBasePtr(input_grad), ToCnnlDataType<T>());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(reduce_mean, ops::ReduceMeanMLUKernel<float>,
                       ops::ReduceMeanMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(reduce_mean_grad, ops::ReduceMeanGradMLUKernel<float>,
                       ops::ReduceMeanGradMLUKernel<plat::float16>);
