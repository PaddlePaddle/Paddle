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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class MeanMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(context.GetPlace());
    auto numel = input->numel();
    auto rank = input->dims().size();
    auto place = context.GetPlace();
    auto stream = context.template device_context<MLUDeviceContext>().stream();

    if (rank == 0) {  // scalar
      memory::Copy(place, out_data, place, in_data, numel * sizeof(T), stream);
      return;
    }

    std::vector<int> reduce_dims;
    reduce_dims.reserve(rank);
    for (decltype(rank) i = 0; i < rank; ++i) {
      reduce_dims.push_back(i);
    }

    MLUCnnlTensorDesc input_desc(*input, CNNL_LAYOUT_ARRAY,
                                 ToCnnlDataType(input->dtype()));
    MLUCnnlTensorDesc output_desc(*output, CNNL_LAYOUT_ARRAY,
                                  ToCnnlDataType(output->dtype()));

    MLUCnnlReduceDesc reduction_desc(
        reduce_dims, CNNL_REDUCE_AVG, ToCnnlDataType<T>(),
        CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES);

    MLUCnnl::Reduce(context, true /*need_workspace*/, reduction_desc.get(),
                    nullptr, input_desc.get(),
                    reinterpret_cast<const void*>(in_data), 0 /*indices_size*/,
                    nullptr, nullptr, output_desc.get(),
                    reinterpret_cast<void*>(out_data));
  }
};

template <typename T>
class MeanMLUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto output_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(output_grad->numel(), 1,
                      platform::errors::InvalidArgument(
                          "Mean Gradient Input Tensor len should be 1. But "
                          "received Out@Grad's elements num is %d.",
                          output_grad->numel()));
    auto input_grad = context.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(context.GetPlace());

    auto in_data = output_grad->data<T>();
    auto numel = input_grad->numel();
    auto rank = input_grad->dims().size();
    auto out_data = input_grad->data<T>();
    auto place = context.GetPlace();
    auto stream = context.template device_context<MLUDeviceContext>().stream();

    if (rank == 0) {  // scalar
      memory::Copy(place, out_data, place, in_data, numel * sizeof(T), stream);
      return;
    }

    // means
    Tensor mean_var(output_grad->dtype());
    mean_var.mutable_data<T>(input_grad->dims(), context.GetPlace());
    MLUCnnlTensorDesc mean_var_desc(mean_var, CNNL_LAYOUT_ARRAY,
                                    ToCnnlDataType(mean_var.dtype()));
    auto value = static_cast<T>(1.0 / static_cast<float>(input_grad->numel()));
    MLUCnnl::Fill(context, CNNL_POINTER_MODE_HOST, &value, mean_var_desc.get(),
                  GetBasePtr(&mean_var));

    // means mul output_grad
    MLUCnnlTensorDesc in_desc(*output_grad, CNNL_LAYOUT_ARRAY,
                              ToCnnlDataType(output_grad->dtype()));
    MLUCnnlTensorDesc out_desc(*input_grad, CNNL_LAYOUT_ARRAY,
                               ToCnnlDataType(input_grad->dtype()));

    MLUCnnlOpTensorDesc op_tensor_desc(CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN);

    MLUCnnl::OpTensor(context, op_tensor_desc.get(), in_desc.get(),
                      reinterpret_cast<const void*>(in_data),
                      mean_var_desc.get(), GetBasePtr(&mean_var),
                      out_desc.get(), reinterpret_cast<void*>(out_data),
                      ToCnnlDataType<T>());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(mean, ops::MeanMLUKernel<float>,
                       ops::MeanMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(mean_grad, ops::MeanMLUGradKernel<float>,
                       ops::MeanMLUGradKernel<plat::float16>);
