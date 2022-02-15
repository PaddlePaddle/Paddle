/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T, typename InputT>
void BincountInner(const framework::ExecutionContext& context) {
  const Tensor* input = context.Input<framework::Tensor>("X");
  const Tensor* weights = context.Input<framework::Tensor>("Weights");
  Tensor* output = context.Output<framework::Tensor>("Out");
  auto& minlength = context.Attr<int>("minlength");

  const InputT* input_data = input->data<InputT>();

  auto input_numel = input->numel();

  if (input_data == nullptr) {
    framework::DDim out_dim{0};
    output->Resize(out_dim);
    output->mutable_data<InputT>(context.GetPlace());
    return;
  }

  PADDLE_ENFORCE_GE(
      *std::min_element(input_data, input_data + input_numel),
      static_cast<InputT>(0),
      platform::errors::InvalidArgument(
          "The elements in input tensor must be non-negative ints"));

  int64_t output_size = static_cast<int64_t>(*std::max_element(
                            input_data, input_data + input_numel)) +
                        1L;
  output_size = std::max(output_size, static_cast<int64_t>(minlength));

  framework::DDim out_dim{output_size};
  output->Resize(out_dim);

  bool has_weights = (weights != nullptr);

  if (has_weights) {
    const T* weights_data = weights->data<T>();
    const auto& weights_type = framework::TransToProtoVarType(weights->dtype());
    if (weights_type == framework::proto::VarType::FP32) {
      float* output_data = output->mutable_data<float>(context.GetPlace());
      pten::funcs::SetConstant<DeviceContext, float>()(
          context.template device_context<DeviceContext>(), output,
          static_cast<float>(0));
      for (int64_t i = 0; i < input_numel; i++) {
        output_data[input_data[i]] += static_cast<float>(weights_data[i]);
      }
    } else {
      double* output_data = output->mutable_data<double>(context.GetPlace());
      pten::funcs::SetConstant<DeviceContext, double>()(
          context.template device_context<DeviceContext>(), output,
          static_cast<double>(0));
      for (int64_t i = 0; i < input_numel; i++) {
        output_data[input_data[i]] += static_cast<double>(weights_data[i]);
      }
    }

  } else {
    int64_t* output_data = output->mutable_data<int64_t>(context.GetPlace());
    pten::funcs::SetConstant<DeviceContext, int64_t>()(
        context.template device_context<DeviceContext>(), output, 0L);
    for (int64_t i = 0; i < input_numel; i++) {
      output_data[input_data[i]] += 1L;
    }
  }
}

template <typename DeviceContext, typename T>
class BincountKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<framework::Tensor>("X");
    const auto& input_type = framework::TransToProtoVarType(input->dtype());

    if (input_type == framework::proto::VarType::INT32) {
      BincountInner<DeviceContext, T, int>(context);
    } else if (input_type == framework::proto::VarType::INT64) {
      BincountInner<DeviceContext, T, int64_t>(context);
    }
  }
};

}  // namespace operators
}  // namespace paddle
