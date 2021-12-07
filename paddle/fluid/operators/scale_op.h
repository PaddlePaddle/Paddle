/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace paddle {
namespace operators {

template <typename T>
static inline T GetAttrFromTensor(const framework::Tensor* tensor) {
  const auto* tensor_data = tensor->data<T>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place()) ||
      platform::is_npu_place(tensor->place())) {
    TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    tensor_data = cpu_tensor.data<T>();
  }
  return tensor_data[0];
}

template <typename DeviceContext, typename T>
class ScaleKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);

    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<framework::Tensor>("ScaleTensor");
      scale = GetAttrFromTensor<T>(scale_tensor);
    }

    auto* out_var = ctx.OutputVar("Out");
    if (in_var->IsType<framework::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<framework::SelectedRows>();
      auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }

    auto* out =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);
    out->mutable_data<T>(in->place());

    PADDLE_ENFORCE_EQ(in->dims(), out->dims(),
                      paddle::platform::errors::InvalidArgument(
                          "the input and output should have the same dim"
                          "but input dim is %s, output dim is %s",
                          in->dims(), out->dims()));

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    EigenScale<std::decay_t<decltype(dev)>, T>::Eval(
        dev, eigen_out, eigen_in, scale, bias, bias_after_scale);
  }
};

}  // namespace operators
}  // namespace paddle
