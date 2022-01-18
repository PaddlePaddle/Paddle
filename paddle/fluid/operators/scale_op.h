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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/pten_utils.h"

// only can include the headers in paddle/top/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/kernels/scale_kernel.h"

namespace paddle {
namespace operators {

template <typename T>
static inline T GetAttrFromTensor(const framework::Tensor* tensor) {
  const auto* tensor_data = tensor->data<T>();
  framework::Tensor cpu_tensor;
  if (platform::is_gpu_place(tensor->place()) ||
      platform::is_npu_place(tensor->place())) {
    paddle::framework::TensorCopySync(*tensor, platform::CPUPlace(),
                                      &cpu_tensor);
    tensor_data = cpu_tensor.data<T>();
  }
  return tensor_data[0];
}

// See Note [ Why still keep the original kernel implementation? ]
template <typename DeviceContext, typename T>
class ScaleKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);

    auto bias = ctx.Attr<float>("bias");
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    auto scale = ctx.Attr<float>("scale");
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<framework::Tensor>("ScaleTensor");
      scale = static_cast<float>(GetAttrFromTensor<T>(scale_tensor));
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
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // call new kernel
    pten::ScaleKernel<T>(dev_ctx, *in, scale, bias, bias_after_scale, out);
  }
};

}  // namespace operators
}  // namespace paddle
