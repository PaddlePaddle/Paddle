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
#ifdef PADDLE_WITH_XPU
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/core/lod_utils.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ConcatXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

template <typename DeviceContext, typename T>
class ConcatGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto out_var_names = ctx.OutputNames(framework::GradVarName("X"));
    auto outs =
        ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    {
      auto dx = outs;
      auto x = ins;
      for (size_t i = 0; i < dx.size(); ++i) {
        if (dx[i] != nullptr) {
          dx[i]->set_lod(x[i]->lod());
        }
      }
    }
    PADDLE_ENFORCE_NE(
        ins[0],
        nullptr,
        platform::errors::InvalidArgument("The input should not be null."));
    auto axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    // get output tensor that the name is not kEmptyVarName
    std::vector<XPUType*> ptrs(outs.size());
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        ptrs[j] = reinterpret_cast<XPUType*>(outs[j]->data<T>());
      } else {
        ptrs[j] = nullptr;
      }
    }
    PADDLE_ENFORCE_GE(axis,
                      0,
                      platform::errors::InvalidArgument(
                          "concat_grad: axis should be larger than or "
                          "equal to 0, but received axis is %d.",
                          axis));
    PADDLE_ENFORCE_LT(
        axis,
        out_grad->dims().size(),
        platform::errors::InvalidArgument(
            "concat_grad: axis should be less than ins[0]->dims()!"
            "But received axis is %d, while ins[0]->dims()"
            "size is %d.",
            axis,
            out_grad->dims().size()));

    auto input_dims = ins[0]->dims();
    std::vector<int> split_list(ins.size());
    std::vector<int> xdims_list(input_dims.size());
    int total_length = 0;
    for (size_t i = 0; i < ins.size(); ++i) {
      split_list[i] = ins[i]->dims()[axis];
      total_length += ins[i]->dims()[axis];
    }
    for (int i = 0; i < input_dims.size(); ++i) {
      if (i == axis) {
        continue;
      }
      xdims_list[i] = input_dims[i];
    }
    xdims_list[axis] = total_length;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::split<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType*>(out_grad->data<T>()),
        ptrs,
        xdims_list,
        split_list,
        axis);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    concat_grad,
    ops::ConcatGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::ConcatGradXPUKernel<paddle::platform::XPUDeviceContext,
                             paddle::platform::float16>);

#endif
