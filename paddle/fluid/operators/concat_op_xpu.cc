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
#include "paddle/fluid/operators/concat_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/xpu/xpu_header.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ConcatXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    framework::LoDTensor* out = ctx.Output<framework::LoDTensor>("Out");
    int axis = ctx.Attr<int>("axis");
    PADDLE_ENFORCE_NE(ins[0], nullptr, platform::errors::InvalidArgument(
                                           "The input should not be null."));
    PADDLE_ENFORCE_NE(ctx.HasInput("AxisTensor"), true,
                      platform::errors::InvalidArgument(
                          "XPU donot surpport AxisTensor for now"));
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    PADDLE_ENFORCE_GE(axis, 0, platform::errors::InvalidArgument(
                                   "concat: axis should be larger than or "
                                   "equal to 0, but received axis is %d.",
                                   axis));
    PADDLE_ENFORCE_LT(axis, ins[0]->dims().size(),
                      platform::errors::InvalidArgument(
                          "concat: axis should be less than ins[0]->dims()!"
                          "But received axis is %d, while ins[0]->dims()"
                          "size is %d.",
                          axis, ins[0]->dims().size()));

    // If axis is 0, the lod of the output is not the same as inputs.
    if (axis == 0 && ins[0]->lod().size() > 0) {
      size_t lod_size_0 = ins[0]->lod().size();
      size_t lod_size = lod_size_0;
      for (size_t i = 1; i < ins.size(); ++i) {
        if (ins[i]->lod().size() > 0) {
          PADDLE_ENFORCE_EQ(
              ins[i]->lod().size(), lod_size_0,
              platform::errors::Unimplemented(
                  "The lod level of all input LoDTensors should be same. "
                  "Maybe different lod level of input LoDTensors can concat,"
                  "it is not supported currently. The lod level of %dth input "
                  "is %d and first input is %d.",
                  i, ins[i]->lod().size(), lod_size_0));
        } else {
          lod_size = 0;
          break;
        }
      }
      if (lod_size) {
        auto* out_lod = out->mutable_lod();
        for (size_t i = 1; i < ins.size(); ++i) {
          auto in_lod = ConvertToLengthBasedLoD(ins[i]->lod());
          AppendLoD(out_lod, in_lod);
        }
      }
    }
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    std::vector<std::vector<int>> xdims_list;
    std::vector<const T*> ptrs;
    for (unsigned int i = 0; i < ins.size(); ++i) {
      if (ins[i] && ins[i]->numel() > 0) {
        ptrs.push_back(ins[i]->data<T>());
        int size = ins[i]->dims().size();
        std::vector<int> tmp_dims(size);
        for (int j = 0; j < size; ++j) {
          tmp_dims[j] = ins[i]->dims()[j];
        }
        xdims_list.push_back(tmp_dims);
      }
    }

    PADDLE_ENFORCE_GT(xdims_list.size(), 0, platform::errors::InvalidArgument(
                                                "No tensor need concat"));
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int r = xpu::concat<T>(dev_ctx.x_context(), ptrs, out->data<T>(),
                           xdims_list, axis);
    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU concat kernel return wrong value[%d %s]", r,
                          XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class ConcatGradXPUKernel : public framework::OpKernel<T> {
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
    PADDLE_ENFORCE_NE(ins[0], nullptr, platform::errors::InvalidArgument(
                                           "The input should not be null."));
    auto axis = ctx.Attr<int>("axis");
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<framework::Tensor>("AxisTensor");
      axis = GetDataFromTensor<int>(axis_tensor)[0];
    }
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    // get output tensor that the name is not kEmptyVarName
    std::vector<T*> ptrs(outs.size());
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        ptrs[j] = outs[j]->data<T>();
      } else {
        ptrs[j] = nullptr;
      }
    }
    PADDLE_ENFORCE_GE(axis, 0, platform::errors::InvalidArgument(
                                   "concat_grad: axis should be larger than or "
                                   "equal to 0, but received axis is %d.",
                                   axis));
    PADDLE_ENFORCE_LT(
        axis, out_grad->dims().size(),
        platform::errors::InvalidArgument(
            "concat_grad: axis should be less than ins[0]->dims()!"
            "But received axis is %d, while ins[0]->dims()"
            "size is %d.",
            axis, out_grad->dims().size()));

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
    int r = xpu::split<T>(dev_ctx.x_context(), out_grad->data<T>(), ptrs,
                          xdims_list, split_list, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
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
    concat, ops::ConcatXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    concat_grad,
    ops::ConcatGradXPUKernel<paddle::platform::XPUDeviceContext, float>);

#endif
