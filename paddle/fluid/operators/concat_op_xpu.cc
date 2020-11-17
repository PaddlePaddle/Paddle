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

#include "paddle/fluid/operators/concat_op.h"

#include <memory>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_MKLDNN
#include <paddle/fluid/platform/mkldnn_helper.h>
#endif

#ifdef PADDLE_WITH_XPU

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ConcatXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");
    int axis = ctx.Attr<int>("axis");
    PADDLE_ENFORCE_NE(ins[0], nullptr, platform::errors::InvalidArgument(
                                           "The input should not be null."));
    PADDLE_ENFORCE_NE(ctx.HasInput("AxisTensor"), true,
                      platform::errors::InvalidArgument(
                          "XPU donot surpport AxisTensor for now"));
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));
    PADDLE_ENFORCE_GE(
        axis, 0, platform::errors::InvalidArgument("concat: axis shoud >= 0!"));
    PADDLE_ENFORCE_LT(axis, ins[0]->dims().size(),
                      platform::errors::InvalidArgument(
                          "concat: axis shoud < ins[0]->dims()!"));
    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);
    std::vector<int> choose_idx;
    int n = 0;
    for (unsigned int i = 0; i < ins.size(); ++i) {
      if (ins[i] && ins[i]->numel() > 0) {
        choose_idx.push_back(i);
        n++;
      }
    }
    PADDLE_ENFORCE_LE(n, 8, platform::errors::InvalidArgument(
                                "XPU only surpport at most 8 tensors for now"));
    PADDLE_ENFORCE_GT(
        n, 0, platform::errors::InvalidArgument("No tensor need concat?"));
    int h = 1;
    int w_except_axis = 1;
    for (int i = 0; i < axis; ++i) {
      h *= (ins[choose_idx[0]]->dims())[i];
    }
    for (int i = axis + 1; i < ins[0]->dims().size(); ++i) {
      w_except_axis *= (ins[choose_idx[0]]->dims())[i];
    }
    for (int i = 1; i < n; ++i) {
      int hh = 1;
      int ww = 1;
      for (int j = 0; j < axis; ++j) {
        hh *= (ins[choose_idx[i]]->dims())[j];
      }
      for (int j = axis + 1; j < ins[i]->dims().size(); ++j) {
        ww *= (ins[choose_idx[i]]->dims())[j];
      }
      PADDLE_ENFORCE_EQ(hh, h, platform::errors::InvalidArgument(
                                   "concat: h should be eual!"));
      PADDLE_ENFORCE_EQ(ww, w_except_axis,
                        platform::errors::InvalidArgument(
                            "concat: w should be eual except for axis!"));
    }
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::unique_ptr<int[]> in_w_host(new int[n]);
    std::unique_ptr<const float* []> ptrs(new const float*[n]);
    for (int i = 0; i < n; ++i) {
      ptrs[i] = ins[choose_idx[i]]->data<T>();
      in_w_host[i] = w_except_axis * (ins[choose_idx[i]]->dims())[axis];
    }
    int r =
        xpu::concat<float>(dev_ctx.x_context(), h, (const int*)in_w_host.get(),
                           n, (const float**)ptrs.get(), out->data<T>());
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r));
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
    std::vector<framework::Tensor*> outputs;
    for (size_t j = 0; j < outs.size(); ++j) {
      if (out_var_names[j] != framework::kEmptyVarName &&
          outs[j]->numel() != 0UL) {
        outs[j]->mutable_data<T>(ctx.GetPlace());
        outputs.push_back(outs[j]);
      } else {
        outputs.push_back(nullptr);
      }
    }
    PADDLE_ENFORCE_GE(axis, 0, platform::errors::InvalidArgument(
                                   "concat_grad: axis shoud >= 0!"));
    PADDLE_ENFORCE_LT(axis, out_grad->dims().size(),
                      platform::errors::InvalidArgument(
                          "concat_grad: axis shoud < ins[0]->dims()!"));
    auto out_grad_stride = framework::stride_numel(out_grad->dims());
    int n = outputs.size();
    PADDLE_ENFORCE_LE(n, 16,
                      platform::errors::InvalidArgument(
                          "XPU only surpport at most 16 tensors for now"));
    int h = out_grad_stride[0] / out_grad_stride[axis];
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    std::unique_ptr<int[]> in_w_host(new int[n]);
    std::unique_ptr<float* []> ptrs(new float*[n]);
    for (int i = 0; i < n; ++i) {
      auto out_stride = framework::stride_numel(outputs[i]->dims());
      ptrs[i] = outputs[i]->data<T>();
      in_w_host[i] = out_stride[axis];
    }
    int r = xpu::concat_grad(dev_ctx.x_context(), h, in_w_host.get(), n,
                             reinterpret_cast<float**>(ptrs.get()),
                             out_grad->data<T>());
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
