/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/slice_op.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SliceXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.Input<framework::Tensor>("Input");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");
    auto in_dims = in->dims();

    // prepare starts, ends on XPU
    int dim_value = 0, start = 0, end = 0;
    // If a negative value is passed for any of the start or end indices,
    // it represents number of elements before the end of that dimension.
    // If the value passed to start or end is larger than the n
    // (the number of elements in this dimension), it represents n.
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = in_dims[axes[i]];
      start = starts[i];
      end = ends[i];
      start = start < 0 ? (start + dim_value) : start;
      end = end < 0 ? (end + dim_value) : end;
      start = std::max(start, 0);
      end = std::max(end, 0);
      end = std::min(end, dim_value);
      PADDLE_ENFORCE_GT(end, start, platform::errors::InvalidArgument(
                                        "end should greater than start"));
      starts[i] = start;
      ends[i] = end;
    }
    size_t shape_size = in_dims.size();
    // the slice XPU kernel require that the length of `start`, `end` must be
    // equal
    // to the dims size of input tensor, therefore, if shape_size > axes.size(),
    // the `starts_extension` and `ends_extension` is necessary.
    std::vector<int> starts_extension(shape_size, 0);
    std::vector<int> ends_extension(shape_size, 0);
    if (shape_size > axes.size()) {
      for (size_t i = 0; i < shape_size; ++i) {
        ends_extension[i] = in_dims[i];
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        starts_extension[axes[i]] = starts[i];
        ends_extension[axes[i]] = ends[i];
      }
    } else {
      starts_extension = std::move(starts);
      ends_extension = std::move(ends);
    }

    // prepare shape on XPU
    std::vector<int> shape(shape_size, 0);
    for (size_t i = 0; i < shape_size; ++i) {
      shape[i] = in_dims[i];
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    const XPUType* in_data = reinterpret_cast<const XPUType*>(in->data<T>());
    XPUType* out_data =
        reinterpret_cast<XPUType*>(out->mutable_data<T>(ctx.GetPlace()));
    int r = xpu::slice<XPUType>(dev_ctx.x_context(), in_data, out_data, shape,
                                starts_extension, ends_extension);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU slice kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class SliceGradXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dinput = ctx.Output<Tensor>(framework::GradVarName("Input"));

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int> axes(axes_int.begin(), axes_int.end());
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    // Get the accurate attribute value of starts and ends
    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int>(ends_tensor_list);
    }

    const auto& in_dims = input->dims();
    int rank = in_dims.size();

    std::vector<int> pad_left(rank);
    std::vector<int> out_dims(rank);
    std::vector<int> pad_right(rank);
    int cnt = 0;
    for (int i = 0; i < in_dims.size(); ++i) {
      int start = 0;
      int end = in_dims[i];
      int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
      if (axis == i) {
        start = starts[cnt];
        if (start < 0) {
          start = (start + in_dims[i]);
        }
        start = std::max(start, static_cast<int>(0));
        end = ends[cnt];
        if (end < 0) {
          end = (end + in_dims[i]);
        }
        end = std::min(end, static_cast<int>(in_dims[i]));
        cnt++;
      }

      pad_left[i] = start;
      out_dims[i] = end - start;
      pad_right[i] = in_dims[i] - out_dims[i] - pad_left[i];
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    const XPUType* dout_data =
        reinterpret_cast<const XPUType*>(dout->data<T>());
    XPUType* din_data =
        reinterpret_cast<XPUType*>(dinput->mutable_data<T>(ctx.GetPlace()));
    int r = xpu::pad<XPUType>(dev_ctx.x_context(), dout_data, din_data,
                              out_dims, pad_left, pad_right, XPUType(0));
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU pad kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    slice, ops::SliceXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SliceXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::SliceXPUKernel<paddle::platform::XPUDeviceContext,
                        paddle::platform::float16>);
REGISTER_OP_XPU_KERNEL(
    slice_grad,
    ops::SliceGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::SliceGradXPUKernel<paddle::platform::XPUDeviceContext, int>,
    ops::SliceGradXPUKernel<paddle::platform::XPUDeviceContext,
                            paddle::platform::float16>);
#endif
