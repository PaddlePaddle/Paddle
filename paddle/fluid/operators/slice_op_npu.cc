/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

void UpdateAttr(const framework::DDim& in_dims, const std::vector<int> axes,
                const std::vector<int> starts, const std::vector<int> ends,
                std::vector<int>* offsets, std::vector<int>* size) {
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    // NOTE(zhiqiu): Becareful that cnt may > axes.size() and result in
    // overflow.
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

    (*offsets)[i] = start;
    (*size)[i] = end - start;
  }
}

template <typename DeviceContext, typename T>
class SliceNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* out = ctx.Output<Tensor>("Out");

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int> axes(axes_int.begin(), axes_int.end());
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

    const auto& in_dims = input->dims();

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

    PADDLE_ENFORCE_EQ(
        starts.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    if (ctx.HasInput("StartsTensor") || ctx.HasInput("EndsTensor") ||
        starts_tensor_list.size() > 0 || ends_tensor_list.size() > 0) {
      // Infer output dims
      auto out_dims = out->dims();
      auto slice_dims = out_dims;
      for (size_t i = 0; i < axes.size(); ++i) {
        // when start == -1 && end == start+1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = in_dims[axes[i]];
          }
        }
      }

      CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims =
          GetSliceDims<int>(in_dims, axes, starts, ends, nullptr, nullptr);
      out_dims = GetDecreasedDims(slice_dims, decrease_axis);

      out->Resize(out_dims);
    }

    out->mutable_data<T>(ctx.GetPlace());

    std::vector<int> offsets(in_dims.size());
    std::vector<int> size(in_dims.size());

    UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    const auto& runner = NpuOpRunner("SliceD", {*input}, {*out},
                                     {{"offsets", offsets}, {"size", size}});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class SliceGradNPUKernel : public framework::OpKernel<T> {
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

    std::vector<int> offsets(rank);
    std::vector<int> size(rank);
    UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    std::vector<std::vector<int64_t>> paddings(rank, std::vector<int64_t>(2));
    for (int i = 0; i < rank; ++i) {
      paddings[i][0] = static_cast<int64_t>(offsets[i]);
      paddings[i][1] = static_cast<int64_t>(in_dims[i] - size[i] - offsets[i]);
    }

    dinput->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner =
        NpuOpRunner("PadD", {*dout}, {*dinput}, {{"paddings", paddings}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    slice, ops::SliceNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SliceNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SliceNPUKernel<paddle::platform::NPUDeviceContext,
                        paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    slice_grad,
    ops::SliceGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SliceGradNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SliceGradNPUKernel<paddle::platform::NPUDeviceContext,
                            paddle::platform::float16>);
