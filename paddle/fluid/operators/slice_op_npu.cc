/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

void UpdateAttr(const framework::DDim& in_dims,
                const std::vector<int> axes,
                const std::vector<int> starts,
                const std::vector<int> ends,
                std::vector<int>* offsets,
                std::vector<int>* size) {
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

template <typename T>
class SliceNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

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
    auto starts_tensor_list =
        ctx.MultiInput<phi::DenseTensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts =
          GetDataFromTensor<int>(ctx.Input<phi::DenseTensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<phi::DenseTensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int>(ctx.Input<phi::DenseTensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int>(ends_tensor_list);
    }

    PADDLE_ENFORCE_EQ(
        starts.size(),
        axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(),
        axes.size(),
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

      phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims = phi::funcs::GetSliceDims<int>(
          in_dims, axes, starts, ends, nullptr, nullptr);
      out_dims = phi::funcs::GetDecreasedDims(slice_dims, decrease_axis);

      out->Resize(out_dims);
    }

    out->mutable_data<T>(ctx.GetPlace());

    std::vector<int> offsets(in_dims.size());
    std::vector<int> size(in_dims.size());

    UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

    auto& dev_ctx = ctx.template device_context<NPUDeviceContext>();
    auto stream = dev_ctx.stream();
#if CANN_VERSION_CODE < 512000
    const auto& runner =
        NpuOpRunner("SliceD", {*input}, {*out}, {{"offsets", offsets}, {
                                                   "size",
                                                   size
                                                 }});
#else
    NpuOpRunner runner;
    runner.SetType("Slice")
        .AddInput(*input)
        .AddInput(std::move(offsets))
        .AddInput(std::move(size))
        .AddOutput(*out);
#endif
    runner.Run(stream);
  }
};

template <typename T>
class SliceGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dinput =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int> axes(axes_int.begin(), axes_int.end());
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    // Get the accurate attribute value of starts and ends
    auto starts_tensor_list =
        ctx.MultiInput<phi::DenseTensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts =
          GetDataFromTensor<int>(ctx.Input<phi::DenseTensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<phi::DenseTensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int>(ctx.Input<phi::DenseTensor>("EndsTensor"));
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

    phi::DenseTensor tmp_dout;
    tmp_dout.ShareDataWith(*dout);
    auto out_dims = dout->dims();
    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto decrease_size = decrease_axis.size();
    if (decrease_size > 0) {
      if (decrease_size == static_cast<size_t>(in_dims.size())) {
        out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
      } else {
        std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
        for (size_t i = 0; i < decrease_size; ++i) {
          origin_out_shape[decrease_axis[i]] = 1;
        }
        int index = 0;
        for (size_t i = 0; i < origin_out_shape.size(); ++i) {
          if (origin_out_shape[i] == -1) {
            origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }
        out_dims = phi::make_ddim(origin_out_shape);
      }
      tmp_dout.Resize(out_dims);
    }

    dinput->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner =
        NpuOpRunner("PadD", {tmp_dout}, {*dinput}, {{"paddings", paddings}});
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(slice,
                       ops::SliceNPUKernel<float>,
                       ops::SliceNPUKernel<int>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::SliceNPUKernel<int64_t>,
#endif
                       ops::SliceNPUKernel<paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(slice_grad,
                       ops::SliceGradNPUKernel<float>,
                       ops::SliceGradNPUKernel<int>,
                       ops::SliceGradNPUKernel<paddle::platform::float16>);
