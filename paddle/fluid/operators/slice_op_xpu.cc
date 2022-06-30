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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline void DealTensorArray(const framework::ExecutionContext& ctx,
                            const std::vector<int>& starts,
                            const std::vector<int>& ends,
                            bool out_is_array) {
  auto in_array = ctx.Input<LoDTensorArray>("Input");
  // If the input is LoDTensorArray, the rank of input is 1.
  int in_size = in_array->size();
  int start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  int end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

  start = std::max(start, static_cast<int>(0));
  end = std::max(end, static_cast<int>(0));
  end = std::min(end, in_size);

  if (starts[0] == -1 && end == 0) {
    end = start + 1;
  }

  PADDLE_ENFORCE_GT(end,
                    start,
                    platform::errors::InvalidArgument(
                        "Attr(ends) should be greater than attr(starts) in "
                        "slice op. But received end = %d, start = %d.",
                        ends[0],
                        starts[0]));
  int out_size = end - start;

  if (out_is_array) {
    auto out_array = ctx.Output<LoDTensorArray>("Out");
    out_array->resize(out_size);

    for (int i = 0; i < out_size; ++i) {
      auto* out_tensor = &out_array->at(i);
      auto in_tensor = in_array->at(i + start);
      out_tensor->set_lod(in_tensor.lod());
      if (in_tensor.memory_size() > 0) {
        paddle::framework::TensorCopy(in_tensor, ctx.GetPlace(), out_tensor);
      } else {
        VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                    "nothing has been written to output array["
                 << i << "].";
      }
    }
  } else {
    auto out = ctx.Output<Tensor>("Out");
    auto in_tensor = in_array->at(start);
    paddle::framework::TensorCopy(in_tensor, ctx.GetPlace(), out);
  }
}
template <typename DeviceContext, typename T>
class SliceXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    Variable* out_var = ctx.OutputVar("Out");
    bool input_is_array = input_var->IsType<LoDTensorArray>();
    bool out_is_array = out_var->IsType<LoDTensorArray>();

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int> axes(axes_int.begin(), axes_int.end());
    std::vector<int> starts(starts_int.begin(), starts_int.end());
    std::vector<int> ends(ends_int.begin(), ends_int.end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

    // Step 1: Get the accurate attribute value of starts and ends
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
        starts.size(),
        axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(),
        axes.size(),
        platform::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    // Step 2: Compute output
    if (input_is_array) {
      DealTensorArray(ctx, starts, ends, out_is_array);
      return;
    } else {
      auto in = ctx.Input<framework::Tensor>("Input");
      auto out = ctx.Output<framework::Tensor>("Out");

      auto in_dims = in->dims();
      auto out_dims = out->dims();
      auto slice_dims = out_dims;

      // 2.1 Infer output dims
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

      // 2.2 Get output
      size_t shape_size = in_dims.size();
      // the slice XPU kernel require that the length of `start`, `end` must be
      // equal
      // to the dims size of input tensor, therefore, if shape_size >
      // axes.size(), the `starts_extension` and `ends_extension` is necessary.
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
      int r = xpu::slice<XPUType>(dev_ctx.x_context(),
                                  in_data,
                                  out_data,
                                  shape,
                                  starts_extension,
                                  ends_extension);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "slice");
    }
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
    int r = xpu::pad<XPUType>(dev_ctx.x_context(),
                              dout_data,
                              din_data,
                              out_dims,
                              pad_left,
                              pad_right,
                              XPUType(0));
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    slice,
    ops::SliceXPUKernel<paddle::platform::XPUDeviceContext, float>,
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
