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

#include "paddle/fluid/operators/strided_slice_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class StridedSliceNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        StridedSliceCompute<1>(ctx);
        break;
      case 2:
        StridedSliceCompute<2>(ctx);
        break;
      case 3:
        StridedSliceCompute<3>(ctx);
        break;
      case 4:
        StridedSliceCompute<4>(ctx);
        break;
      case 5:
        StridedSliceCompute<5>(ctx);
        break;
      case 6:
        StridedSliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceCompute(const framework::ExecutionContext& ctx) const {
    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    auto in = ctx.Input<framework::Tensor>("Input");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto in_dims = in->dims();

    // list<int>
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    auto strides_int = ctx.Attr<std::vector<int>>("strides");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> strides(strides_int.begin(), strides_int.end());

    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");
    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    // vector<Tensor<int32>>
    auto list_new_ends_tensor =
        ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto list_new_strides_tensor =
        ctx.MultiInput<framework::Tensor>("StridesTensorList");

    // Tensor<int32>
    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (ctx.HasInput("StartsTensor")) {
      auto* starts_tensor = ctx.Input<framework::Tensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (ctx.HasInput("EndsTensor")) {
      auto* ends_tensor = ctx.Input<framework::Tensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_strides_tensor.size() > 0) {
      strides = GetDataFromTensorList<int64_t>(list_new_strides_tensor);
    } else if (ctx.HasInput("StridesTensor")) {
      auto* strides_tensor = ctx.Input<framework::Tensor>("StridesTensor");
      strides = GetDataFromTensor<int64_t>(strides_tensor);
    }

    // out dims calculation
    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    StridedSliceOutDims(starts, ends, strides, axes, infer_flags, in_dims,
                        decrease_axis, out_dims_vector.data(), axes.size(),
                        false);
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    // check whether need to reverse (false: stride > 0; true: stride < 0)
    std::vector<int> reverse_vector(starts.size(), 0);
    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), in_dims, infer_flags,
                        decrease_axis, starts.size());

    // input tensors used for ScatterElements
    Tensor in_starts_indices;
    Tensor in_ends_indices;
    Tensor in_strides_indices;
    in_starts_indices.mutable_data<int64_t>({D}, place);
    in_ends_indices.mutable_data<int64_t>({D}, place);
    in_strides_indices.mutable_data<int64_t>({D}, place);

    // initialization
    FillNpuTensorWithConstant<int64_t>(&in_starts_indices, 0);
    TensorFromVector(out_dims_vector, ctx.device_context(), &in_ends_indices);
    FillNpuTensorWithConstant<int64_t>(&in_strides_indices, 1);

    // output tensors used for ScatterElements
    Tensor out_starts_indices = in_starts_indices;
    Tensor out_ends_indices = in_ends_indices;
    Tensor out_strides_indices = in_strides_indices;

    // call ScatterElements to prepare
    // starts_indices/ends_indices/strides_indices
    Tensor ids_tmp;
    Tensor updates_starts_tmp;
    Tensor updates_ends_tmp;
    Tensor updates_strides_tmp;

    ids_tmp.mutable_data<int64_t>({static_cast<int>(axes.size())}, place);
    updates_starts_tmp.mutable_data<int64_t>({static_cast<int>(axes.size())},
                                             place);
    updates_ends_tmp.mutable_data<int64_t>({static_cast<int>(axes.size())},
                                           place);
    updates_strides_tmp.mutable_data<int64_t>({static_cast<int>(axes.size())},
                                              place);

    std::vector<int64_t> ids_vector(axes.size(), 0);
    std::vector<int64_t> updates_starts_vector(axes.size(), 0);
    std::vector<int64_t> updates_ends_vector(axes.size(), 0);
    std::vector<int64_t> updates_strides_vector(axes.size(), 0);
    for (size_t axis = 0; axis < axes.size(); axis++) {
      ids_vector[axis] = axes[axis];
      updates_starts_vector[axis] = starts[axis];
      updates_ends_vector[axis] = ends[axis];
      updates_strides_vector[axis] = strides[axis];
    }

    TensorFromVector(ids_vector, ctx.device_context(), &ids_tmp);
    TensorFromVector(updates_starts_vector, ctx.device_context(),
                     &updates_starts_tmp);
    TensorFromVector(updates_ends_vector, ctx.device_context(),
                     &updates_ends_tmp);
    TensorFromVector(updates_strides_vector, ctx.device_context(),
                     &updates_strides_tmp);

    const auto& runner_starts = NpuOpRunner(
        "ScatterElements", {in_starts_indices, ids_tmp, updates_starts_tmp},
        {out_starts_indices}, {});
    runner_starts.Run(stream);

    const auto& runner_ends = NpuOpRunner(
        "ScatterElements", {in_ends_indices, ids_tmp, updates_ends_tmp},
        {out_ends_indices}, {});
    runner_ends.Run(stream);

    const auto& runner_strides = NpuOpRunner(
        "ScatterElements", {in_strides_indices, ids_tmp, updates_strides_tmp},
        {out_strides_indices}, {});
    runner_strides.Run(stream);

    auto out_dims_origin = out_dims;
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            out_dims[decrease_axis[i]], 1,
            platform::errors::InvalidArgument(
                "the size of decrease dimension should be 1, but received %d.",
                out_dims[decrease_axis[i]]));
        out_dims_origin[decrease_axis[i]] = 0;
      }

      for (int i = 0; i < out_dims_origin.size(); ++i) {
        if (out_dims_origin[i] != 0) {
          new_out_shape.push_back(out_dims_origin[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }
      out_dims_origin = framework::make_ddim(new_out_shape);
    }

    bool need_reverse = false;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        need_reverse = true;
        break;
      }
    }

    out->Resize(out_dims);
    out->mutable_data<T>(place);

    const auto& runner = NpuOpRunner(
        "StridedSlice",
        {*in, out_starts_indices, out_ends_indices, out_strides_indices},
        {*out}, {{"begin_mask", 0},
                 {"end_mask", 0},
                 {"ellipsis_mask", 0},
                 {"new_axis_mask", 0},
                 {"shrink_axis_mask", 0}});
    runner.Run(stream);

    if (need_reverse) {
      Tensor out_tmp;
      out_tmp.mutable_data<T>(out_dims, place);
      TensorCopy(*out, place,
                 ctx.template device_context<platform::DeviceContext>(),
                 &out_tmp);

      Tensor reverse_axis;
      std::vector<int> reverse_axis_vector;
      for (size_t axis = 0; axis < axes.size(); axis++) {
        if (reverse_vector[axis] == 1) {
          reverse_axis_vector.push_back(axes[axis]);
        }
      }
      reverse_axis.mutable_data<int>(
          {static_cast<int>(reverse_axis_vector.size())}, place);
      TensorFromVector(reverse_axis_vector, ctx.device_context(),
                       &reverse_axis);

      const auto& runner_reverse =
          NpuOpRunner("ReverseV2", {out_tmp, reverse_axis}, {*out});
      runner_reverse.Run(stream);
    }

    if (decrease_axis.size() > 0) {
      out->Resize(out_dims_origin);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    strided_slice,
    ops::StridedSliceNPUKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::StridedSliceNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::StridedSliceNPUKernel<paddle::platform::NPUDeviceContext, int64_t>,
    ops::StridedSliceNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::StridedSliceNPUKernel<paddle::platform::NPUDeviceContext, double>);
