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
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class StridedSliceNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    PADDLE_ENFORCE_EQ(is_tensor_array, false,
                      platform::errors::InvalidArgument(
                          "Tensor array as input is not supported."));
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
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input is supported up to 6."));
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

    // construct the starts_indices, ends_indices and strides_indices tensor for
    // calling StridedSlice op
    std::vector<int64_t> starts_indices_vector(D, 0);
    std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                             out_dims_vector.end());
    std::vector<int64_t> strides_indices_vector(D, 1);

    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices_vector[axis_index] = starts[axis];
      ends_indices_vector[axis_index] = ends[axis];
      strides_indices_vector[axis_index] = strides[axis];
    }

    Tensor starts_indices_tensor;
    Tensor ends_indices_tensor;
    Tensor strides_indices_tensor;

    starts_indices_tensor.mutable_data<int64_t>({D}, place);
    ends_indices_tensor.mutable_data<int64_t>({D}, place);
    strides_indices_tensor.mutable_data<int64_t>({D}, place);

    paddle::framework::TensorFromVector(
        starts_indices_vector, ctx.device_context(), &starts_indices_tensor);
    paddle::framework::TensorFromVector(
        ends_indices_vector, ctx.device_context(), &ends_indices_tensor);
    paddle::framework::TensorFromVector(
        strides_indices_vector, ctx.device_context(), &strides_indices_tensor);

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
        "StridedSlice", {*in, starts_indices_tensor, ends_indices_tensor,
                         strides_indices_tensor},
        {*out}, {{"begin_mask", 0},
                 {"end_mask", 0},
                 {"ellipsis_mask", 0},
                 {"new_axis_mask", 0},
                 {"shrink_axis_mask", 0}});
    runner.Run(stream);

    if (need_reverse) {
      Tensor out_tmp;
      out_tmp.mutable_data<T>(out_dims, place);
      paddle::framework::TensorCopy(
          *out, place, ctx.template device_context<platform::DeviceContext>(),
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
      paddle::framework::TensorFromVector(reverse_axis_vector,
                                          ctx.device_context(), &reverse_axis);

      const auto& runner_reverse =
          NpuOpRunner("ReverseV2", {out_tmp, reverse_axis}, {*out});
      runner_reverse.Run(stream);
    }

    if (decrease_axis.size() > 0) {
      out->Resize(out_dims_origin);
    }
  }
};

template <typename DeviceContext, typename T>
class StridedSliceGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    PADDLE_ENFORCE_EQ(is_tensor_array, false,
                      platform::errors::InvalidArgument(
                          "Tensor array as input is not supported."));
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();

    switch (rank) {
      case 1:
        StridedSliceGradCompute<1>(ctx);
        break;
      case 2:
        StridedSliceGradCompute<2>(ctx);
        break;
      case 3:
        StridedSliceGradCompute<3>(ctx);
        break;
      case 4:
        StridedSliceGradCompute<4>(ctx);
        break;
      case 5:
        StridedSliceGradCompute<5>(ctx);
        break;
      case 6:
        StridedSliceGradCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input is supported up to 6."));
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceGradCompute(const framework::ExecutionContext& ctx) const {
    auto place = ctx.GetPlace();
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::NPUDeviceContext>();

    auto* input = ctx.Input<framework::Tensor>("Input");
    auto input_dims = input->dims();
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
    dx->mutable_data<T>(input_dims, place);

    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    auto strides_int = ctx.Attr<std::vector<int>>("strides");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> strides(strides_int.begin(), strides_int.end());

    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");
    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    auto list_new_ends_tensor =
        ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto list_new_strides_tensor =
        ctx.MultiInput<framework::Tensor>("StridesTensorList");

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

    std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
    StridedSliceOutDims(starts, ends, strides, axes, infer_flags, input_dims,
                        decrease_axis, out_dims_vector.data(), axes.size(),
                        false);

    std::vector<int> reverse_vector(starts.size(), 0);
    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), input_dims, infer_flags,
                        decrease_axis, starts.size());

    std::vector<int64_t> starts_indices_vector(D, 0);
    std::vector<int64_t> ends_indices_vector(out_dims_vector.begin(),
                                             out_dims_vector.end());
    std::vector<int64_t> strides_indices_vector(D, 1);

    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices_vector[axis_index] = starts[axis];
      ends_indices_vector[axis_index] = ends[axis];
      strides_indices_vector[axis_index] = strides[axis];
    }

    Tensor starts_indices_tensor;
    Tensor ends_indices_tensor;
    Tensor strides_indices_tensor;

    starts_indices_tensor.mutable_data<int64_t>({D}, place);
    ends_indices_tensor.mutable_data<int64_t>({D}, place);
    strides_indices_tensor.mutable_data<int64_t>({D}, place);

    paddle::framework::TensorFromVector(starts_indices_vector, dev_ctx,
                                        &starts_indices_tensor);
    paddle::framework::TensorFromVector(ends_indices_vector, dev_ctx,
                                        &ends_indices_tensor);
    paddle::framework::TensorFromVector(strides_indices_vector, dev_ctx,
                                        &strides_indices_tensor);

    std::vector<int64_t> input_dims_vector;
    for (int i = 0; i < input_dims.size(); i++) {
      input_dims_vector.push_back(input_dims[i]);
    }
    Tensor input_dims_tensor;
    paddle::framework::TensorFromVector(input_dims_vector, dev_ctx,
                                        &input_dims_tensor);

    bool need_reverse = false;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        need_reverse = true;
        break;
      }
    }

    auto stream = dev_ctx.stream();
    framework::NPUAttributeMap attr_input = {{"begin_mask", 0},
                                             {"end_mask", 0},
                                             {"ellipsis_mask", 0},
                                             {"new_axis_mask", 0},
                                             {"shrink_axis_mask", 0}};

    if (need_reverse) {
      Tensor reverse_axis;
      std::vector<int> reverse_axis_vector;
      for (size_t axis = 0; axis < axes.size(); axis++) {
        if (reverse_vector[axis] == 1) {
          reverse_axis_vector.push_back(axes[axis]);
        }
      }
      reverse_axis.mutable_data<int>(
          {static_cast<int>(reverse_axis_vector.size())}, place);
      paddle::framework::TensorFromVector(reverse_axis_vector, dev_ctx,
                                          &reverse_axis);

      Tensor dout_tmp;
      dout_tmp.mutable_data<T>(dout->dims(), place);
      const auto& runner_reverse =
          NpuOpRunner("ReverseV2", {*dout, reverse_axis}, {dout_tmp});
      runner_reverse.Run(stream);

      const auto& runner =
          NpuOpRunner("StridedSliceGrad",
                      {input_dims_tensor, starts_indices_tensor,
                       ends_indices_tensor, strides_indices_tensor, dout_tmp},
                      {*dx}, attr_input);
      runner.Run(stream);
    } else {
      const auto& runner =
          NpuOpRunner("StridedSliceGrad",
                      {input_dims_tensor, starts_indices_tensor,
                       ends_indices_tensor, strides_indices_tensor, *dout},
                      {*dx}, attr_input);
      runner.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    strided_slice, ops::StridedSliceNPUKernel<plat::NPUDeviceContext, bool>,
    ops::StridedSliceNPUKernel<plat::NPUDeviceContext, int>,
    ops::StridedSliceNPUKernel<plat::NPUDeviceContext, int64_t>,
    ops::StridedSliceNPUKernel<plat::NPUDeviceContext, float>,
    ops::StridedSliceNPUKernel<plat::NPUDeviceContext, double>);

REGISTER_OP_NPU_KERNEL(
    strided_slice_grad,
    ops::StridedSliceGradNPUKernel<plat::NPUDeviceContext, plat::float16>,
    ops::StridedSliceGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::StridedSliceGradNPUKernel<plat::NPUDeviceContext, double>,
    ops::StridedSliceGradNPUKernel<plat::NPUDeviceContext, int>,
    ops::StridedSliceGradNPUKernel<plat::NPUDeviceContext, int64_t>);
