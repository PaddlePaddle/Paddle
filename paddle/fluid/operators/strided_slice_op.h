/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/slice_op.h"
namespace paddle {
namespace operators {

static void StridedSliceOutDims(
    const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
    const std::vector<int64_t>& strides, const std::vector<int>& axes,
    const std::vector<int>& infer_flags, const framework::DDim in_dims,
    const std::vector<int>& decrease_axis, int64_t* out_dims_vector,
    const size_t size, bool infer_shape) {
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector[i] = in_dims[i];
  }
  int64_t stride_index, start_index, end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }

    PADDLE_ENFORCE_NE(stride_index, 0,
                      platform::errors::InvalidArgument(
                          "stride index in StridedSlice operator is 0."));
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool neg_dim_condition = ((stride_index < 0 && (start_index < end_index)) ||
                              (stride_index > 0 && (start_index > end_index)));
    PADDLE_ENFORCE_EQ(neg_dim_condition, false,
                      platform::errors::InvalidArgument(
                          "The start index and end index are invalid for their "
                          "corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void StridedSliceFunctor(int64_t* starts, int64_t* ends,
                                int64_t* strides, int* axes, int* reverse_axis,
                                const framework::DDim dims,
                                const std::vector<int>& infer_flags,
                                const std::vector<int>& decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(),
                           axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
      starts[axis_index] = std::max<int64_t>(starts[axis_index], 0);
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
        if (ends[axis_index] < 0) {
          ends[axis_index] = 0;
        }
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        auto end_dim = axis_size - 1 < starts[axis_index] ? axis_size - 1
                                                          : starts[axis_index];
        auto offset = (end_dim - ends[axis_index]) % strides[axis_index];
        offset = offset == 0 ? strides[axis_index] : offset;

        starts[axis_index] = starts[axis_index] + offset;
        ends[axis_index] = ends[axis_index] + offset;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

template <typename DeviceContext, typename T>
class StridedSliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    int rank = is_tensor_array
                   ? 1
                   : ctx.Input<framework::Tensor>("Input")->dims().size();
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
  void StridedSliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    framework::DDim in_dims;
    auto* input_var = context.InputVar("Input");

    bool is_input_var_array = input_var->IsType<LoDTensorArray>();
    if (is_input_var_array) {
      const int64_t size = input_var->Get<framework::LoDTensorArray>().size();
      in_dims = framework::make_ddim({size});
    } else {
      in_dims = context.Input<framework::Tensor>("Input")->dims();
    }

    auto starts_int = context.Attr<std::vector<int>>("starts");
    auto ends_int = context.Attr<std::vector<int>>("ends");
    auto strides_int = context.Attr<std::vector<int>>("strides");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> strides(strides_int.begin(), strides_int.end());

    auto axes = context.Attr<std::vector<int>>("axes");
    auto infer_flags = context.Attr<std::vector<int>>("infer_flags");
    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto reverse_axis = Eigen::array<bool, D>();

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");
    auto list_new_strides_tensor =
        context.MultiInput<framework::Tensor>("StridesTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (context.HasInput("StartsTensor")) {
      auto* starts_tensor = context.Input<framework::Tensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (context.HasInput("EndsTensor")) {
      auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_strides_tensor.size() > 0) {
      strides = GetDataFromTensorList<int64_t>(list_new_strides_tensor);
    } else if (context.HasInput("StridesTensor")) {
      auto* strides_tensor = context.Input<framework::Tensor>("StridesTensor");
      strides = GetDataFromTensor<int64_t>(strides_tensor);
    }

    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    StridedSliceOutDims(starts, ends, strides, axes, infer_flags, in_dims,
                        decrease_axis, out_dims_vector.data(), axes.size(),
                        false);
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));

    std::vector<int> reverse_vector(starts.size(), 0);
    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), in_dims, infer_flags,
                        decrease_axis, starts.size());

    for (size_t axis = 0; axis < D; axis++) {
      starts_indices[axis] = 0;
      ends_indices[axis] = out_dims[axis];
      strides_indices[axis] = 1;
      reverse_axis[axis] = false;
    }
    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      strides_indices[axis_index] = strides[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }

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

    if (is_input_var_array) {
      PADDLE_ENFORCE_EQ(
          starts_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_op' is `TensorArray`, the "
              "dimension of start index  should be 1, but received %d.",
              starts_indices.size()));

      PADDLE_ENFORCE_EQ(
          ends_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_op' is `TensorArray`, the "
              "dimension of end index should be 1, but received %d.",
              ends_indices.size()));

      PADDLE_ENFORCE_EQ(
          strides_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_op' is `TensorArray`, the "
              "dimension of stride should be 1, but received %d.",
              strides_indices.size()));

      auto* output_var = context.OutputVar("Out");

      PADDLE_ENFORCE_EQ(
          output_var->IsType<LoDTensorArray>(), true,
          platform::errors::InvalidArgument(
              "When the input of `strided_slice_op` is `TensorArray`. The "
              "output is excepted `TensorArray` , but received %s.",
              framework::ToTypeName(output_var->Type())));

      PADDLE_ENFORCE_EQ(
          out_dims_origin.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_op' is `TensorArray`, the "
              "dimension of Output should be 1, but received %d",
              out_dims_origin.size()));

      auto& in_array = input_var->Get<framework::LoDTensorArray>();

      auto* out_array = context.Output<framework::LoDTensorArray>("Out");

      out_array->resize(out_dims_origin[0]);
      size_t const in_array_size = in_array.size();
      for (size_t i = 0; i < out_array->size(); i++) {
        size_t in_offset =
            (starts_indices[0] % in_array_size) + i * strides_indices[0];

        int64_t out_offset = i;
        if (need_reverse) {
          out_offset = out_array->size() - i - 1;
        }

        auto& in_tensor = in_array.at(in_offset);
        PADDLE_ENFORCE_GT(
            in_tensor.memory_size(), 0,
            platform::errors::PreconditionNotMet(
                "The input LoDTensorArray Input[%d] holds no memory.",
                in_offset));
        auto* out_tensor = &out_array->at(out_offset);

        out_tensor->set_lod(in_tensor.lod());
        paddle::framework::TensorCopy(in_tensor, context.GetPlace(),
                                      out_tensor);
      }

    } else {
      auto in = context.Input<framework::Tensor>("Input");
      auto out = context.Output<framework::Tensor>("Out");
      out->Resize(out_dims);
      out->mutable_data<T>(context.GetPlace());
      auto in_t = framework::EigenTensor<T, D, Eigen::RowMajor,
                                         Eigen::DenseIndex>::From(*in);
      auto out_t =
          framework::EigenTensor<T, D, Eigen::RowMajor,
                                 Eigen::DenseIndex>::From(*out, out_dims);
      if (need_reverse) {
        framework::Tensor tmp;
        tmp.mutable_data<T>(out_dims, context.GetPlace());
        auto tmp_t = framework::EigenTensor<T, D, Eigen::RowMajor,
                                            Eigen::DenseIndex>::From(tmp);
        tmp_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, strides_indices);
        out_t.device(place) = tmp_t.reverse(reverse_axis);
      } else {
        out_t.device(place) =
            in_t.stridedSlice(starts_indices, ends_indices, strides_indices);
      }

      if (decrease_axis.size() > 0) {
        out->Resize(out_dims_origin);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class StridedSliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    int rank = is_tensor_array
                   ? 1
                   : ctx.Input<framework::Tensor>("Input")->dims().size();
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
    }
  }

 private:
  template <size_t D>
  void StridedSliceGradCompute(
      const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto& dev_ctx = context.template device_context<DeviceContext>();

    framework::DDim out_dims;
    auto* out_var = context.OutputVar(framework::GradVarName("Input"));
    bool is_out_var_array = out_var->IsType<LoDTensorArray>();
    if (is_out_var_array) {
      // Note(weixin):Since the shape of `framework::GradVarName("Input")` of
      // StridedSliceGrad cannot be calculated by
      // `framework::GradVarName("Output")`, the dim of "Input" is used to
      // calculate the output shape. when set it to inplace OP, there may be
      // some problems.
      const int64_t size =
          context.Input<framework::LoDTensorArray>("Input")->size();

      out_dims = framework::make_ddim({size});
    } else {
      out_dims =
          context.Output<framework::Tensor>(framework::GradVarName("Input"))
              ->dims();
    }

    auto starts_int = context.Attr<std::vector<int>>("starts");
    auto ends_int = context.Attr<std::vector<int>>("ends");
    auto strides_int = context.Attr<std::vector<int>>("strides");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> strides(strides_int.begin(), strides_int.end());

    auto axes = context.Attr<std::vector<int>>("axes");
    auto infer_flags = context.Attr<std::vector<int>>("infer_flags");
    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");
    auto list_new_strides_tensor =
        context.MultiInput<framework::Tensor>("StridesTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (context.HasInput("StartsTensor")) {
      auto* starts_tensor = context.Input<framework::Tensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (context.HasInput("EndsTensor")) {
      auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_strides_tensor.size() > 0) {
      strides = GetDataFromTensorList<int64_t>(list_new_strides_tensor);
    } else if (context.HasInput("StridesTensor")) {
      auto* strides_tensor = context.Input<framework::Tensor>("StridesTensor");
      strides = GetDataFromTensor<int64_t>(strides_tensor);
    }

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    auto reverse_axis = Eigen::array<bool, D>();
    std::vector<int> reverse_vector(starts.size(), 0);

    StridedSliceFunctor(starts.data(), ends.data(), strides.data(), axes.data(),
                        reverse_vector.data(), out_dims, infer_flags,
                        decrease_axis, starts.size());

    for (size_t axis = 0; axis < D; axis++) {
      starts_indices[axis] = 0;
      ends_indices[axis] = out_dims[axis];
      strides_indices[axis] = 1;
    }
    for (size_t axis = 0; axis < axes.size(); axis++) {
      int axis_index = axes[axis];
      starts_indices[axis_index] = starts[axis];
      ends_indices[axis_index] = ends[axis];
      strides_indices[axis_index] = strides[axis];
      reverse_axis[axis_index] = (reverse_vector[axis] == 1) ? true : false;
    }

    bool need_reverse = false;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        need_reverse = true;
        break;
      }
    }

    if (is_out_var_array) {
      PADDLE_ENFORCE_EQ(
          starts_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
              "dimension of start index  should be 1, but received %d.",
              starts_indices.size()));
      PADDLE_ENFORCE_EQ(
          ends_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_op' is `TensorArray`, the "
              "dimension of end index should be 1, but received %d.",
              ends_indices.size()));
      PADDLE_ENFORCE_EQ(
          strides_indices.size(), 1,
          platform::errors::InvalidArgument(
              "When the input of 'strided_slice_grad_op' is `TensorArray`, the "
              "dimension of stride should be 1, but received %d.",
              strides_indices.size()));

      auto* d_input_var = context.InputVar(framework::GradVarName("Out"));

      PADDLE_ENFORCE_EQ(
          d_input_var->IsType<LoDTensorArray>(), true,
          platform::errors::InvalidArgument(
              "When the output of `strided_slice_grad_op` is "
              "`TensorArray`, the input is excepted `TensorArray` , "
              "but received %s.",
              framework::ToTypeName(d_input_var->Type())));

      PADDLE_ENFORCE_EQ(
          out_dims.size(), 1,
          platform::errors::InvalidArgument(
              "When the output of `strided_slice_grad_op` is `TensorArray`, "
              "the dimension of output should be 1, but received %d.",
              out_dims.size()));
      auto& d_in_array = d_input_var->Get<framework::LoDTensorArray>();

      auto* d_out_array = context.Output<framework::LoDTensorArray>(
          framework::GradVarName("Input"));

      d_out_array->resize(out_dims[0]);
      auto const d_out_array_size = d_out_array->size();
      auto* input_tensor_array =
          context.Input<framework::LoDTensorArray>("Input");

      for (size_t j = 0; j < d_out_array_size; j++) {
        auto& dim = input_tensor_array->at(j).dims();
        auto* d_out_tensor = &d_out_array->at(j);

        int64_t sub = j - starts_indices[0];

        int64_t in_offset = sub / strides_indices[0];

        if (need_reverse) {
          in_offset = d_in_array.size() - in_offset - 1;
        }

        if ((sub % strides_indices[0] == 0) && (0 <= in_offset) &&
            (static_cast<size_t>(in_offset) < d_in_array.size())) {
          auto& in_tensor = d_in_array.at(in_offset);
          PADDLE_ENFORCE_GT(
              in_tensor.memory_size(), 0,
              platform::errors::PreconditionNotMet(
                  "The input LoDTensorArray Input[%d] holds no memory.",
                  in_offset));

          d_out_tensor->set_lod(in_tensor.lod());
          paddle::framework::TensorCopy(in_tensor, context.GetPlace(),
                                        d_out_tensor);

        } else {
          d_out_tensor->Resize(dim);

          if (!d_out_tensor->IsInitialized()) {
            d_out_tensor->mutable_data<T>(context.GetPlace());
          }

          math::SetConstant<DeviceContext, T> set_zero;
          set_zero(dev_ctx, d_out_tensor, static_cast<T>(0));
        }
      }

    } else {
      auto* d_input =
          context.Input<framework::Tensor>(framework::GradVarName("Out"));
      auto* d_out =
          context.Output<framework::Tensor>(framework::GradVarName("Input"));

      d_out->mutable_data<T>(context.GetPlace());

      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(dev_ctx, d_out, static_cast<T>(0));

      auto in_dims = d_input->dims();

      auto in_t = framework::EigenTensor<T, D, Eigen::RowMajor,
                                         Eigen::DenseIndex>::From(*d_input);
      auto out_t =
          framework::EigenTensor<T, D, Eigen::RowMajor,
                                 Eigen::DenseIndex>::From(*d_out, out_dims);
      if (need_reverse) {
        framework::Tensor reverse_input;
        reverse_input.mutable_data<T>(in_dims, context.GetPlace());
        auto reverse_in_t =
            framework::EigenTensor<T, D, Eigen::RowMajor,
                                   Eigen::DenseIndex>::From(reverse_input);

        reverse_in_t.device(place) = in_t.reverse(reverse_axis);
        out_t.stridedSlice(starts_indices, ends_indices, strides_indices)
            .device(place) = reverse_in_t;
      } else {
        out_t.stridedSlice(starts_indices, ends_indices, strides_indices)
            .device(place) = in_t;
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
