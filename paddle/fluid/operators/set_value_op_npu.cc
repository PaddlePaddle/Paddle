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

#include "paddle/fluid/operators/set_value_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using NPUDeviceContext = platform::NPUDeviceContext;

template <typename T>
class SetValueNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<Tensor>("Input");
    auto* value_tensor = ctx.Input<Tensor>("ValueTensor");
    auto* out = ctx.Output<Tensor>("Out");

    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    auto steps_tensor_list = ctx.MultiInput<Tensor>("StepsTensorList");

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto decrease_axes = ctx.Attr<std::vector<int64_t>>("decrease_axes");
    auto none_axes = ctx.Attr<std::vector<int64_t>>("none_axes");

    if (!starts_tensor_list.empty()) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }
    if (!ends_tensor_list.empty()) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }
    if (!steps_tensor_list.empty()) {
      steps = GetDataFromTensorList<int64_t>(steps_tensor_list);
    }

    auto in_dims = in->dims();
    CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends, &steps);
    auto decrease_slice_dims = GetDecreasedDims(slice_dims, decrease_axes);

    auto slice_dims_for_assign = decrease_slice_dims;
    if (!none_axes.empty()) {
      std::vector<int64_t> slice_dims_with_none;

      size_t none_axes_cur = 0, decrease_axes_cur = 0;
      for (int i = 0; i < slice_dims.size(); ++i) {
        while (none_axes_cur < none_axes.size() &&
               none_axes[none_axes_cur] <= i) {
          slice_dims_with_none.push_back(1);
          none_axes_cur++;
        }
        if (decrease_axes_cur < decrease_axes.size() &&
            decrease_axes[decrease_axes_cur] == i) {
          decrease_axes_cur++;
        } else {
          slice_dims_with_none.push_back(slice_dims[i]);
        }
      }
      while (none_axes_cur < none_axes.size()) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }

      slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
    }

    paddle::framework::TensorCopy(*in, ctx.GetPlace(), out);

    auto starts_indices = std::vector<int64_t>(in_dims.size(), 0);
    auto ends_indices = std::vector<int64_t>(in_dims.size(), 0);
    auto strides_indices = std::vector<int64_t>(in_dims.size(), 0);

    for (int i = 0; i < in_dims.size(); ++i) {
      starts_indices[i] = 0;
      ends_indices[i] = slice_dims[i];
      strides_indices[i] = 1;
    }
    for (size_t i = 0; i < axes.size(); i++) {
      int axis_index = axes[i];
      starts_indices[axis_index] = starts[i];
      ends_indices[axis_index] = ends[i];
      strides_indices[axis_index] = steps[i];
    }

    int64_t stride_step = phi::product(in_dims);
    std::vector<int64_t> index_indices(1, 0);
    for (size_t i = 0; i < strides_indices.size(); ++i) {
      auto index_size = index_indices.size();
      stride_step /= in_dims[i];
      for (size_t j = 0; j < index_size; ++j) {
        auto start_index = *index_indices.begin();
        if (strides_indices[i] > 0) {
          for (int64_t k = starts_indices[i]; k < ends_indices[i];
               k += strides_indices[i]) {
            index_indices.push_back(start_index + k * stride_step);
          }
        } else {
          for (int64_t k = starts_indices[i]; k > ends_indices[i];
               k += strides_indices[i]) {
            index_indices.push_back(start_index + k * stride_step);
          }
        }
        index_indices.erase(index_indices.begin());
      }
    }

    PADDLE_ENFORCE_EQ(
        static_cast<int64_t>(index_indices.size()),
        phi::product(slice_dims_for_assign),
        platform::errors::InvalidArgument(
            "OP(set_value) error index indices and value update not match "));

    Tensor value_t(in->type());
    if (value_tensor != nullptr) {
      value_t.ShareDataWith(*value_tensor);
    } else {
      auto value_dims = phi::make_ddim(shape);
      CheckIsDimsMatch(slice_dims_for_assign, value_dims);

      value_t.mutable_data<T>(value_dims, ctx.GetPlace());
      auto value_name =
          GetValueName(framework::TransToProtoVarType(in->dtype()));
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
    }

    auto stream = ctx.template device_context<NPUDeviceContext>().stream();

    Tensor value_temp(in->type());
    if (slice_dims_for_assign == value_t.dims()) {
      value_temp.ShareDataWith(value_t);
    } else {
      value_temp.Resize(slice_dims_for_assign);
      value_temp.mutable_data<T>(ctx.GetPlace());
      NpuOpRunner runner_brd;
      runner_brd.SetType("BroadcastTo")
          .AddInput(value_t)
          .AddInput(phi::vectorize(slice_dims_for_assign))
          .AddOutput(value_temp)
          .Run(stream);
    }

    int64_t input_numel = phi::product(in_dims);
    int64_t index_numel = index_indices.size();

    Tensor in_temp, out_temp, val_temp;
    in_temp.ShareDataWith(*in);
    out_temp.ShareDataWith(*out);
    val_temp.ShareDataWith(value_temp);
    in_temp.Resize(phi::make_ddim({input_numel}));
    out_temp.Resize(phi::make_ddim({input_numel}));
    val_temp.Resize(phi::make_ddim({index_numel}));

    NpuOpRunner runner;
    runner.SetType("ScatterUpdate")
        .AddInput(in_temp)
        .AddInput(std::move(index_indices))
        .AddInput(val_temp)
        .AddOutput(out_temp)
#if (CANN_VERSION_CODE >= 504001)
        .AddAttrs({{"use_locking", false}})
#endif
        .Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(set_value, ops::SetValueNPUKernel<int>,
#ifdef PADDLE_WITH_ASCEND_INT64
                       ops::SetValueNPUKernel<int64_t>,
#endif
                       ops::SetValueNPUKernel<float>)
