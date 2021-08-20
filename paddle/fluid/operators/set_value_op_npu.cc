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
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class SetValueNPUKernel : public framework::OpKernel<T> {
 private:
  using Vector_Int64 = std::vector<int64_t>;
  void GetNPUStartEndSteps(const Vector_Int64& start, const Vector_Int64& end,
                           const Vector_Int64& steps, const Vector_Int64& axes,
                           const framework::DDim& in_dim,
                           std::vector<std::vector<int64_t>>& output) const {
    int rank = in_dim.size();
    for (int i = 0; i < rank; ++i) {
      int axis_size = in_dim[i];
      auto iter = find(axes.begin(), axes.end(), i);
      if (iter != axes.end()) {
        int idx = iter - axes.begin();
        output[0].push_back(start[idx]);  // set as the same as raw input
        output[1].push_back(end[idx]);
        output[2].push_back(steps[idx]);
      } else {
        output[0].push_back(0);          // begin 0
        output[1].push_back(axis_size);  // end = last one
        output[2].push_back(1);          // step = 1
      }
    }
  }

  inline std::vector<int> MininumPadNumberMakeSureLastDimGT8(
      const std::vector<std::vector<int64_t>>& npu_slice) const {
    int rank = npu_slice[0].size();
    int last_dim_start = npu_slice[0][rank - 1];
    int last_dim_end = npu_slice[1][rank - 1];
    int last_dim_step = npu_slice[2][rank - 1];
    int min_end = last_dim_start + last_dim_step * min_last_dim_value_;
    int raw_last_dim_len = (last_dim_end - last_dim_start) / last_dim_step;
    return std::vector<int>({std::max(0, min_end - last_dim_end),
                             min_last_dim_value_ - raw_last_dim_len});
  }

  inline void TileTensor(const framework::ExecutionContext* ctx,
                         const Tensor* input, Tensor* output) const {
    VLOG(4) << "start to tile tensor function, which calls the npu operator "
               "TileWithAxis";
    // UNSQUEEZE last dim + TILE last dim * min_last_dim_value_
    Tensor reshape_tensor;
    auto reshape_dims = framework::vectorize<int>(input->dims());
    reshape_dims.push_back(1);
    reshape_tensor.ShareDataWith(*input);
    reshape_tensor.Resize(framework::make_ddim(reshape_dims));

    auto output_dims = framework::vectorize<int>(input->dims());
    output_dims.push_back(min_last_dim_value_);
    output->mutable_data<T>(framework::make_ddim(output_dims), ctx->GetPlace());

    framework::NPUAttributeMap attr;
    attr["axis"] = static_cast<int>(reshape_dims.size() - 1);
    attr["tiles"] = min_last_dim_value_;
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("TileWithAxis", {reshape_tensor}, {*output}, attr).Run(stream);
  }

  inline void BroadcastToD(const framework::ExecutionContext* ctx,
                           const Tensor* input,
                           const std::vector<int64_t>* shape,
                           Tensor* output) const {
    VLOG(4) << "Start BroadCast To";
    auto new_shape = std::vector<int32_t>(shape->begin(), shape->end());
    output->mutable_data<T>(framework::make_ddim(new_shape), ctx->GetPlace());
    framework::NPUAttributeMap attr;
    attr["shape"] = new_shape;
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("BroadcastToD", {*input}, {*output}, attr).Run(stream);
  }

  inline void CropTensor(const framework::ExecutionContext* ctx,
                         const Tensor* input, Tensor* output) const {
    auto out_dims = output->dims();
    auto in_dims = input->dims();
    int rank = in_dims.size();
    in_dims[rank - 1] = 1;
    output->Resize(in_dims);  // unsqueeze output -> [..., 1]
    framework::NPUAttributeMap attr;
    attr["axis"] = 0;
    attr["offsets"] = std::vector<int>(rank, 0);
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("Crop", {*input, *output}, {*output}, attr).Run(stream);
    output->Resize(out_dims);  // restore it
  }

  void SliceAssignNPU(const framework::ExecutionContext* ctx,
                      const Tensor* value_tensor, Vector_Int64& start,
                      Vector_Int64& end, Vector_Int64& steps,
                      Vector_Int64& axes, Tensor* assigned_tensor) const {
    // must ensure assigned_tensor and value_tensor have the same shape
    // not support steps < 0
    // output is also the assigned_tensor.
    VLOG(4) << "start function SliceAssignND";
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    for (size_t i = 0; i < steps.size(); ++i) {
      PADDLE_ENFORCE_GT(steps[i], 0,
                        platform::errors::InvalidArgument(
                            "Currently NPU set_value operator doesn't support "
                            "negative steps, but got %d as step",
                            steps[i]));
    }
    std::vector<std::vector<int64_t>> npu_slice(3);
    GetNPUStartEndSteps(start, end, steps, axes, assigned_tensor->dims(),
                        npu_slice);
    auto tile_numbers = MininumPadNumberMakeSureLastDimGT8(npu_slice);
    int assigned_tensor_tile_number = tile_numbers[0];
    int value_tensor_tile_number = tile_numbers[1];

    VLOG(4) << "tile number is : " << assigned_tensor_tile_number << " "
            << value_tensor_tile_number;

    Tensor tiled_assigned_tns, tiled_value_tns;
    if (assigned_tensor_tile_number > 0) {
      TileTensor(ctx, assigned_tensor, &tiled_assigned_tns);
      TileTensor(ctx, value_tensor, &tiled_value_tns);
      // output have different shape, so use a tmp variable before_crop_output;
      // add last dim = min_last_dim_value_ in slice
      npu_slice[0].push_back(0);
      npu_slice[1].push_back(min_last_dim_value_);
      npu_slice[2].push_back(1);
    }

    framework::NPUAttributeMap attr_input;
    attr_input["begin"] =
        std::vector<int>(npu_slice[0].begin(), npu_slice[0].end());
    attr_input["end"] =
        std::vector<int>(npu_slice[1].begin(), npu_slice[1].end());
    attr_input["strides"] =
        std::vector<int>(npu_slice[2].begin(), npu_slice[2].end());
    attr_input["begin_mask"] = 0;
    attr_input["end_mask"] = 0;
    attr_input["ellipsis_mask"] = 0;
    attr_input["new_axis_mask"] = 0;
    attr_input["shrink_axis_mask"] = 0;
    if (assigned_tensor_tile_number > 0) {
      NpuOpRunner("StridedSliceAssignD", {tiled_assigned_tns, tiled_value_tns},
                  {tiled_assigned_tns}, attr_input)
          .Run(stream);  // Remember, set output = input, and this op will
                         // change the input value.
    } else {
      NpuOpRunner("StridedSliceAssignD", {*assigned_tensor, *value_tensor},
                  {*assigned_tensor}, attr_input)
          .Run(stream);
    }
    if (assigned_tensor_tile_number > 0) {
      CropTensor(ctx, &tiled_assigned_tns /*initialzied*/,
                 assigned_tensor /*initalized*/);
    }
  }

  void ModifyAxesAccordingNoneAxes(const Vector_Int64& none_axes,
                                   Vector_Int64& axes_to_modify) const {
    if (none_axes.empty()) return;
    auto none_axes_copy = none_axes;
    sort(none_axes_copy.begin(), none_axes_copy.end());
    for (size_t i = 0; i < axes_to_modify.size(); ++i) {
      int axis = axes_to_modify[i];
      auto upper =
          upper_bound(none_axes_copy.begin(), none_axes_copy.end(), axis);
      // Example: none_axes = [1,3,4,5,7]
      //          axis = 4
      //          find the element number less or equal than 4, which is
      //          3(1,3,4)
      //          axis becomes  4 + 3 = 7 ;
      axes_to_modify[i] = axis + (upper - none_axes_copy.begin());
    }
  }

  void UnsqueezeAccordingNoneAxes(const Vector_Int64& none_axes,
                                  Vector_Int64& slice_dims) const {
    // note : axes will change, because new axes inserted.
    // sum array to modify the axes. because more simply
    if (none_axes.empty()) return;
    Vector_Int64 slice_dims_with_none;
    size_t none_axes_cur = 0;
    for (size_t i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= static_cast<int>(i)) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      slice_dims_with_none.push_back(slice_dims[i]);
    }
    // if the none_axes.size() > slice_dims.size(), append 1 after last dim
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }
    slice_dims = slice_dims_with_none;
  }

  void ModiftyDimsAccordingNoneAndDecrease(Vector_Int64& slice_dim,
                                           Vector_Int64& value_dim,
                                           Vector_Int64& axes,
                                           Vector_Int64& none_axes,
                                           Vector_Int64& dec_axes) const {
    // change the value of slice_dim, value_dim, start, end, steps, axes by none
    // and decrease axes
    // after change, this values can be passed to SliceAssignNPU() directly.

    // Modity Slice Dim
    UnsqueezeAccordingNoneAxes(none_axes, slice_dim);
    ModifyAxesAccordingNoneAxes(none_axes, dec_axes);
    ModifyAxesAccordingNoneAxes(none_axes, axes);
    // Modity Value Dim by new slice dim
    auto slice_dim_reverse = slice_dim;
    auto value_dim_reverse = value_dim;
    std::reverse(slice_dim_reverse.begin(), slice_dim_reverse.end());
    std::reverse(value_dim_reverse.begin(), value_dim_reverse.end());

    Vector_Int64 new_value_dim;
    PADDLE_ENFORCE_GE(
        slice_dim.size(), value_dim.size(),
        platform::errors::InvalidArgument("The size of expanded slice_dim(%d) "
                                          "must greater than the value_dim(%d)",
                                          slice_dim.size(), value_dim.size()));

    size_t value_cur = 0;
    size_t rank = slice_dim.size();
    for (size_t i = 0; i < rank; ++i) {
      auto& xsize = slice_dim_reverse[i];
      if (value_cur >= value_dim_reverse.size()) {
        new_value_dim.push_back(1);
        continue;
      }
      auto& vsize = value_dim_reverse[value_cur];
      auto it = find(dec_axes.begin(), dec_axes.end(), rank - 1 - i);
      if (it != dec_axes.end()) {
        // found, insert one dim ;
        PADDLE_ENFORCE_EQ(xsize, 1, platform::errors::InvalidArgument(
                                        "The dims refered by decrease axes is "
                                        "not equal to 1, some wrongs happen"));
        new_value_dim.push_back(1);
        continue;
      }
      if (xsize == vsize || vsize == 1) {
        new_value_dim.push_back(vsize);
        ++value_cur;
        continue;
      }
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The shape of value_tensor can't be broadcast to value tensor, "
          "please check input"));
    }
    for (; value_cur < value_dim_reverse.size(); ++value_cur) {
      if (value_dim_reverse[value_cur] != 1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The shape of value_tensor can't be broadcast to value tensor, "
            "please check input"));
      }
    }
    std::reverse(new_value_dim.begin(), new_value_dim.end());
    value_dim = new_value_dim;
    return;
  }

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(2) << "Start Set Value Npu Kernel";
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");
    auto starts_tensor_list =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto steps_tensor_list =
        ctx.MultiInput<framework::Tensor>("StepsTensorList");
    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto decrease_axes = ctx.Attr<std::vector<int64_t>>("decrease_axes");
    auto none_axes = ctx.Attr<std::vector<int64_t>>("none_axes");
    auto dtype = in->type();

    if (dtype == framework::proto::VarType::FP64 ||
        dtype == framework::proto::VarType::INT64 ||
        dtype == framework::proto::VarType::BOOL) {
      auto value_type_name = GetValueName(dtype);
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The NPU setvalue kernel currently only support FLOAT32 and INT32, "
          "but got type: %s",
          value_type_name.data()));
    }

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
    auto place = ctx.GetPlace();

    // aforementioned code is copyed directly from CPU kernel.
    // (@xiongkun03) the following is redesigned by xiongkun. because NPU can do
    // step slice assignment. so we deal with all none_axes and decrease_axes
    // here.
    // 1. we insert 1 into assigned_tensor_shape according to none_axes;
    // 2. we insert 1 into value_tensor_shape(value tensor) according to
    // decrease_axes;
    // 3. we reshape back the assigned_tensor. and return it.
    // note : we use a tmp_value_tensor as value_tns. it shares data with
    // value_tensor;
    // I believe the logic is more simple than cpu logic.

    TensorCopy(*in, place, out);
    Tensor value_t(dtype);

    if (value_tensor == nullptr) {
      auto value_dims = framework::make_ddim(shape);
      value_t.mutable_data<T>(value_dims, place);
      auto value_name = GetValueName(dtype);
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
    }

    const Tensor* value_tensor_ptr =
        (value_tensor == nullptr) ? &value_t : value_tensor;
    auto value_dims_vec = framework::vectorize(value_tensor_ptr->dims());
    auto slice_dims_vec = framework::vectorize(slice_dims);
    auto in_dims_vec = framework::vectorize(in_dims);

    UnsqueezeAccordingNoneAxes(none_axes, in_dims_vec);
    ModiftyDimsAccordingNoneAndDecrease(slice_dims_vec, value_dims_vec, axes,
                                        none_axes,
                                        decrease_axes);  // Modify and Check

    Tensor reshaped_value_tensor, broadcast_value_tensor;
    reshaped_value_tensor.ShareDataWith(*value_tensor_ptr);
    reshaped_value_tensor.Resize(framework::make_ddim(value_dims_vec));

    BroadcastToD(&ctx, &reshaped_value_tensor, &slice_dims_vec,
                 &broadcast_value_tensor /*inner function initialized*/);

    out->Resize(framework::make_ddim(in_dims_vec));
    SliceAssignNPU(&ctx, &broadcast_value_tensor, starts, ends, steps, axes,
                   out);
    out->Resize(in_dims);  // Reshape Back
  }

 private:
  const int min_last_dim_value_ =
      32 / sizeof(T);  // 16 for float16 , 8 for float32
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    set_value, ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, float>)
