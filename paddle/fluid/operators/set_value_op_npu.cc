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
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/mean_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class SetValueNPUKernel : public framework::OpKernel<T> {
 private:
  using vec64 = std::vector<int64_t>;
  using vec32 = std::vector<int32_t>;
  using vec_vec64 = std::vector<std::vector<int64_t>>;
  inline void GetNPUStartEndSteps(vec_vec64& output, vec64& start, vec64& end,
                                  vec64& steps, vec64& axes,
                                  framework::DDim in_dim) const {
    int rank = in_dim.size();
    for (int i = 0; i < rank; ++i) {
      int axis_size = in_dim[i];
      auto iter = find(axes.begin(), axes.end(), i);
      if (iter != axes.end()) {
        // find
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
      vec_vec64& npu_slice) const {
    int min_value = 32 / sizeof(T);  // 16 for float16 , 8 for float32
    int rank = npu_slice[0].size();
    int last_dim_start = npu_slice[0][rank - 1];
    int last_dim_end = npu_slice[1][rank - 1];
    int last_dim_step = npu_slice[2][rank - 1];
    int min_end = last_dim_start + last_dim_step * min_value;
    int raw_last_dim_len = (last_dim_end - last_dim_start) / last_dim_step;
    return std::vector<int>(
        {std::max(0, min_end - last_dim_end), min_value - raw_last_dim_len});
  }

  inline void UnsqueezeLastDim(const framework::ExecutionContext* ctx,
                               Tensor* output, const Tensor* input) const {
    return;
    VLOG(4) << "Start Unsqueeze tensor" << std::endl;
    auto after_reshape_dims = framework::vectorize<int>(input->dims());
    after_reshape_dims.push_back(1);
    output->mutable_data<T>(framework::make_ddim(after_reshape_dims),
                            ctx->GetPlace());
    framework::NPUAttributeMap attr;
    attr["axes"] = static_cast<int>(input->dims().size());
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("Unsqueeze", {*input}, {*output}, attr).Run(stream);
  }

  inline void TileTensor(const framework::ExecutionContext* ctx, Tensor* output,
                         const Tensor* input, int pad_number) const {
    VLOG(4) << "start Tile tensor function, which call the NPU operator PadD"
            << std::endl;
    // UNSQUEEZE last dim + TILE last dim * 8
    Tensor after_reshape_tensor;
    auto after_reshape_dims = framework::vectorize<int>(input->dims());
    after_reshape_dims.push_back(1);
    after_reshape_tensor.ShareDataWith(*input);
    after_reshape_tensor.Resize(framework::make_ddim(after_reshape_dims));
    // UnsqueezeLastDim(ctx, &after_reshape_tensor, input) ;

    auto output_dims = framework::vectorize<int>(input->dims());
    output_dims.push_back(8);
    output->mutable_data<T>(framework::make_ddim(output_dims), ctx->GetPlace());

    framework::NPUAttributeMap attr;
    attr["axis"] = static_cast<int>(after_reshape_dims.size() - 1);
    attr["tiles"] = 8;
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("TileWithAxis", {after_reshape_tensor}, {*output}, attr)
        .Run(stream);
  }

  inline void BroadcastToD(const framework::ExecutionContext* ctx,
                           Tensor* output, const Tensor* input,
                           std::vector<int64_t> shape) const {
    VLOG(4) << "Start BroadCast To" << std::endl;
    vec32 shape32 = vec32(shape.begin(), shape.end());
    output->mutable_data<T>(framework::make_ddim(shape32), ctx->GetPlace());
    framework::NPUAttributeMap attr;
    attr["shape"] = shape32;
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    NpuOpRunner("BroadcastToD", {*input}, {*output}, attr).Run(stream);
  }

  inline void CropTensor(const framework::ExecutionContext* ctx, Tensor* output,
                         const Tensor* input) const {
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

  inline void SliceAssignNPU(const framework::ExecutionContext* ctx,
                             Tensor* lefthand, const Tensor* righthand,
                             vec64& start, vec64& end, vec64& steps,
                             vec64& axes) const {
    // must ensure lefthand and righthand have the same shape
    // not support steps < 0
    // output is also the lefthand.
    VLOG(4) << "start function SliceAssignND" << std::endl;
    auto stream =
        ctx->template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    for (long unsigned int i = 0; i < steps.size(); ++i) {
      PADDLE_ENFORCE_GT(
          steps[i], 0,
          platform::errors::InvalidArgument(
              "currently NPU set_value operator don't support negative steps"));
    }
    vec_vec64 npu_slice(3);
    GetNPUStartEndSteps(npu_slice, start, end, steps, axes, lefthand->dims());
    auto tile_numbers = MininumPadNumberMakeSureLastDimGT8(npu_slice);
    int lefthand_tile_number = tile_numbers[0];
    int righthand_tile_number = tile_numbers[1];

    VLOG(4) << "tile number is : " << lefthand_tile_number << " "
            << righthand_tile_number << std::endl;

    Tensor tiled_left, tiled_right;
    if (lefthand_tile_number > 0) {
      TileTensor(ctx, &tiled_left, lefthand, 0);
      TileTensor(ctx, &tiled_right, righthand, 0);
      // output have different shape, so use a tmp variable before_crop_output;
      // add last dim = 8 in slice
      npu_slice[0].push_back(0);
      npu_slice[1].push_back(8);
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
    if (lefthand_tile_number > 0) {
      NpuOpRunner("StridedSliceAssignD", {tiled_left, tiled_right},
                  {tiled_left}, attr_input)
          .Run(stream);  // Remember, set output = input, and this op will
                         // change the input value.
    } else {
      NpuOpRunner("StridedSliceAssignD", {*lefthand, *righthand}, {*lefthand},
                  attr_input)
          .Run(stream);
    }
    if (lefthand_tile_number > 0) {
      CropTensor(ctx, lefthand /*initalized*/, &tiled_left /*initialzied*/);
    }
  }

  inline void ModifyAxesAccordingNoneAxes(const vec64& none_axes,
                                          vec64& axes_to_modify) const {
    if (!none_axes.size()) return;
    auto none_axes_copy = none_axes;
    sort(none_axes_copy.begin(), none_axes_copy.end());
    for (unsigned int i = 0; i < axes_to_modify.size(); ++i) {
      int axis = axes_to_modify[i];
      auto upper =
          upper_bound(none_axes_copy.begin(), none_axes_copy.end(), axis);
      // Example: none_axes = [1,3,4,5,7]
      // 	    axis = 4
      //          find the element number less or equal than 4, which is
      //          3(1,3,4)
      //          axis becomes  4 + 3 = 7 ;
      axes_to_modify[i] = axis + (upper - none_axes_copy.begin());
    }
  }

  inline void UnsqueezeAccordingNoneAxes(vec64& slice_dims,
                                         vec64& none_axes) const {
    // note : axes will change, because new axes inserted.
    // sum array to modify the axes. because more simply
    if (!none_axes.size()) return;
    vec64 slice_dims_with_none;
    size_t none_axes_cur = 0;
    for (unsigned int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
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

  inline void ModiftyDimsAccordingNoneAndDecrease(vec64& slice_dim,
                                                  vec64& value_dim, vec64& axes,
                                                  vec64& none_axes,
                                                  vec64& dec_axes) const {
    // change the value of slice_dim, value_dim, start, end, steps, axes by none
    // and decrease axes
    // after change, this values can be passed to SliceAssignNPU() directly.

    // Modity Slice Dim
    UnsqueezeAccordingNoneAxes(slice_dim, none_axes);
    ModifyAxesAccordingNoneAxes(none_axes, dec_axes);
    ModifyAxesAccordingNoneAxes(none_axes, axes);
    // Modity Value Dim by new slice dim
    vec64 slice_dim_reverse = slice_dim;
    vec64 value_dim_reverse = value_dim;
    std::reverse(slice_dim_reverse.begin(), slice_dim_reverse.end());
    std::reverse(value_dim_reverse.begin(), value_dim_reverse.end());

    vec64 new_value_dim;
    PADDLE_ENFORCE_GE(
        slice_dim.size(), value_dim.size(),
        platform::errors::InvalidArgument("The size of expanded slice_dim(%d) "
                                          "must greater than the value_dim(%d)",
                                          slice_dim.size(), value_dim.size()));

    size_t value_cur = 0;
    int rank = slice_dim.size();
    for (int i = 0; i < rank; ++i) {
      auto& xsize = slice_dim_reverse[i];
      if (value_cur >= value_dim_reverse.size()) {
        new_value_dim.push_back(1);
        continue;
      }
      auto& vsize = value_dim_reverse[value_cur];
      auto it = find(dec_axes.begin(), dec_axes.end(), rank - 1 - i);
      if (it != dec_axes.end()) {
        // found, insert one dim ;
        PADDLE_ENFORCE_EQ(xsize, 1,
                          platform::errors::InvalidArgument("Bugs Here!"));
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
    VLOG(2) << "Start Set Value Npu Kernel" << std::endl;
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
    // step slice assignment. so we deal all none_axes and decrease_axes here.
    // 1. we insert 1 into lefthand_shape according to none_axes;
    // 2. we insert 1 into righthand_shape(value tensor) according to
    // decrease_axes;
    // 3. we reshape back the lefthand. and return it.
    // note : we use a tmp_value_tensor as right hand. it shares data with
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

    UnsqueezeAccordingNoneAxes(in_dims_vec, none_axes);
    ModiftyDimsAccordingNoneAndDecrease(slice_dims_vec, value_dims_vec, axes,
                                        none_axes,
                                        decrease_axes);  // Modify and Check

    Tensor reshaped_value_tensor, broadcast_value_tensor;
    reshaped_value_tensor.ShareDataWith(*value_tensor_ptr);
    reshaped_value_tensor.Resize(framework::make_ddim(value_dims_vec));

    BroadcastToD(&ctx, &broadcast_value_tensor /*inner function initialized*/,
                 &reshaped_value_tensor, slice_dims_vec);

    out->Resize(framework::make_ddim(in_dims_vec));
    SliceAssignNPU(&ctx, out, &broadcast_value_tensor, starts, ends, steps,
                   axes);
    out->Resize(in_dims);  // Reshape Back
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    set_value, ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SetValueNPUKernel<paddle::platform::NPUDeviceContext, float>)
