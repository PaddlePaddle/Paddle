/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/slice_op.h"
#include "paddle/phi/kernels/funcs/strided_slice.h"

namespace paddle {
namespace operators {

static void ProcessStridedSliceParams(
    const std::vector<int>& axes,
    const DDim& input_dims,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& strides,
    const std::vector<int>& infer_flags,
    const std::vector<int>& decrease_axis,
    std::vector<int>* starts_indices_vector,
    std::vector<int>* ends_indices_vector,
    std::vector<int>* strides_indices_vector) {
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int64_t start = starts[axis];
    int64_t end = ends[axis];
    int64_t stride = strides[axis];

    int axis_index = axes[axis];
    int64_t dim_size = input_dims[axis_index];

    bool decrease_axis_affect = false;
    if (start == -1 && end == 0 && infer_flags[axis] == -1) {
      auto ret =
          std::find(decrease_axis.begin(), decrease_axis.end(), axis_index);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }

    if (stride < 0) {
      if (start < 0) {
        start = std::max(start, -dim_size);
      } else {
        start = std::min(start, dim_size - 1) - dim_size;
      }
      if (end < 0) {
        end = std::max(end, -dim_size - 1);
      } else {
        end = end - dim_size;
      }
    } else {
      if (start < 0) {
        start = std::max(start, -dim_size) + dim_size;
      } else {
        start = std::min(start, dim_size - 1);
      }
      if (end < 0) {
        end = end + dim_size;
      } else {
        end = std::min(end, dim_size);
      }
    }

    if (decrease_axis_affect) {
      if (stride < 0) {
        end = start - 1;
      } else {
        end = start + 1;
      }
    }

    (*starts_indices_vector)[axis_index] = static_cast<int>(start);
    (*ends_indices_vector)[axis_index] = static_cast<int>(end);
    (*strides_indices_vector)[axis_index] = static_cast<int>(stride);
  }
}

template <typename T>
class StridedSliceMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    PADDLE_ENFORCE_EQ(is_tensor_array,
                      false,
                      platform::errors::InvalidArgument(
                          "Tensor array as input is not supported."));
    int rank = ctx.Input<phi::DenseTensor>("Input")->dims().size();
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
      case 7:
        StridedSliceCompute<7>(ctx);
        break;
      case 8:
        StridedSliceCompute<8>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input is supported up to 8."));
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceCompute(const framework::ExecutionContext& ctx) const {
    auto place = ctx.GetPlace();

    auto in = ctx.Input<phi::DenseTensor>("Input");
    auto out = ctx.Output<phi::DenseTensor>("Out");
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
    auto list_new_starts_tensor =
        ctx.MultiInput<phi::DenseTensor>("StartsTensorList");
    auto list_new_ends_tensor =
        ctx.MultiInput<phi::DenseTensor>("EndsTensorList");
    auto list_new_strides_tensor =
        ctx.MultiInput<phi::DenseTensor>("StridesTensorList");

    // Tensor<int32>
    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (ctx.HasInput("StartsTensor")) {
      auto* starts_tensor = ctx.Input<phi::DenseTensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (ctx.HasInput("EndsTensor")) {
      auto* ends_tensor = ctx.Input<phi::DenseTensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_strides_tensor.size() > 0) {
      strides = GetDataFromTensorList<int64_t>(list_new_strides_tensor);
    } else if (ctx.HasInput("StridesTensor")) {
      auto* strides_tensor = ctx.Input<phi::DenseTensor>("StridesTensor");
      strides = GetDataFromTensor<int64_t>(strides_tensor);
    }

    // out dims calculation
    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    phi::funcs::StridedSliceOutDims(starts,
                                    ends,
                                    strides,
                                    axes,
                                    infer_flags,
                                    in_dims,
                                    decrease_axis,
                                    out_dims_vector.data(),
                                    axes.size(),
                                    false);
    framework::DDim out_dims(phi::make_ddim(out_dims_vector));

    // construct the starts_indices, ends_indices and strides_indices tensor for
    // calling StridedSlice op
    std::vector<int> starts_indices_vector(D, 0);
    std::vector<int> ends_indices_vector(out_dims_vector.begin(),
                                         out_dims_vector.end());
    std::vector<int> strides_indices_vector(D, 1);

    ProcessStridedSliceParams(axes,
                              in_dims,
                              starts,
                              ends,
                              strides,
                              infer_flags,
                              decrease_axis,
                              &starts_indices_vector,
                              &ends_indices_vector,
                              &strides_indices_vector);

    auto out_dims_origin = out_dims;
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            out_dims[decrease_axis[i]],
            1,
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
      out_dims_origin = phi::make_ddim(new_out_shape);
    }

    out->Resize(out_dims_origin);
    out->mutable_data<T>(place);

    MLUCnnlTensorDesc in_desc(*in);
    MLUCnnlTensorDesc out_desc(
        out_dims_vector.size(), out_dims_vector.data(), ToCnnlDataType<T>());
    MLUCnnl::StridedSlice(ctx,
                          starts_indices_vector.data(),
                          ends_indices_vector.data(),
                          strides_indices_vector.data(),
                          in_desc.get(),
                          GetBasePtr(in),
                          out_desc.get(),
                          GetBasePtr(out));
  }
};

template <typename T>
class StridedSliceGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    PADDLE_ENFORCE_EQ(is_tensor_array,
                      false,
                      platform::errors::InvalidArgument(
                          "Tensor array as input is not supported."));
    int rank = ctx.Input<phi::DenseTensor>("Input")->dims().size();

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
      case 7:
        StridedSliceGradCompute<7>(ctx);
        break;
      case 8:
        StridedSliceGradCompute<8>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input is supported up to 8."));
        break;
    }
  }

 private:
  template <size_t D>
  void StridedSliceGradCompute(const framework::ExecutionContext& ctx) const {
    auto place = ctx.GetPlace();

    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto input_dims = input->dims();
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
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
        ctx.MultiInput<phi::DenseTensor>("EndsTensorList");
    auto list_new_starts_tensor =
        ctx.MultiInput<phi::DenseTensor>("StartsTensorList");
    auto list_new_strides_tensor =
        ctx.MultiInput<phi::DenseTensor>("StridesTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
    } else if (ctx.HasInput("StartsTensor")) {
      auto* starts_tensor = ctx.Input<phi::DenseTensor>("StartsTensor");
      starts = GetDataFromTensor<int64_t>(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
    } else if (ctx.HasInput("EndsTensor")) {
      auto* ends_tensor = ctx.Input<phi::DenseTensor>("EndsTensor");
      ends = GetDataFromTensor<int64_t>(ends_tensor);
    }

    if (list_new_strides_tensor.size() > 0) {
      strides = GetDataFromTensorList<int64_t>(list_new_strides_tensor);
    } else if (ctx.HasInput("StridesTensor")) {
      auto* strides_tensor = ctx.Input<phi::DenseTensor>("StridesTensor");
      strides = GetDataFromTensor<int64_t>(strides_tensor);
    }

    std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
    phi::funcs::StridedSliceOutDims(starts,
                                    ends,
                                    strides,
                                    axes,
                                    infer_flags,
                                    input_dims,
                                    decrease_axis,
                                    out_dims_vector.data(),
                                    axes.size(),
                                    false);

    std::vector<int> starts_indices_vector(D, 0);
    std::vector<int> ends_indices_vector(out_dims_vector.begin(),
                                         out_dims_vector.end());
    std::vector<int> strides_indices_vector(D, 1);

    ProcessStridedSliceParams(axes,
                              input_dims,
                              starts,
                              ends,
                              strides,
                              infer_flags,
                              decrease_axis,
                              &starts_indices_vector,
                              &ends_indices_vector,
                              &strides_indices_vector);

    MLUCnnlTensorDesc dout_desc(
        out_dims_vector.size(), out_dims_vector.data(), ToCnnlDataType<T>());
    MLUCnnlTensorDesc dx_desc(*input);
    MLUCnnl::StridedSliceGrad(ctx,
                              starts_indices_vector.data(),
                              ends_indices_vector.data(),
                              strides_indices_vector.data(),
                              dout_desc.get(),
                              GetBasePtr(dout),
                              dx_desc.get(),
                              GetBasePtr(dx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(strided_slice,
                       ops::StridedSliceMLUKernel<plat::float16>,
                       ops::StridedSliceMLUKernel<bool>,
                       ops::StridedSliceMLUKernel<int>,
                       ops::StridedSliceMLUKernel<int64_t>,
                       ops::StridedSliceMLUKernel<float>);

REGISTER_OP_MLU_KERNEL(strided_slice_grad,
                       ops::StridedSliceGradMLUKernel<plat::float16>,
                       ops::StridedSliceGradMLUKernel<float>,
                       ops::StridedSliceGradMLUKernel<bool>,
                       ops::StridedSliceGradMLUKernel<int>,
                       ops::StridedSliceGradMLUKernel<int64_t>);
