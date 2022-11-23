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
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class SliceMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

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

    const auto& in_dims = input->dims();
    auto slice_dims = out->dims();
    bool reset_slice_dims = false;
    if (ctx.HasInput("StartsTensor") || ctx.HasInput("EndsTensor") ||
        starts_tensor_list.size() > 0 || ends_tensor_list.size() > 0) {
      // Infer output dims
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
      reset_slice_dims = true;
      auto out_dims = phi::funcs::GetDecreasedDims(slice_dims, decrease_axis);

      out->Resize(out_dims);
    }
    if (slice_dims.size() != in_dims.size() && !reset_slice_dims) {
      phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims = phi::funcs::GetSliceDims<int>(
          in_dims, axes, starts, ends, nullptr, nullptr);
    }

    int in_dim_size = input->dims().size();
    if (static_cast<int>(axes.size()) != in_dim_size) {
      std::vector<int> tmp_starts(in_dim_size, 0);
      const auto& in_dims_vec = phi::vectorize(input->dims());
      std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
      for (size_t i = 0; i < axes.size(); ++i) {
        tmp_starts[axes[i]] = starts[i];
        tmp_ends[axes[i]] = ends[i];
      }
      starts.swap(tmp_starts);
      ends.swap(tmp_ends);
    }
    std::vector<int> strides(in_dim_size, 1);

    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc out_desc(slice_dims.size(),
                               phi::vectorize(slice_dims).data(),
                               ToCnnlDataType<T>());
    MLUCnnl::StridedSlice(ctx,
                          starts.data(),
                          ends.data(),
                          strides.data(),
                          input_desc.get(),
                          GetBasePtr(input),
                          out_desc.get(),
                          GetBasePtr(out));
  }
};

template <typename T>
class SliceGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dinput =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));

    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts = ctx.Attr<std::vector<int>>("starts");
    auto ends = ctx.Attr<std::vector<int>>("ends");

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
    auto slice_dims = dout->dims();
    if (slice_dims.size() != in_dims.size()) {
      phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims = phi::funcs::GetSliceDims<int>(
          in_dims, axes, starts, ends, nullptr, nullptr);
    }

    int in_dim_size = input->dims().size();
    if (static_cast<int>(axes.size()) != in_dim_size) {
      std::vector<int> tmp_starts(in_dim_size, 0);
      const auto& in_dims_vec = phi::vectorize(input->dims());
      std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
      for (size_t i = 0; i < axes.size(); ++i) {
        tmp_starts[axes[i]] = starts[i];
        tmp_ends[axes[i]] = ends[i];
      }
      starts.swap(tmp_starts);
      ends.swap(tmp_ends);
    }
    std::vector<int> strides(in_dim_size, 1);

    dinput->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc dout_desc(slice_dims.size(),
                                phi::vectorize(slice_dims).data(),
                                ToCnnlDataType<T>());
    MLUCnnlTensorDesc dinput_desc(*dinput);
    MLUCnnl::StridedSliceGrad(ctx,
                              starts.data(),
                              ends.data(),
                              strides.data(),
                              dout_desc.get(),
                              GetBasePtr(dout),
                              dinput_desc.get(),
                              GetBasePtr(dinput));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(slice,
                       ops::SliceMLUKernel<float>,
                       ops::SliceMLUKernel<int>,
                       ops::SliceMLUKernel<bool>,
                       ops::SliceMLUKernel<int64_t>,
                       ops::SliceMLUKernel<double>,
                       ops::SliceMLUKernel<paddle::platform::float16>);

REGISTER_OP_MLU_KERNEL(slice_grad,
                       ops::SliceGradMLUKernel<float>,
                       ops::SliceGradMLUKernel<int>,
                       ops::SliceGradMLUKernel<bool>,
                       ops::SliceGradMLUKernel<int64_t>,
                       ops::SliceGradMLUKernel<paddle::platform::float16>);
