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

#pragma once
#include <algorithm>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using Variable = framework::Variable;
using LoDTensorArray = framework::LoDTensorArray;
using DDim = framework::DDim;

inline void DealTensorArray(const framework::ExecutionContext& ctx,
                            const std::vector<int64_t>& starts,
                            const std::vector<int64_t>& ends,
                            bool out_is_array) {
  auto in_array = ctx.Input<LoDTensorArray>("Input");
  // If the input is LoDTensorArray, the rank of input is 1.
  int64_t in_size = in_array->size();
  int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  int64_t end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

  start = std::max(start, static_cast<int64_t>(0));
  end = std::max(end, static_cast<int64_t>(0));
  end = std::min(end, in_size);

  if (starts[0] == -1 && end == 0) {
    end = start + 1;
  }

  PADDLE_ENFORCE_GT(end, start,
                    platform::errors::InvalidArgument(
                        "Attr(ends) should be greater than attr(starts) in "
                        "slice op. But received end = %d, start = %d.",
                        ends[0], starts[0]));
  int64_t out_size = end - start;

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
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    Variable* out_var = ctx.OutputVar("Out");
    bool input_is_array = input_var->IsType<LoDTensorArray>();
    bool out_is_array = out_var->IsType<LoDTensorArray>();

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int64_t> axes(axes_int.begin(), axes_int.end());
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

    // Step 1: Get the accurate attribute value of starts and ends
    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }

    PADDLE_ENFORCE_EQ(
        starts.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    // Step 2: Compute output
    if (input_is_array) {
      DealTensorArray(ctx, starts, ends, out_is_array);
      return;
    }
  }
};

template <typename DeviceContext, typename T>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

    // Get the accurate attribute value of starts and ends
    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }

    Variable* d_input_var = ctx.OutputVar(framework::GradVarName("Input"));
    const Variable* d_out_var = ctx.InputVar(framework::GradVarName("Out"));
    bool d_input_is_array = d_input_var->IsType<LoDTensorArray>();
    bool d_out_is_array = d_out_var->IsType<LoDTensorArray>();

    if (d_input_is_array) {
      auto* input_array = ctx.Input<LoDTensorArray>("Input");
      auto* d_in_arr =
          ctx.Output<LoDTensorArray>(framework::GradVarName("Input"));

      int64_t d_in_size = input_array->size();
      d_in_arr->resize(d_in_size);
      // If the input is LoDTensorArray, the rank of input is 1.
      // So only use the 0th element of starts.
      int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
      start = std::max(start, static_cast<int64_t>(0));
      // set zero
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto& dev_ctx = *pool.Get(ctx.GetPlace());
      phi::funcs::SetConstant<DeviceContext, T> functor;
      for (int i = 0; i < d_in_size; ++i) {
        auto dim = input_array->at(i).dims();
        d_in_arr->at(i).Resize(dim);
        d_in_arr->at(i).mutable_data<T>(ctx.GetPlace());
        functor(reinterpret_cast<const DeviceContext&>(dev_ctx),
                &d_in_arr->at(i), static_cast<T>(0));
      }

      if (d_out_is_array) {
        auto* d_out_arr =
            ctx.Input<LoDTensorArray>(framework::GradVarName("Out"));
        int d_out_size = d_out_arr->size();
        for (int i = 0; i < d_out_size; ++i) {
          paddle::framework::TensorCopy(d_out_arr->at(i), ctx.GetPlace(),
                                        &(d_in_arr->at(start + i)));
        }
      } else {
        auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
        paddle::framework::TensorCopy(*d_out, ctx.GetPlace(),
                                      &(d_in_arr->at(start)));
      }
      return;
    }
  }

 private:
};
}  // namespace operators
}  // namespace paddle
