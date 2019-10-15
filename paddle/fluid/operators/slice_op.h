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

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

inline std::vector<int> get_new_data_from_tensorlist(
    const std::vector<const Tensor*>& list_new_data_tensor) {
  // get tensor from
  std::vector<int> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    PADDLE_ENFORCE_EQ(tensor->dims(), framework::make_ddim({1}),
                      "shape of dim tensor should be [1]");
    if (platform::is_gpu_place(tensor->place())) {
      framework::Tensor temp;
      TensorCopySync(*tensor, platform::CPUPlace(), &temp);
      vec_new_data.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
    } else {
      vec_new_data.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
    }
  }
  return vec_new_data;
}
inline std::vector<int> get_new_data_from_tensor(
    const Tensor* new_data_tensor) {
  std::vector<int> vec_new_data;
  auto* new_data = new_data_tensor->data<int>();
  framework::Tensor cpu_starts_tensor;
  if (platform::is_gpu_place(new_data_tensor->place())) {
    TensorCopySync(*new_data_tensor, platform::CPUPlace(), &cpu_starts_tensor);
    new_data = cpu_starts_tensor.data<int>();
  }
  vec_new_data =
      std::vector<int>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        SliceCompute<1>(ctx);
        break;
      case 2:
        SliceCompute<2>(ctx);
        break;
      case 3:
        SliceCompute<3>(ctx);
        break;
      case 4:
        SliceCompute<4>(ctx);
        break;
      case 5:
        SliceCompute<5>(ctx);
        break;
      case 6:
        SliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void SliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");
    auto out_dims = out->dims();
    auto in_dims = in->dims();

    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");
    auto ends = context.Attr<std::vector<int>>("ends");
    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = context.Attr<std::vector<int>>("infer_flags");

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");

    bool need_infer = false;
    if (context.HasInput("StartsTensor") || context.HasInput("EndsTensor")) {
      need_infer = true;
    }
    if (list_new_starts_tensor.size() > 0 || list_new_ends_tensor.size() > 0) {
      need_infer = true;
    }

    if (need_infer) {
      if (context.HasInput("StartsTensor")) {
        auto* starts_tensor = context.Input<framework::Tensor>("StartsTensor");
        starts = get_new_data_from_tensor(starts_tensor);
      } else if (list_new_starts_tensor.size() > 0) {
        starts = get_new_data_from_tensorlist(list_new_starts_tensor);
      }
      PADDLE_ENFORCE_EQ(
          starts.size(), axes.size(),
          "The size of starts must be equal to the size of axes.");
      if (context.HasInput("EndsTensor")) {
        auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
        ends = get_new_data_from_tensor(ends_tensor);
      } else if (list_new_ends_tensor.size() > 0) {
        ends = get_new_data_from_tensorlist(list_new_ends_tensor);
      }
      PADDLE_ENFORCE_EQ(ends.size(), axes.size(),
                        "The size of ends must be equal to the size of axes.");
      out_dims = in_dims;
      int dim_value, start, end;
      for (size_t i = 0; i < axes.size(); ++i) {
        dim_value = out_dims[axes[i]];
        if (dim_value > 0) {
          // when end = start+1 and start == -1
          if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
            auto ret =
                std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
            if (ret != decrease_axis.end()) {
              ends[i] = 10000000;
            }
          }

          start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
          end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
          start = std::max(start, 0);
          end = std::max(end, 0);
          end = std::min(end, dim_value);
          PADDLE_ENFORCE_GT(end, start, "end should greater than start");
          out_dims[axes[i]] = end - start;
        }
      }
      out->Resize(out_dims);
      // generate new shape
      if (decrease_axis.size() > 0) {
        std::vector<int> new_out_shape;
        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          PADDLE_ENFORCE_EQ(out_dims[decrease_axis[i]], 1,
                            "decrease dim should be 1");
          out_dims[decrease_axis[i]] = 0;
        }

        for (int i = 0; i < out_dims.size(); ++i) {
          if (out_dims[i] != 0) {
            new_out_shape.push_back(out_dims[i]);
          }
        }
        if (new_out_shape.size() == 0) {
          new_out_shape.push_back(1);
        }

        out_dims = framework::make_ddim(new_out_shape);
      }
    }

    // resize out_dims
    if (decrease_axis.size() > 0) {
      if (decrease_axis.size() == (size_t)in_dims.size()) {
        std::vector<int> vec_origin_out_shape(decrease_axis.size(), 1);
        out->Resize(framework::make_ddim(vec_origin_out_shape));
      } else {
        std::vector<int> vec_origin_out_shape(
            out_dims.size() + decrease_axis.size(), -1);

        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          vec_origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
          if (vec_origin_out_shape[i] == -1) {
            vec_origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }

        out->Resize(framework::make_ddim(vec_origin_out_shape));
      }
    }

    out->mutable_data<T>(context.GetPlace());

    auto new_out_dims = out->dims();
    auto offsets = Eigen::array<int, D>();
    auto extents = Eigen::array<int, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = new_out_dims[i];
    }
    int start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, 0);
      offsets[axes[i]] = start;
    }
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out, new_out_dims);
    out_t.device(place) = in_t.slice(offsets, extents);

    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    size_t rank = ctx.Input<framework::Tensor>("Input")->dims().size();
    switch (rank) {
      case 1:
        SliceCompute<1>(ctx);
        break;
      case 2:
        SliceCompute<2>(ctx);
        break;
      case 3:
        SliceCompute<3>(ctx);
        break;
      case 4:
        SliceCompute<4>(ctx);
        break;
      case 5:
        SliceCompute<5>(ctx);
        break;
      case 6:
        SliceCompute<6>(ctx);
        break;
    }
  }

 private:
  template <size_t D>
  void SliceCompute(const framework::ExecutionContext& context) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_input =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    d_input->mutable_data<T>(context.GetPlace());
    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();
    auto axes = context.Attr<std::vector<int>>("axes");
    auto starts = context.Attr<std::vector<int>>("starts");
    auto ends = context.Attr<std::vector<int>>("ends");

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");

    if (list_new_starts_tensor.size() > 0) {
      starts = get_new_data_from_tensorlist(list_new_starts_tensor);
    } else if (context.HasInput("StartsTensor")) {
      auto* starts_tensor = context.Input<framework::Tensor>("StartsTensor");
      starts = get_new_data_from_tensor(starts_tensor);
    }

    if (list_new_ends_tensor.size() > 0) {
      ends = get_new_data_from_tensorlist(list_new_ends_tensor);
    } else if (context.HasInput("EndsTensor")) {
      auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
      ends = get_new_data_from_tensor(ends_tensor);
    }

    auto decrease_axis = context.Attr<std::vector<int>>("decrease_axis");
    if (decrease_axis.size() > 0) {
      if (decrease_axis.size() == (size_t)in_dims.size()) {
        // all dims decrease
        std::vector<int> vec_origin_out_shape(decrease_axis.size(), 1);
        out_dims = framework::make_ddim(vec_origin_out_shape);
      } else {
        std::vector<int> vec_origin_out_shape(
            out_dims.size() + decrease_axis.size(), -1);

        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          vec_origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
          if (vec_origin_out_shape[i] == -1) {
            vec_origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }

        out_dims = framework::make_ddim(vec_origin_out_shape);
      }
    }

    auto offsets = Eigen::array<int, D>();
    auto extents = Eigen::array<int, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    int start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, 0);
      offsets[axes[i]] = start;
    }
    Eigen::array<std::pair<int, int>, D> paddings;
    for (size_t i = 0; i < paddings.size(); ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = (in_dims[i] - out_dims[i]) - offsets[i];
    }
    auto d_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input);
    auto d_out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);
    d_in_t.device(place) = d_out_t.pad(paddings, 0);
  }
};
}  // namespace operators
}  // namespace paddle
