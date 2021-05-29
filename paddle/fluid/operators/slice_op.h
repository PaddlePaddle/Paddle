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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<framework::LoDTensorArray>();
    int rank = is_tensor_array
                   ? 1
                   : ctx.Input<framework::Tensor>("Input")->dims().size();

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
    const framework::Variable* input_var = context.InputVar("Input");
    framework::Variable* out_var = context.OutputVar("Out");
    bool input_is_tensor_array = input_var->IsType<framework::LoDTensorArray>();
    bool out_is_tensor_array = out_var->IsType<framework::LoDTensorArray>();

    auto axes = context.Attr<std::vector<int>>("axes");

    auto starts_int = context.Attr<std::vector<int>>("starts");
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    auto ends_int = context.Attr<std::vector<int>>("ends");
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
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
        starts = GetDataFromTensor<int64_t>(starts_tensor);
      } else if (list_new_starts_tensor.size() > 0) {
        starts = GetDataFromTensorList<int64_t>(list_new_starts_tensor);
      }
      if (context.HasInput("EndsTensor")) {
        auto* ends_tensor = context.Input<framework::Tensor>("EndsTensor");
        ends = GetDataFromTensor<int64_t>(ends_tensor);
      } else if (list_new_ends_tensor.size() > 0) {
        ends = GetDataFromTensorList<int64_t>(list_new_ends_tensor);
      }
    }
    PADDLE_ENFORCE_EQ(
        starts.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));
    if (input_is_tensor_array) {
      auto in_array = context.Input<framework::LoDTensorArray>("Input");
      // If the input is LoDTensorArray, the rank of input is 1.
      int64_t in_size = in_array->size();
      int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
      int64_t end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

      start = std::max(start, static_cast<int64_t>(0));
      end = std::max(end, static_cast<int64_t>(0));
      end = std::min(end, in_size);

      PADDLE_ENFORCE_GT(end, start,
                        platform::errors::InvalidArgument(
                            "Attr(ends) should be greater than attr(starts) in "
                            "slice op. But received end = %d, start = %d.",
                            ends[0], starts[0]));
      int64_t out_size = end - start;

      if (out_is_tensor_array) {
        auto out_array = context.Output<framework::LoDTensorArray>("Out");
        out_array->resize(out_size);

        for (int i = 0; i < out_size; ++i) {
          auto* out_tensor = &out_array->at(i);
          auto in_tensor = in_array->at(i + start);
          out_tensor->set_lod(in_tensor.lod());
          if (in_tensor.memory_size() > 0) {
            TensorCopy(in_tensor, context.GetPlace(), out_tensor);
          } else {
            VLOG(10)
                << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                   "nothing has been written to output array["
                << i << "].";
          }
        }
      } else {
        auto out = context.Output<framework::Tensor>("Out");
        auto in_tensor = in_array->at(start);
        TensorCopy(in_tensor, context.GetPlace(), out);
      }

      return;
    }

    auto in = context.Input<framework::Tensor>("Input");
    auto out = context.Output<framework::Tensor>("Out");

    auto out_dims = out->dims();
    auto in_dims = in->dims();
    if (need_infer) {
      out_dims = in_dims;
      int64_t dim_value, start, end;
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
          start = std::max(start, static_cast<int64_t>(0));
          end = std::max(end, static_cast<int64_t>(0));
          end = std::min(end, dim_value);
          PADDLE_ENFORCE_GT(
              end, start,
              platform::errors::InvalidArgument(
                  "Attr(ends) should be greater than attr(starts) in "
                  "slice op. But received end = %d, start = %d.",
                  ends[i], starts[i]));
          out_dims[axes[i]] = end - start;
        }
      }
      out->Resize(out_dims);
      // generate new shape
      if (decrease_axis.size() > 0) {
        std::vector<int64_t> new_out_shape;
        for (size_t i = 0; i < decrease_axis.size(); ++i) {
          PADDLE_ENFORCE_EQ(
              out_dims[decrease_axis[i]], 1,
              platform::errors::InvalidArgument("decrease dim should be 1"));
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
    auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = new_out_dims[i];
    }
    int64_t start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axes[i]] = start;
    }
    auto in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *in);
    auto out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *out, new_out_dims);

    if (in->numel() <= Eigen::NumTraits<int>::highest()) {
      // similar to tf.slice:
      // if element number less than INT_MAX, change the type of index to int
      Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
      for (size_t i = 0; i < D; i++) {
        offsets_32bit[i] = offsets[i];
        extents_32bit[i] = extents[i];
      }
      EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
          place, framework::To32BitIndex(out_t), framework::To32BitIndex(in_t),
          offsets_32bit, extents_32bit);
    } else {
      EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(place, out_t, in_t,
                                                            offsets, extents);
    }

    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<framework::LoDTensorArray>();
    size_t rank = is_tensor_array
                      ? 1
                      : ctx.Input<framework::Tensor>("Input")->dims().size();

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
    auto axes = context.Attr<std::vector<int>>("axes");

    auto starts_int = context.Attr<std::vector<int>>("starts");
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());

    auto ends_int = context.Attr<std::vector<int>>("ends");
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

    auto list_new_ends_tensor =
        context.MultiInput<framework::Tensor>("EndsTensorList");
    auto list_new_starts_tensor =
        context.MultiInput<framework::Tensor>("StartsTensorList");

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
    framework::Variable* d_input_var =
        context.OutputVar(framework::GradVarName("Input"));
    const framework::Variable* d_out_var =
        context.InputVar(framework::GradVarName("Out"));
    bool d_input_is_tensor_array =
        d_input_var->IsType<framework::LoDTensorArray>();
    bool d_out_is_tensor_array = d_out_var->IsType<framework::LoDTensorArray>();

    if (d_input_is_tensor_array) {
      auto* input_array = context.Input<framework::LoDTensorArray>("Input");
      auto* d_input_array = context.Output<framework::LoDTensorArray>(
          framework::GradVarName("Input"));

      int64_t d_in_size = input_array->size();
      d_input_array->resize(d_in_size);
      // If the input is LoDTensorArray, the rank of input is 1.
      // So only use the 0th element of starts.
      int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
      start = std::max(start, static_cast<int64_t>(0));
      // set zero
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto& dev_ctx = *pool.Get(context.GetPlace());
      T value = T(0);
      math::SetConstant<DeviceContext, T> functor;
      for (int i = 0; i < d_in_size; ++i) {
        auto dim = input_array->at(i).dims();
        d_input_array->at(i).Resize(dim);
        d_input_array->at(i).mutable_data<T>(context.GetPlace());
        functor(reinterpret_cast<const DeviceContext&>(dev_ctx),
                &d_input_array->at(i), static_cast<T>(value));
      }

      if (d_out_is_tensor_array) {
        auto* d_out_array = context.Input<framework::LoDTensorArray>(
            framework::GradVarName("Out"));
        int d_out_size = d_out_array->size();
        for (int i = 0; i < d_out_size; ++i) {
          TensorCopy(d_out_array->at(i), context.GetPlace(),
                     &(d_input_array->at(start + i)));
        }

      } else {
        auto* d_out =
            context.Input<framework::Tensor>(framework::GradVarName("Out"));
        TensorCopy(*d_out, context.GetPlace(), &(d_input_array->at(start)));
      }
      return;
    }

    auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* d_input =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    d_input->mutable_data<T>(context.GetPlace());

    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();

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

    auto offsets = Eigen::array<int64_t, D>();
    auto extents = Eigen::array<int64_t, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }
    int64_t start;
    for (size_t i = 0; i < axes.size(); ++i) {
      start = starts[i];
      if (start < 0) {
        start = (start + in_dims[axes[i]]);
      }
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axes[i]] = start;
    }
    Eigen::array<std::pair<int64_t, int64_t>, D> paddings;
    for (size_t i = 0; i < paddings.size(); ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = (in_dims[i] - out_dims[i]) - offsets[i];
    }
    EigenPaddingCompute(context, d_input, in_dims, d_out, out_dims, paddings);
  }

  template <size_t D>
  void EigenPaddingCompute(
      const framework::ExecutionContext& context, framework::Tensor* d_input,
      const framework::DDim& in_dims, const framework::Tensor* d_out,
      const framework::DDim& out_dims,
      const Eigen::array<std::pair<int64_t, int64_t>, D>& paddings) const {
    if (D <= 3) {
      // if dimension less than 3, cannot reduce dimension
      LaunchEigenPadding(context, d_input, in_dims, d_out, out_dims, paddings);
    } else {  // else we can reduce dimension
      // count not-zero padding number, and record the dimension
      int need_pad_num = 0, pad_dim = -1;
      for (size_t i = 0; i < D; i++) {
        if (paddings[i].first != 0 || paddings[i].second != 0) {
          need_pad_num++;
          pad_dim = i;
        }
      }

      if (need_pad_num == 0) {
        // do not need padding, pass if data address same, else copy
        if (d_input->mutable_data<T>(context.GetPlace()) == d_out->data<T>()) {
          // inplace, do not any operator, pass
        } else {
          framework::TensorCopy(
              *d_out, context.GetPlace(),
              context.template device_context<platform::DeviceContext>(),
              d_input);
        }
      } else if (need_pad_num == 1) {
        // only need padding one dimension, we can reduce dimension.
        // only the padding dimension is available for us.
        // How to reduce dimension(5 to 3 for example):
        // before(D=5):
        // in_dims:        [x1,  x2,  x3,  x4,  x5]
        // padding.first:  [0,   0,   a,   0,  0]
        // padding.second: [0,   0,   b,   0,  0]
        //                     | |
        //                     V V
        // after(D=3):
        // reshaped_in_dims:        [x1*x2,  x3,  x4*x5]
        // reshaped_padding.first:  [0,      a,     0]
        // reshaped_padding.second: [0,      b,     0]

        if (pad_dim == D - 1) {
          // only last dimension need padding,
          // reshape the dimension of tensor in 2: [preceding, padding]
          std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

          // first dimension is the accumulate of preceding dimension
          for (int i = 0; i < pad_dim; i++) {
            in_tore_shape[0] *= in_dims[i];
            out_tore_shape[0] *= out_dims[i];
          }
          // second dimension is the padding dimension
          in_tore_shape[1] = in_dims[pad_dim];
          out_tore_shape[1] = out_dims[pad_dim];

          // convert array from std::vector to DDim
          framework::DDim reshaped_in_dims =
              framework::make_ddim(in_tore_shape);
          framework::DDim reshaped_out_dims =
              framework::make_ddim(out_tore_shape);

          // after reshape: the first dimension do not need padding,
          // set padding[0] zero
          reshaped_padding[0].first = reshaped_padding[0].second = 0;
          // the second dimension is the previous padding dimension
          reshaped_padding[1].first = paddings[pad_dim].first;
          reshaped_padding[1].second = paddings[pad_dim].second;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        } else if (pad_dim == 0) {
          // only first dimension need padding,
          // reshape the dimension of tensor in 2: [padding, succeeding]
          // similar to (D - 1)
          std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

          // first dimension is the padding dimension
          in_tore_shape[0] = in_dims[pad_dim];
          out_tore_shape[0] = out_dims[pad_dim];
          // sencond dimension is the accumulate of succeeding dimension
          for (size_t i = pad_dim + 1; i < D; i++) {
            in_tore_shape[1] *= in_dims[i];
            out_tore_shape[1] *= out_dims[i];
          }

          // convert array from std::vector to DDim
          framework::DDim reshaped_in_dims =
              framework::make_ddim(in_tore_shape);
          framework::DDim reshaped_out_dims =
              framework::make_ddim(out_tore_shape);

          // after reshape:
          // the first dimension is the previous padding dimension
          reshaped_padding[0].first = paddings[pad_dim].first;
          reshaped_padding[0].second = paddings[pad_dim].second;
          // the second dimension do not need padding, set padding[1] zero
          reshaped_padding[1].first = reshaped_padding[1].second = 0;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        } else {
          // other dimension need padding
          // reshape the dimension of tensor in 3:
          // [preceding, padding, succeeding]
          std::vector<int64_t> in_tore_shape(3, 1), out_tore_shape(3, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 3> reshaped_padding;

          // first dimension is the accumulate of preceding dimension
          for (int i = 0; i < pad_dim; i++) {
            in_tore_shape[0] *= in_dims[i];
            out_tore_shape[0] *= out_dims[i];
          }
          // second dimension is the padding dimension
          in_tore_shape[1] = in_dims[pad_dim];
          out_tore_shape[1] = out_dims[pad_dim];
          // third dimension is the accumulate of succeeding dimension
          for (size_t i = pad_dim + 1; i < D; i++) {
            in_tore_shape[2] *= in_dims[i];
            out_tore_shape[2] *= out_dims[i];
          }

          // convert array from std::vector to DDim
          framework::DDim reshaped_in_dims =
              framework::make_ddim(in_tore_shape);
          framework::DDim reshaped_out_dims =
              framework::make_ddim(out_tore_shape);

          // after reshape:
          // the first dimension do not need padding, set padding[0] zero
          reshaped_padding[0].first = reshaped_padding[2].second = 0;
          // the second dimension is the previous padding dimension
          reshaped_padding[1].first = paddings[pad_dim].first;
          reshaped_padding[1].second = paddings[pad_dim].second;
          // the third dimension do not need padding, set padding[2] zero
          reshaped_padding[2].first = reshaped_padding[2].second = 0;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        }
      } else {
        // need padding at many dimension, cannot reduce dimension
        LaunchEigenPadding(context, d_input, in_dims, d_out, out_dims,
                           paddings);
      }
    }
  }

  template <size_t D>
  void LaunchEigenPadding(
      const framework::ExecutionContext& context, framework::Tensor* d_input,
      const framework::DDim& in_dims, const framework::Tensor* d_out,
      const framework::DDim& out_dims,
      const Eigen::array<std::pair<int64_t, int64_t>, D>& paddings) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto d_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input, in_dims);
    auto d_out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);

    if (d_input->numel() <= Eigen::NumTraits<int>::highest()) {
      // similar to tf.pad:
      // if element number less than INT_MAX, change the type of index to int
      Eigen::array<std::pair<int, int>, D> paddings_32bit;
      for (size_t i = 0; i < D; i++) {
        paddings_32bit[i] =
            std::make_pair(paddings[i].first, paddings[i].second);
      }
      EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
          place, framework::To32BitIndex(d_in_t),
          framework::To32BitIndex(d_out_t), paddings_32bit, static_cast<T>(0));
    } else {
      EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
          place, d_in_t, d_out_t, paddings, static_cast<T>(0));
    }
  }
};
}  // namespace operators
}  // namespace paddle
