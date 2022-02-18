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

#pragma once
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

template <typename T, typename Type>
static void getMode(Type input_height, Type input_width, int input_dim,
                    const framework::Tensor* input, T* t_out, Type* t_indices) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = framework::EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = framework::EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    std::sort(col_vec.begin(), col_vec.end(),
              [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                return (!std::isnan(static_cast<double>(l.first)) &&
                        std::isnan(static_cast<double>(r.first))) ||
                       (l.first < r.first);
              });
    T mode = 0;
    int64_t indice = 0;
    int64_t cur_freq = 0;
    int64_t max_freq = 0;
    for (int64_t i = 0; i < input_width; ++i) {
      ++cur_freq;
      if (i == input_width - 1 || (col_vec[i + 1].first != col_vec[i].first)) {
        if (cur_freq > max_freq) {
          max_freq = cur_freq;
          mode = col_vec[i].first;
          indice = col_vec[i].second;
        }
        cur_freq = 0;
      }
    }
    t_out[i] = mode;
    t_indices[i] = indice;
  }
}

template <typename T, typename Type>
static void ModeAssign(const Type& input_height, const Type& input_width,
                       const int& input_dim, const framework::Tensor* input,
                       const framework::Tensor* indices, T* output_data) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = framework::EigenVector<T>::Flatten(*input);
      auto e_indices = framework::EigenVector<Type>::Flatten(*indices);
      output_data[i * input_width + e_indices(0)] = e_input(0);
    } else {
      auto e_input = framework::EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices =
          framework::EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      output_data[i * input_width + e_indices(i, 0)] = e_input(i, 0);
    }
  }
}

template <typename DeviceContext, typename T>
class ModeCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("X");
    auto* output = context.Output<framework::Tensor>("Out");
    auto* indices = context.Output<framework::Tensor>("Indices");
    const auto& in_dims = input->dims();
    bool keepdim = static_cast<bool>(context.Attr<bool>("keepdim"));

    // axis < 0, cacluate the real axis
    int axis = static_cast<int>(context.Attr<int>("axis"));
    if (axis < 0) axis += in_dims.size();

    T* output_data = output->mutable_data<T>(context.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(context.GetPlace());
    auto out_dims = output->dims();
    // if axis is not the last dim, transpose it to the last dim, do the
    // calculation,
    // then tranpose it back to orginal axis.
    if (axis == in_dims.size() - 1) {
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      getMode<T, int64_t>(input_height, input_width, in_dims.size(), input,
                          output_data, indices_data);
    } else {
      std::vector<int> trans_axis;
      for (int i = 0; i < axis; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.emplace_back(axis);

      if (!keepdim) {
        std::vector<int> tmp_out_shape;
        for (int i = 0; i < axis; i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        tmp_out_shape.emplace_back(1);
        for (int i = axis + 1; i < in_dims.size(); i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        framework::DDim tmp_out_dim = framework::make_ddim(tmp_out_shape);
        output->Resize(tmp_out_dim);
        indices->Resize(tmp_out_dim);
      }

      // get the trans input_dims, out_dims
      framework::DDim trans_shape(in_dims);
      framework::DDim trans_out_shape(in_dims);

      for (size_t i = 0; i < trans_axis.size(); i++) {
        trans_shape[i] = in_dims[trans_axis[i]];
        trans_out_shape[i] = in_dims[trans_axis[i]];
      }
      trans_out_shape[in_dims.size() - 1] = 1;

      framework::Tensor trans_input;
      trans_input.mutable_data<T>(trans_shape, context.GetPlace());
      int ndims = trans_axis.size();
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();

      // transpose the input value
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, *input,
                                                  &trans_input, trans_axis);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_shape, 0, trans_shape.size() - 1));
      const int64_t input_width = trans_shape[trans_shape.size() - 1];
      framework::Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_out_shape, context.GetPlace());
      framework::Tensor tmp_indices;
      auto* t_ind = tmp_indices.mutable_data<int64_t>(trans_out_shape,
                                                      context.GetPlace());

      getMode<T, int64_t>(input_height, input_width, in_dims.size(),
                          &trans_input, t_out, t_ind);
      // transpose back
      TransCompute<platform::CPUDeviceContext, int64_t>(
          ndims, dev_context, tmp_indices, indices, trans_axis);
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, tmp_out,
                                                  output, trans_axis);
      if (!keepdim) {
        output->Resize(out_dims);
        indices->Resize(out_dims);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class ModeGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<framework::Tensor>("Indices");
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    int axis = static_cast<int>(context.Attr<int>("axis"));
    bool keepdim = static_cast<bool>(context.Attr<bool>("keepdim"));

    auto in_dims = x->dims();
    auto out_dims = indices->dims();

    // axis < 0, get the real axis
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    if (!keepdim) {
      std::vector<int> tmp_out_shape;
      for (int i = 0; i < axis; i++) {
        tmp_out_shape.emplace_back(out_dims[i]);
      }
      tmp_out_shape.emplace_back(1);
      for (int i = axis + 1; i < in_dims.size(); i++) {
        tmp_out_shape.emplace_back(out_dims[i - 1]);
      }
      out_dims = framework::make_ddim(tmp_out_shape);
    }
    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    if (axis == in_dims.size() - 1) {
      // allocate the memory for the input_grad
      // assign the out_grad to input_grad directly
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];

      // init the output grad with 0, because some input elements has no grad
      memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
      // Assign the output_grad to input_grad
      if (keepdim) {
        ModeAssign(input_height, input_width, in_dims.size(), out_grad, indices,
                   x_grad_data);
      } else {
        auto& dev_context =
            context.template device_context<platform::CPUDeviceContext>();
        framework::Tensor out_grad_tmp;
        framework::Tensor indices_tmp;
        out_grad_tmp.mutable_data<T>(out_grad->dims(), dev_context.GetPlace());
        indices_tmp.mutable_data<int64_t>(indices->dims(),
                                          dev_context.GetPlace());
        framework::TensorCopy(*out_grad, dev_context.GetPlace(), dev_context,
                              &out_grad_tmp);
        framework::TensorCopy(*indices, dev_context.GetPlace(), dev_context,
                              &indices_tmp);
        out_grad_tmp.Resize(out_dims);
        indices_tmp.Resize(out_dims);
        ModeAssign(input_height, input_width, in_dims.size(), &out_grad_tmp,
                   &indices_tmp, x_grad_data);
      }
    } else {
      // can not assign grad to input_grad, must do the transpose
      std::vector<int> trans_axis;
      for (int i = 0; i < axis; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.emplace_back(out_dims.size() - 1);
      for (int i = axis + 1; i < out_dims.size() - 1; i++) {
        trans_axis.emplace_back(i);
      }
      trans_axis.emplace_back(axis);
      framework::DDim trans_shape(out_dims);
      framework::DDim trans_in_shape(in_dims);
      for (size_t i = 0; i < trans_axis.size(); i++) {
        trans_shape[i] = out_dims[trans_axis[i]];
        trans_in_shape[i] = in_dims[trans_axis[i]];
      }
      // transpose the out_grad, indices
      framework::Tensor trans_dO;
      trans_dO.mutable_data<T>(trans_shape, context.GetPlace());
      framework::Tensor trans_ind;
      trans_ind.mutable_data<int64_t>(trans_shape, context.GetPlace());
      int ndims = trans_axis.size();
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();

      if (keepdim) {
        // Do transpose
        TransCompute<platform::CPUDeviceContext, T>(
            ndims, dev_context, *out_grad, &trans_dO, trans_axis);
        TransCompute<platform::CPUDeviceContext, int64_t>(
            ndims, dev_context, *indices, &trans_ind, trans_axis);
      } else {
        framework::Tensor out_grad_tmp;
        framework::Tensor indices_tmp;
        out_grad_tmp.mutable_data<T>(out_grad->dims(), dev_context.GetPlace());
        indices_tmp.mutable_data<int64_t>(indices->dims(),
                                          dev_context.GetPlace());
        framework::TensorCopy(*out_grad, dev_context.GetPlace(), dev_context,
                              &out_grad_tmp);
        framework::TensorCopy(*indices, dev_context.GetPlace(), dev_context,
                              &indices_tmp);
        out_grad_tmp.Resize(out_dims);
        indices_tmp.Resize(out_dims);
        // Do transpose
        TransCompute<platform::CPUDeviceContext, T>(
            ndims, dev_context, out_grad_tmp, &trans_dO, trans_axis);
        TransCompute<platform::CPUDeviceContext, int64_t>(
            ndims, dev_context, indices_tmp, &trans_ind, trans_axis);
      }
      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_in_shape, 0, trans_in_shape.size() - 1));
      const int64_t input_width = trans_in_shape[trans_in_shape.size() - 1];

      // Assign the out_grad to tranpose input_grad
      framework::Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_in_shape, context.GetPlace());
      memset(t_out, 0, x_grad->numel() * sizeof(T));

      ModeAssign<T, int64_t>(input_height, input_width, in_dims.size(),
                             &trans_dO, &trans_ind, t_out);

      // Transpose back
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, tmp_out,
                                                  x_grad, trans_axis);
    }
  }
};

}  // namespace operators
}  // namespace paddle
