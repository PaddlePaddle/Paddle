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
static void getKthvalue(Type input_height, Type input_width, int input_dim,
                        const framework::Tensor* input, T* t_out,
                        Type* t_indices, const int& k) {
  bool partial_sort_flag = (k * 64) < input_width;
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
    if (partial_sort_flag) {
      std::partial_sort(
          col_vec.begin(), col_vec.begin() + k, col_vec.end(),
          [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
            return (!std::isnan(static_cast<double>(l.first)) &&
                    std::isnan(static_cast<double>(r.first))) ||
                   (l.first < r.first);
          });
    } else {
      std::nth_element(
          col_vec.begin(), col_vec.begin() + k - 1, col_vec.end(),
          [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
            return (!std::isnan(static_cast<double>(l.first)) &&
                    std::isnan(static_cast<double>(r.first))) ||
                   (l.first < r.first);
          });
    }
    t_out[i] = col_vec[k - 1].first;
    t_indices[i] = col_vec[k - 1].second;
  }
}

template <typename T, typename Type>
static void kthvalueAssign(const Type& input_height, const Type& input_width,
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
class KthvalueCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("X");
    auto* output = context.Output<framework::Tensor>("Out");
    auto* indices = context.Output<framework::Tensor>("Indices");
    const auto& in_dims = input->dims();
    int k = static_cast<int>(context.Attr<int>("k"));
    bool keepdim = static_cast<bool>(context.Attr<bool>("keepdim"));
    int axis = static_cast<int>(context.Attr<int>("axis"));
    if (axis < 0) axis += in_dims.size();
    T* output_data = output->mutable_data<T>(context.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(context.GetPlace());
    auto out_dims = output->dims();
    if (axis == in_dims.size() - 1) {
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      getKthvalue<T, int64_t>(input_height, input_width, in_dims.size(), input,
                              output_data, indices_data, k);
    } else {
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);
      if (!keepdim) {
        std::vector<int> tmp_out_shape;
        for (int i = 0; i < axis; i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        tmp_out_shape.emplace_back(1);
        for (int i = axis + 1; i < in_dims.size(); i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        framework::DDim tmp_out_dims = framework::make_ddim(tmp_out_shape);
        output->Resize(tmp_out_dims);
        indices->Resize(tmp_out_dims);
      }
      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(in_dims);

      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
        trans_out_dims[i] = in_dims[trans[i]];
      }
      trans_out_dims[in_dims.size() - 1] = 1;
      framework::Tensor trans_inp;
      trans_inp.mutable_data<T>(trans_dims, context.GetPlace());
      int ndims = trans.size();
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, *input,
                                                  &trans_inp, trans);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];
      framework::Tensor tmp_out, tmp_indices;
      T* t_out = tmp_out.mutable_data<T>(trans_out_dims, context.GetPlace());
      auto* t_ind =
          tmp_indices.mutable_data<int64_t>(trans_out_dims, context.GetPlace());

      getKthvalue<T, int64_t>(input_height, input_width, in_dims.size(),
                              &trans_inp, t_out, t_ind, k);
      TransCompute<platform::CPUDeviceContext, int64_t>(
          ndims, dev_context, tmp_indices, indices, trans);
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, tmp_out,
                                                  output, trans);
      if (!keepdim) {
        output->Resize(out_dims);
        indices->Resize(out_dims);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class KthvalueGradCPUKernel : public framework::OpKernel<T> {
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
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];
      memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
      if (keepdim) {
        kthvalueAssign(input_height, input_width, in_dims.size(), out_grad,
                       indices, x_grad_data);
      } else {
        auto& dev_context =
            context.template device_context<platform::CPUDeviceContext>();
        framework::Tensor out_grad_tmp, indices_tmp;
        out_grad_tmp.mutable_data<T>(out_grad->dims(), dev_context.GetPlace());
        indices_tmp.mutable_data<int64_t>(indices->dims(),
                                          dev_context.GetPlace());
        framework::TensorCopy(*out_grad, dev_context.GetPlace(), dev_context,
                              &out_grad_tmp);
        framework::TensorCopy(*indices, dev_context.GetPlace(), dev_context,
                              &indices_tmp);
        out_grad_tmp.Resize(out_dims);
        indices_tmp.Resize(out_dims);
        kthvalueAssign(input_height, input_width, in_dims.size(), &out_grad_tmp,
                       &indices_tmp, x_grad_data);
      }
    } else {
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(out_dims.size() - 1);
      for (int i = axis + 1; i < out_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);
      framework::DDim trans_dims(out_dims);
      framework::DDim trans_in_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = out_dims[trans[i]];
        trans_in_dims[i] = in_dims[trans[i]];
      }
      framework::Tensor trans_dO, trans_ind;
      trans_dO.mutable_data<T>(trans_dims, context.GetPlace());
      trans_ind.mutable_data<int64_t>(trans_dims, context.GetPlace());
      int ndims = trans.size();
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();
      if (keepdim) {
        TransCompute<platform::CPUDeviceContext, T>(
            ndims, dev_context, *out_grad, &trans_dO, trans);
        TransCompute<platform::CPUDeviceContext, int64_t>(
            ndims, dev_context, *indices, &trans_ind, trans);
      } else {
        framework::Tensor out_grad_tmp, indices_tmp;
        out_grad_tmp.mutable_data<T>(out_grad->dims(), dev_context.GetPlace());
        indices_tmp.mutable_data<int64_t>(indices->dims(),
                                          dev_context.GetPlace());
        framework::TensorCopy(*out_grad, dev_context.GetPlace(), dev_context,
                              &out_grad_tmp);
        framework::TensorCopy(*indices, dev_context.GetPlace(), dev_context,
                              &indices_tmp);
        out_grad_tmp.Resize(out_dims);
        indices_tmp.Resize(out_dims);
        TransCompute<platform::CPUDeviceContext, T>(
            ndims, dev_context, out_grad_tmp, &trans_dO, trans);
        TransCompute<platform::CPUDeviceContext, int64_t>(
            ndims, dev_context, indices_tmp, &trans_ind, trans);
      }
      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_in_dims, 0, trans_in_dims.size() - 1));
      const int64_t input_width = trans_in_dims[trans_in_dims.size() - 1];
      framework::Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_in_dims, context.GetPlace());
      memset(t_out, 0, x_grad->numel() * sizeof(T));
      kthvalueAssign<T, int64_t>(input_height, input_width, in_dims.size(),
                                 &trans_dO, &trans_ind, t_out);
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_context, tmp_out,
                                                  x_grad, trans);
    }
  }
};
}  // namespace operators
}  // namespace paddle
