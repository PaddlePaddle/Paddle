/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"

#define INLINE_FOR2(sizei, sizej)     \
  for (int64_t i = 0; i < sizei; i++) \
    for (int64_t j = 0; j < sizej; j++)

namespace paddle {
namespace operators {
namespace math {

// SelectedRows + SelectedRows will simplely concat value and rows.
// The real computation happens in dealing with LoDTensor.
template <typename DeviceContext, typename T>
struct SelectedRowsAdd {
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output);
};

template <typename DeviceContext, typename T>
struct SelectedRowsAddTensor {
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output);
};

// input2 = input1 + input2
template <typename DeviceContext, typename T>
struct SelectedRowsAddTo {
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input1,
                  const int64_t input2_offset, framework::SelectedRows* input2);
};

// input2 = [all input in input1] + input2
template <typename DeviceContext, typename T>
struct SelectedRowsSumTo {
  void operator()(const DeviceContext& context,
                  const std::vector<framework::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  framework::SelectedRows* input2);
};

// input2 = input1 + input2
template <typename DeviceContext, typename T>
struct SelectedRowsAddToTensor {
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2);
};

namespace scatter {
// functors for manuplating SelectedRows data
template <typename DeviceContext, typename T>
struct MergeAdd {
  // unary functor, merge by adding duplicated rows in
  // the input SelectedRows object.
  framework::SelectedRows operator()(const DeviceContext& context,
                                     const framework::SelectedRows& input);
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output);
};

template <>
struct MergeAdd<platform::CPUDeviceContext, float> {
  framework::SelectedRows operator()(const platform::CPUDeviceContext& context,
                                     const framework::SelectedRows& input) {
    framework::SelectedRows out;
    (*this)(context, input, &out);
    return out;
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output) {
    framework::SelectedRows& out = *output;
    auto input_rows = input.rows();
    std::vector<int64_t> merge_rows;
    merge_rows.reserve(input_rows.size());
    std::unordered_map<int64_t, size_t> rows_pos_map;
    rows_pos_map.reserve(input_rows.size());
    size_t idx = 0u;
    for (std::vector<int64_t>::iterator iter = input_rows.begin();
         iter != input_rows.end(); ++iter) {
      if (rows_pos_map.find(*iter) == rows_pos_map.end()) {
        rows_pos_map[*iter] = idx++;
        merge_rows.emplace_back(*iter);
      }
    }

    auto input_width = input.value().dims()[1];
    out.set_rows(merge_rows);
    out.set_height(input.height());
    out.mutable_value()->mutable_data<float>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CPUDeviceContext, float> constant_functor;
    constant_functor(context, out.mutable_value(), 0.0);

    auto* out_data = out.mutable_value()->data<float>();
    auto* input_data = input.value().data<float>();

    auto blas = GetBlas<platform::CPUDeviceContext, float>(context);
    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_pos_map[input_rows[i]];
      float* y = out_data + out_i * input_width;
      const float* x = input_data + i * input_width;
      blas.AXPY(input_width, 1., x, y);
    }
  }
};

template <>
struct MergeAdd<platform::CPUDeviceContext, double> {
  framework::SelectedRows operator()(const platform::CPUDeviceContext& context,
                                     const framework::SelectedRows& input) {
    framework::SelectedRows out;
    (*this)(context, input, &out);
    return out;
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output) {
    framework::SelectedRows& out = *output;
    auto input_rows = input.rows();
    std::vector<int64_t> merge_rows;
    merge_rows.reserve(input_rows.size());
    std::unordered_map<int64_t, size_t> rows_pos_map;
    rows_pos_map.reserve(input_rows.size());
    size_t idx = 0u;
    for (std::vector<int64_t>::iterator iter = input_rows.begin();
         iter != input_rows.end(); ++iter) {
      if (rows_pos_map.find(*iter) == rows_pos_map.end()) {
        rows_pos_map[*iter] = idx++;
        merge_rows.emplace_back(*iter);
      }
    }

    auto input_width = input.value().dims()[1];
    out.set_rows(merge_rows);
    out.set_height(input.height());
    out.mutable_value()->mutable_data<double>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CPUDeviceContext, double> constant_functor;
    constant_functor(context, out.mutable_value(), 0.0);

    auto* out_data = out.mutable_value()->data<double>();
    auto* input_data = input.value().data<double>();

    auto blas = GetBlas<platform::CPUDeviceContext, double>(context);
    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_pos_map[input_rows[i]];
      double* y = out_data + out_i * input_width;
      const double* x = input_data + i * input_width;
      blas.AXPY(input_width, 1., x, y);
    }
  }
};

template <typename DeviceContext, typename T>
struct Add {
  framework::SelectedRows operator()(const DeviceContext& context,
                                     const framework::SelectedRows& input1,
                                     const framework::SelectedRows& input2) {
    framework::SelectedRows out;
    out.set_rows(input1.rows());
    out.set_height(input1.height());
    out.mutable_value()->mutable_data<T>(input1.value().dims(),
                                         context.GetPlace());
    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
    auto e_in2 = framework::EigenVector<T>::Flatten(input2.value());
    e_out.device(*context.eigen_device()) = e_in1 + e_in2;
    return out;
  }
};

template <typename DeviceContext, typename T>
struct Mul {
  // multiply two SelectedRows
  framework::SelectedRows operator()(const DeviceContext& context,
                                     const framework::SelectedRows& input1,
                                     const framework::SelectedRows& input2) {
    framework::SelectedRows out;
    out.set_rows(input1.rows());
    out.set_height(input1.height());
    out.mutable_value()->mutable_data<T>(input1.value().dims(),
                                         context.GetPlace());
    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
    auto e_in2 = framework::EigenVector<T>::Flatten(input2.value());
    e_out.device(*context.eigen_device()) = e_in1 * e_in2;
    return out;
  }
  // multiply scalar to SelectedRows
  framework::SelectedRows operator()(const DeviceContext& context,
                                     const framework::SelectedRows& input1,
                                     const T input2) {
    framework::SelectedRows out;
    out.set_rows(input1.rows());
    out.set_height(input1.height());
    out.mutable_value()->mutable_data<T>(input1.value().dims(),
                                         context.GetPlace());
    auto e_out = framework::EigenVector<T>::Flatten(*(out.mutable_value()));
    auto e_in1 = framework::EigenVector<T>::Flatten(input1.value());
    e_out.device(*context.eigen_device()) = input2 * e_in1;
    return out;
  }
};

enum class ScatterOps { ASSIGN, ADD, SUB, SUBBY, MUL, DIV, DIVBY };

// out = seleted_rows_in / tensor
template <typename DeviceContext, typename T>
struct UpdateToTensor {
  void operator()(const DeviceContext& context, const ScatterOps& op,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2);
};

}  // namespace scatter
}  // namespace math
}  // namespace operators
}  // namespace paddle
