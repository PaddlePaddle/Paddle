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

#include <map>
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
    std::vector<int64_t> input_rows(input.rows());

    std::map<int64_t, std::vector<int64_t>> merge_row_map;
    for (size_t i = 0; i < input_rows.size(); ++i) {
      merge_row_map[input_rows[i]].push_back(i);
    }

    std::vector<int64_t> merge_rows(merge_row_map.size());
    size_t idx = 0;
    int64_t input_width = input.value().dims()[1];
    out.set_height(input.height());

    auto* out_data = out.mutable_value()->mutable_data<float>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());
    auto* in_data = input.value().data<float>();

    auto blas = GetBlas<platform::CPUDeviceContext, float>(context);
    for (auto& row_pair : merge_row_map) {
      auto* out_ptr = out_data + idx * input_width;
      auto& rows = row_pair.second;
      merge_rows[idx] = row_pair.first;
      ++idx;
      // rows.size() is always larger than 0
      blas.VCOPY(input_width, in_data + rows[0] * input_width, out_ptr);

      for (size_t i = 1; i < rows.size(); ++i) {
        blas.AXPY(input_width, 1., in_data + rows[i] * input_width, out_ptr);
      }
    }

    out.set_rows(merge_rows);
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
    std::vector<int64_t> input_rows(input.rows());

    std::map<int64_t, std::vector<int64_t>> merge_row_map;
    for (size_t i = 0; i < input_rows.size(); ++i) {
      merge_row_map[input_rows[i]].push_back(i);
    }

    std::vector<int64_t> merge_rows(merge_row_map.size());
    size_t idx = 0;
    int64_t input_width = input.value().dims()[1];
    out.set_height(input.height());

    auto* out_data = out.mutable_value()->mutable_data<double>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());
    auto* in_data = input.value().data<double>();

    auto blas = GetBlas<platform::CPUDeviceContext, double>(context);
    for (auto& row_pair : merge_row_map) {
      auto* out_ptr = out_data + idx * input_width;
      auto& rows = row_pair.second;
      merge_rows[idx] = row_pair.first;
      ++idx;
      // rows.size() is always larger than 0
      blas.VCOPY(input_width, in_data + rows[0] * input_width, out_ptr);

      for (size_t i = 1; i < rows.size(); ++i) {
        blas.AXPY(input_width, 1., in_data + rows[i] * input_width, out_ptr);
      }
    }

    out.set_rows(merge_rows);
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
