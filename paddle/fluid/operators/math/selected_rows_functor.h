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
#include <set>
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

template <>
struct SelectedRowsSumTo<platform::CPUDeviceContext, float> {
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  framework::SelectedRows* input2) {
    // Ensure all selected rows have the same height
    // auto start = std::chrono::system_clock::now();
    size_t size = 0u;
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      auto& in_rows = (*iter)->rows();
      size += in_rows.end() - in_rows.begin();
      auto in1_height = (*iter)->height();
      PADDLE_ENFORCE_EQ(in1_height, input2->height());
    }

    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // LOG(ERROR) << "selected rows check, cost: " << diff.count();

    // start = std::chrono::system_clock::now();
    // concat rows
    auto& in2_rows = *(input2->mutable_rows());
    in2_rows.reserve(in2_rows.size() + size);
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      const framework::Vector<int64_t>& in_rows = (*iter)->rows();
      in2_rows.insert(in2_rows.end(), in_rows.begin(), in_rows.end());
    }

    // end = std::chrono::system_clock::now();
    // diff = end - start;
    // LOG(ERROR) << "selected rows concat rows, cost: " << diff.count();

    // start = std::chrono::system_clock::now();
    auto* in2_value = input2->mutable_value();
    auto* in2_data = in2_value->data<float>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, float>(context);
    size_t offset = 0u;
    for (size_t i = 0u; i != input1.size(); ++i) {
      auto& in_value = input1[i]->value();
      const auto* in_data = in_value.data<float>();
      offset += input2_offsets[i];
      blas.VCOPY(in_value.numel(), in_data, in2_data + offset);
    }
    // end = std::chrono::system_clock::now();
    // diff = end - start;
    // LOG(ERROR) << "selected rows value copy, cost: " << diff.count();
  }
};

template <>
struct SelectedRowsSumTo<platform::CPUDeviceContext, double> {
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  framework::SelectedRows* input2) {
    // Ensure all selected rows have the same height
    size_t size = 0u;
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      auto& in_rows = (*iter)->rows();
      size += in_rows.end() - in_rows.begin();
      auto in1_height = (*iter)->height();
      PADDLE_ENFORCE_EQ(in1_height, input2->height());
    }

    // concat rows
    auto& in2_rows = *(input2->mutable_rows());
    in2_rows.reserve(in2_rows.size() + size);
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      const framework::Vector<int64_t>& in_rows = (*iter)->rows();
      in2_rows.insert(in2_rows.end(), in_rows.begin(), in_rows.end());
    }

    auto* in2_value = input2->mutable_value();
    auto* in2_data = in2_value->data<float>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, float>(context);
    size_t offset = 0u;
    for (size_t i = 0u; i != input1.size(); ++i) {
      auto& in_value = input1[i]->value();
      const auto* in_data = in_value.data<float>();
      offset += input2_offsets[i];
      blas.VCOPY(in_value.numel(), in_data, in2_data + offset);
    }
  }
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
};

template <>
struct MergeAdd<platform::CPUDeviceContext, double> {
  framework::SelectedRows operator()(const platform::CPUDeviceContext& context,
                                     const framework::SelectedRows& input) {
    framework::SelectedRows out;

    auto input_rows = input.rows();
    std::set<int64_t> row_set(input_rows.begin(), input_rows.end());
    std::vector<int64_t> merge_rows(row_set.begin(), row_set.end());
    std::unordered_map<int64_t, size_t> rows_pos_map;
    rows_pos_map.reserve(merge_rows.size());
    for (std::vector<int64_t>::iterator iter = merge_rows.begin();
         iter != merge_rows.end(); ++iter) {
      rows_pos_map[*iter] = iter - merge_rows.begin();
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
      blas.VADD(input_width, x, const_cast<const double*>(y), y);
    }

    return out;
  }
};

template <>
struct MergeAdd<platform::CPUDeviceContext, float> {
  framework::SelectedRows operator()(const platform::CPUDeviceContext& context,
                                     const framework::SelectedRows& input) {
    framework::SelectedRows out;

    auto start = std::chrono::system_clock::now();
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

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(ERROR) << "adam_op merge_add prepare, cost: " << diff.count();

    start = std::chrono::system_clock::now();
    auto input_width = input.value().dims()[1];
    out.set_rows(merge_rows);
    out.set_height(input.height());
    out.mutable_value()->mutable_data<float>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CPUDeviceContext, float> constant_functor;
    constant_functor(context, out.mutable_value(), 0.0);

    end = std::chrono::system_clock::now();
    diff = end - start;
    LOG(ERROR) << "adam_op merge_add set_constant, cost: " << diff.count();

    start = std::chrono::system_clock::now();
    auto* out_data = out.mutable_value()->data<float>();
    auto* input_data = input.value().data<float>();

    auto blas = GetBlas<platform::CPUDeviceContext, float>(context);
    for (size_t i = 0; i < input_rows.size(); i++) {
      size_t out_i = rows_pos_map[input_rows[i]];

      float* y = out_data + out_i * input_width;
      const float* x = input_data + i * input_width;
      blas.AXPY(input_width, 1., x, y);
    }

    end = std::chrono::system_clock::now();
    diff = end - start;
    LOG(ERROR) << "adam_op merge_add do_merge, cost: " << diff.count()
               << " size " << input_rows.size();

    return out;
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
