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
#include "paddle/fluid/framework/selected_rows_utils.h"
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

// FIXME: The result of SelectedRowsAddToTensor maybe non deterministic,
// because it uses CudaAtomicAdd.
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
                                     const framework::SelectedRows& input,
                                     const bool sorted_result = false);
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output,
                  const bool sorted_result = false);
  void operator()(const DeviceContext& context,
                  const std::vector<const framework::SelectedRows*>& inputs,
                  framework::SelectedRows* output,
                  const bool sorted_result = false);
};

template <typename DeviceContext, typename T>
struct MergeAverage {
  framework::SelectedRows operator()(const DeviceContext& context,
                                     const framework::SelectedRows& input);
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output);
  void operator()(const DeviceContext& context,
                  const std::vector<const framework::SelectedRows*>& inputs,
                  framework::SelectedRows* output);
};

enum class ScatterOps { ASSIGN, ADD, SUB, SUBBY, MUL, DIV, DIVBY };

// out = selected_rows_in / tensor
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
