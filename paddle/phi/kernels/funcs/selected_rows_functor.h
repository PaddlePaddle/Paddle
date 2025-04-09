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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define INLINE_FOR2(sizei, sizej)     \
  for (int64_t i = 0; i < sizei; i++) \
    for (int64_t j = 0; j < sizej; j++)

namespace phi {
namespace funcs {

// SelectedRows + SelectedRows will simply concat value and rows.
// The real computation happens in dealing with DenseTensor.
template <typename DeviceContext, typename T>
struct SelectedRowsAdd {
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input1,
                  const phi::SelectedRows& input2,
                  phi::SelectedRows* output);
};

template <typename DeviceContext, typename T>
struct SelectedRowsAddTensor {
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input1,
                  const phi::DenseTensor& input2,
                  phi::DenseTensor* output);
};

// input2 = input1 + input2
template <typename DeviceContext, typename T>
struct SelectedRowsAddTo {
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input1,
                  const int64_t input2_offset,
                  phi::SelectedRows* input2);
};

// input2 = [all input in input1] + input2
template <typename DeviceContext, typename T>
struct SelectedRowsSumTo {
  void operator()(const DeviceContext& context,
                  const std::vector<phi::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  phi::SelectedRows* input2);
};

// FIXME: The result of SelectedRowsAddToTensor maybe non deterministic,
// because it uses CudaAtomicAdd.
// input2 = input1 + input2
template <typename DeviceContext, typename T>
struct SelectedRowsAddToTensor {
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input1,
                  phi::DenseTensor* input2);
};

namespace scatter {
// functors for manipulating SelectedRows data
template <typename DeviceContext, typename T>
struct MergeAdd {
  // unary functor, merge by adding duplicated rows in
  // the input SelectedRows object.
  phi::SelectedRows operator()(const DeviceContext& context,
                               const phi::SelectedRows& input,
                               const bool sorted_result = false);
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output,
                  const bool sorted_result = false);
  void operator()(const DeviceContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output,
                  const bool sorted_result = false);
};

template <typename DeviceContext, typename T>
struct MergeAverage {
  phi::SelectedRows operator()(const DeviceContext& context,
                               const phi::SelectedRows& input);
  void operator()(const DeviceContext& context,
                  const phi::SelectedRows& input,
                  phi::SelectedRows* output);
  void operator()(const DeviceContext& context,
                  const std::vector<const phi::SelectedRows*>& inputs,
                  phi::SelectedRows* output);
};

enum class ScatterOps { ASSIGN, ADD, SUB, SUBBY, MUL, DIV, DIVBY };

// out = selected_rows_in / tensor
template <typename DeviceContext, typename T>
struct UpdateToTensor {
  void operator()(const DeviceContext& context,
                  const ScatterOps& op,
                  const phi::SelectedRows& input1,
                  phi::DenseTensor* input2);
};

}  // namespace scatter
}  // namespace funcs
}  // namespace phi
