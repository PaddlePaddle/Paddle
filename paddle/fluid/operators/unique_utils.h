/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

// Binary function 'less than'
template <typename InT>
struct LessThan {
  int col;
  const InT* in_trans_data;

  LessThan(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
};

// Binary function 'equal_to'
template <typename InT>
struct BinaryEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

// Binary function 'not_equal_to'
template <typename InT>
struct BinaryNotEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryNotEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return true;
      }
    }
    return false;
  }
};

// index_select() function for Tensor
template <typename InT, typename IndexT>
void IndexSelect(const framework::ExecutionContext& context,
                 const Tensor& input, const Tensor& index, Tensor* output,
                 int dim) {
  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];

  std::vector<InT> input_vec;
  std::vector<IndexT> index_vec;
  TensorToVector(input, context.device_context(), &input_vec);
  TensorToVector(index, context.device_context(), &index_vec);
  std::vector<InT> out_vec(output->numel());

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_vec[i], 0,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim], index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i], input_dim[dim],
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim], index_vec[i]));
  }

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_vec[j];
      for (auto k = 0; k < slice_size; k++) {
        out_vec[output_start_offset + j * slice_size + k] =
            input_vec[input_start_offset + index_value * slice_size + k];
      }
    }
  }
  output->mutable_data<InT>(context.GetPlace());
  framework::TensorFromVector(out_vec, context.device_context(), output);
  output->Resize(output_dim);
}
}  // namespace operators
}  // namespace paddle
