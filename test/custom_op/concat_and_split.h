// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>
#include <vector>
#include "paddle/extension.h"

int64_t GetRows(std::vector<int64_t> shape, int64_t axis) {
  int64_t rows = 1;
  for (int64_t i = 0; i < axis; ++i) {
    rows *= shape[i];
  }
  return rows;
}

std::vector<int64_t> GetCols(const std::vector<paddle::Tensor>& ins,
                             int64_t rows,
                             int64_t* cols) {
  std::vector<int64_t> cols_vec(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    int64_t t_cols = ins[i].size() / rows;
    *cols += t_cols;
    cols_vec[i] = t_cols;
  }
  return cols_vec;
}

template <typename data_t>
void ConcatCpuKernel(const std::vector<paddle::Tensor>& ins,
                     paddle::Tensor* out,
                     int64_t axis) {
  size_t num = ins.size();
  int64_t out_rows = GetRows(ins[0].shape(), axis);
  int64_t out_cols = 0;
  auto ins_cols = GetCols(ins, out_rows, &out_cols);

  auto* out_data = out->mutable_data<data_t>();
  int64_t col_idx = 0;
  for (size_t i = 0; i < num; ++i) {
    int64_t col_len = ins_cols[i];
    auto* in_data = ins[i].data<data_t>();
    for (int j = 0; j < out_rows; ++j) {
      std::memcpy(out_data + j * out_cols + col_idx,
                  in_data + j * col_len,
                  sizeof(data_t) * col_len);
    }
    col_idx += col_len;
  }
}

template <typename data_t>
void SplitCpuKernel(const paddle::Tensor& in,
                    const std::vector<paddle::Tensor>& ref_ins,
                    std::vector<paddle::Tensor>* outs,
                    int64_t axis) {
  size_t num = outs->size();
  int64_t in_rows = GetRows(ref_ins[0].shape(), axis);
  int64_t in_cols = 0;
  auto out_cols = GetCols(ref_ins, in_rows, &in_cols);

  for (size_t i = 0; i < in_rows; ++i) {
    auto* in_data = in.data<data_t>() + i * in_cols;
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = out_cols[j];
      auto* out_data = outs->at(j).mutable_data<data_t>() + i * col_len;
      std::memcpy(out_data, in_data + col_idx, sizeof(data_t) * col_len);
      col_idx += col_len;
    }
  }
}
