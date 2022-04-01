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

#include "paddle/phi/infermeta/nullary.h"

namespace phi {

void CreateInferMeta(const ScalarArray& shape,
                     DataType dtype,
                     MetaTensor* out) {
  CreateInferMetaBase(shape.GetData(), dtype, DataLayout::NCHW, out);
}

void CreateInferMetaBase(const std::vector<int64_t>& shape,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out) {
  auto out_dims = phi::make_ddim(shape);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(layout);
}

void EyeInferMeta(int64_t num_rows,
                  int64_t num_columns,
                  DataType dtype,
                  MetaTensor* out) {
  if (num_columns == -1) num_columns = num_rows;
  out->set_dims({num_rows, num_columns});
  out->set_dtype(dtype);
}

void GaussianRandomInferMeta(const ScalarArray& shape,
                             float mean,
                             float std,
                             int seed,
                             DataType dtype,
                             MetaTensor* out) {
  auto out_dims = phi::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void TruncatedGaussianRandomInferMeta(const std::vector<int>& shape,
                                      float mean,
                                      float std,
                                      int seed,
                                      DataType dtype,
                                      MetaTensor* out) {
  auto out_dims = phi::make_ddim(shape);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void TrilIndicesInferMeta(int rows,
                          int cols,
                          int offset,
                          DataType dtype,
                          MetaTensor* out){
                         
  // number of elements in the first row of the tril,bounded by [0, cols]
  auto m_first_row = offset > 0 ?std::min<int64_t>(cols, 1 + offset) : rows + offset > 0; 
  // number of elements in the last row of the tril, bounded by [0, cols]
  auto m_last_row = std::max<int64_t>(0, std::min<int64_t>(cols, rows + offset));
  // number of rows, bounded by [0, rows]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(rows, rows + offset));
  auto n_row_trapezoid = (m_last_row - m_first_row + 1);
  // calculate # of elements in the top trapezoid
  auto tril_size = (m_first_row + m_last_row) * n_row_trapezoid >> 1;

  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * cols;
  }
  std :: vector<int64_t> tmp ;
  tmp.push_back(2);tmp.push_back(tril_size);
  auto out_dims = phi::make_ddim(tmp);
  out->set_dims(out_dims);
  out->set_dtype(dtype);

  }

}  // namespace phi
