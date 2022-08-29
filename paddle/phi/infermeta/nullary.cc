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

void AssignValueInferMeta(const std::vector<int>& shape,
                          DataType dtype,
                          MetaTensor* out) {
  out->set_dims(phi::make_ddim(shape));
  out->set_dtype(dtype);
}

void CreateInferMeta(const IntArray& shape, DataType dtype, MetaTensor* out) {
  if (!shape.FromTensor()) {
    const auto& data = shape.GetData();
    for (size_t i = 0; i < data.size(); ++i) {
      PADDLE_ENFORCE_GE(
          data[i],
          0,
          phi::errors::InvalidArgument(
              "Each value of attribute 'shape' is expected to be no less "
              "than 0. But recieved: shape[%u] = %d; shape = [%s].",
              i,
              data[i],
              phi::make_ddim(data)));
    }
  }
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

void EyeInferMeta(const Scalar& num_rows,
                  const Scalar& num_columns,
                  DataType dtype,
                  MetaTensor* out,
                  MetaConfig config) {
  int64_t rows, columns;
  if (!config.is_runtime && num_rows.FromTensor()) {
    rows = -1;
  } else {
    rows = num_rows.to<int64_t>();
  }

  if (!config.is_runtime && num_columns.FromTensor()) {
    columns = -1;
  } else {
    columns = num_columns.to<int64_t>();
    if (columns == -1) columns = rows;
  }
  out->set_dims({rows, columns});
  out->set_dtype(dtype);
}

void GaussianRandomInferMeta(const IntArray& shape,
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

void RandpermInferMeta(int n, DataType dtype, MetaTensor* out) {
  out->set_dims(phi::make_ddim({n}));
  out->set_dtype(dtype);
}

void UniformRandomInferMeta(const IntArray& shape,
                            DataType dtype,
                            MetaTensor* out) {
  auto out_dims = phi::make_ddim(shape.GetData());
  out->set_dims(out_dims);
  out->set_dtype(dtype);
  out->set_layout(DataLayout::NCHW);
}

void RandintInferMeta(
    int low, int high, const IntArray& shape, DataType dtype, MetaTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(
      out, errors::InvalidArgument("Output(Out) of RandintOp is null."));
  PADDLE_ENFORCE_LT(
      low,
      high,
      errors::InvalidArgument("randint's low must less then high, "
                              "but received: low = %d, high = %d.",
                              low,
                              high));

  auto& shape_vector = shape.GetData();
  PADDLE_ENFORCE_EQ(
      shape_vector.empty(),
      false,
      errors::InvalidArgument("The shape information should not be empty, it "
                              "must be set by Attr(shape)."));

  std::vector<int64_t> tensor_shape;
  tensor_shape.reserve(shape_vector.size());
  for (auto dim : shape_vector) {
    tensor_shape.push_back(static_cast<int64_t>(dim));
  }
  out->set_dims(make_ddim(tensor_shape));
  out->set_dtype(dtype);
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

void TrilIndicesInferMeta(
    int rows, int cols, int offset, DataType dtype, MetaTensor* out) {
  // number of elements in the first row of the tril,bounded by [0, cols]
  auto n_first_row =
      offset > 0 ? std::min<int64_t>(cols, 1 + offset) : rows + offset > 0;
  // number of elements in the last row of the tril, bounded by [0, cols]
  auto n_last_row =
      std::max<int64_t>(0, std::min<int64_t>(cols, rows + offset));
  // number of rows, bounded by [0, rows]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(rows, rows + offset));
  auto n_row_trapezoid = (n_last_row - n_first_row + 1);
  // calculate # of elements in the top trapezoid
  auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * cols;
  }
  std::vector<int64_t> tmp = {2, tril_size};
  auto out_dims = phi::make_ddim(tmp);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
}

void TriuIndicesInferMeta(
    int row, int col, int offset, DataType dtype, MetaTensor* out) {
  // number of elements in the first row of the tril,bounded by [0, cols]
  // use total item number minus bottom rectangle item number to get
  // the above rectangle item number
  //     triu_size = rows * cols - tril_size
  // so the `offset` need to be set as `offset-1` in order to include
  // the item on the diagonal line
  offset = offset - 1;
  auto n_first_row =
      offset > 0 ? std::min<int64_t>(col, 1 + offset) : row + offset > 0;
  // number of elements in the last row of the tril, bounded by [0, cols]
  auto n_last_row = std::max<int64_t>(0, std::min<int64_t>(col, row + offset));
  // number of rows, bounded by [0, rows]
  auto n_row_all = std::max<int64_t>(0, std::min<int64_t>(row, row + offset));
  auto n_row_trapezoid = (n_last_row - n_first_row + 1);
  // calculate # of elements in the top trapezoid
  auto tril_size = (n_first_row + n_last_row) * n_row_trapezoid >> 1;
  // calculate # of elements in the bottom rectangle if there is any
  auto diff_row = n_row_all - n_row_trapezoid;
  if (diff_row > 0) {
    tril_size += diff_row * col;
  }
  std::vector<int64_t> tmp = {2, row * col - tril_size};
  auto out_dims = phi::make_ddim(tmp);
  out->set_dims(out_dims);
  out->set_dtype(dtype);
}
}  // namespace phi
