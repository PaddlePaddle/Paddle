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

void EyeInferMeta(int64_t num_rows,
                  int64_t num_columns,
                  DataType dtype,
                  MetaTensor* out) {
  if (num_columns == -1) num_columns = num_rows;
  out->set_dims({num_rows, num_columns});
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
                            float min,
                            float max,
                            int seed,
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

}  // namespace phi
