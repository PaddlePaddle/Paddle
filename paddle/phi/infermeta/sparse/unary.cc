/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/sparse/unary.h"
#include "paddle/phi/infermeta/unary.h"

#include "paddle/phi/core/infermeta_utils.h"

namespace phi {
namespace sparse {

void CreateLikeInferMeta(const MetaTensor& x, DataType dtype, MetaTensor* out) {
  phi::CreateLikeInferMeta(x, dtype, out);
}

void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void SparseCooTensorInferMeta(const MetaTensor& values,
                              const MetaTensor& indices,
                              const IntArray& dense_shape,
                              MetaTensor* out) {
  out->set_dims(phi::make_ddim(dense_shape.GetData()));
  out->set_dtype(values.dtype());
  out->set_layout(values.layout());
}

void ValuesInferMeta(const MetaTensor& x, MetaTensor* out) {
  const auto& x_dims = x.dims();
  out->set_dims({-1, x_dims[x_dims.size() - 1]});
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void IndicesInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->set_dims({-1});
  out->set_dtype(DataType::INT32);
  out->set_layout(DataLayout::NCHW);
}

void CastInferMeta(const MetaTensor& x,
                   const DataType index_dtype,
                   const DataType value_dtype,
                   MetaTensor* out) {
  phi::CastInferMeta(x, value_dtype, out);
}

}  // namespace sparse
}  // namespace phi
