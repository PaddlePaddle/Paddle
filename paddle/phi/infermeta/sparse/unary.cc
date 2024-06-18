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
#include "paddle/phi/core/infermeta_utils.h"

namespace phi::sparse {

void IndicesInferMeta(const MetaTensor& x, MetaTensor* out) {
  // TODO(zhangkaihuo) Currently, we cannot get sparse_dim from tensor.
  // correct shape is: shape[0] = x.sparse_dim()
  // In the 3D point cloud model:
  // the input x is 5-D tensor, non_zero_elements is 1-D tensor
  out->set_dims({x.dims().size() - 1, -1});
  out->set_dtype(DataType::INT32);
  out->set_layout(DataLayout::NCHW);
}

void ValuesInferMeta(const MetaTensor& x, MetaTensor* out) {
  const auto& x_dims = x.dims();
  out->set_dims({-1, x_dims[x_dims.size() - 1]});
  out->set_dtype(x.dtype());
  out->set_layout(x.layout());
}

void CastInferMeta(const MetaTensor& x,
                   DataType index_dtype,
                   DataType out_dtype,
                   MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_layout(x.layout());
  out->share_lod(x);
  // In inplace case, setting the dtype of out will reset the dtype of x at the
  // same time, which will cause bugs, so move the dtype setting of out to the
  // kernel

  if (!(out->is_same_tensor(x))) {
    out->set_dtype(out_dtype);
  }
}

}  // namespace phi::sparse
