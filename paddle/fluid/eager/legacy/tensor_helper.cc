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

#include "paddle/fluid/eager/legacy/tensor_helper.h"

#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/place.h"

namespace egr {
pten::TensorMeta* MutableMeta(egr::EagerTensor* tensor) {
  if (!tensor->defined()) {
    pten::TensorMeta meta;
    pten::TensorStatus status;
    auto dense_tensor = std::make_shared<pten::DenseTensor>(meta, status);
    tensor->set_impl(dense_tensor);
  }
  return std::static_pointer_cast<pten::DenseTensor>(tensor->impl())
      ->mutable_meta();
}
void InitializeTensor(egr::EagerTensor* tensor, const pten::TensorMeta& meta,
                      const pten::TensorStatus& status) {
  // TODO(jiabin) Support init Tensor with other kinds of TensorBase.
  if (!tensor->defined()) {
    auto dense_tensor = std::make_shared<pten::DenseTensor>(meta, status);
    tensor->set_impl(dense_tensor);
  }
}
void InitializeTensor(egr::EagerTensor* tensor) {
  // TODO(jiabin) Support init Tensor with other kinds of TensorBase.
  if (!tensor->defined()) {
    pten::TensorMeta meta;
    pten::TensorStatus status;
    auto dense_tensor = std::make_shared<pten::DenseTensor>(meta, status);
    tensor->set_impl(dense_tensor);
  }
}
}  // namespace egr
