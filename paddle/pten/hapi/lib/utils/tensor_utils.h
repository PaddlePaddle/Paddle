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

#pragma once

#include <memory>

#include "paddle/fluid/framework/lod_tensor.h"

#include "paddle/pten/core/candidate/dense_tensor.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/hapi/lib/utils/allocator.h"
#include "paddle/pten/hapi/lib/utils/storage.h"

namespace paddle {
namespace experimental {

using namespace pten::candidate;  // NOLINT

template <typename DstLoD, typename SrcLoD>
void SetLoD(DstLoD* dst, const SrcLoD& src) {
  dst->reserve(src.size());
  dst->clear();
  for (auto&& v : src) {
    dst->emplace_back(v);
  }
}

std::shared_ptr<DenseTensor> MakeSharedDenseTensor(
    const paddle::framework::Tensor& src) {
  DenseTensorMeta meta{pten::TransToPtenDataType(src.type()),
                       src.dims(),
                       pten::TransToPtenDataLayout(src.layout())};
  auto shared_storage = pten::make_intrusive<SharedStorage>(src.Holder());
  return std::make_shared<DenseTensor>(std::move(shared_storage),
                                       std::move(meta));
}

std::shared_ptr<DenseTensor> MakeSharedDenseTensor(
    const paddle::framework::LoDTensor& src) {
  DenseTensorMeta meta{pten::TransToPtenDataType(src.type()),
                       src.dims(),
                       pten::TransToPtenDataLayout(src.layout())};
  SetLoD(&meta.lod, src.lod());
  auto shared_storage = pten::make_intrusive<SharedStorage>(src.Holder());
  return std::make_shared<DenseTensor>(std::move(shared_storage),
                                       std::move(meta));
}

void MovesStorage(DenseTensor* src, paddle::framework::Tensor* dst) {
  CHECK(src);
  CHECK(dst);
  dst->Resize(src->dims());
  auto storage = src->release();
  CHECK(storage->OwnsMemory());
  std::shared_ptr<paddle::memory::allocation::Allocation> holder(
      new TensorStorage(std::move(storage)));
  dst->ResetHolderWithType(holder, pten::TransToProtoVarType(src->data_type()));
}

void MovesStorage(DenseTensor* src, paddle::framework::LoDTensor* dst) {
  CHECK(src);
  CHECK(dst);
  SetLoD(dst->mutable_lod(), src->lod());
  MovesStorage(src, static_cast<paddle::framework::Tensor*>(dst));
}

}  // namespace experimental
}  // namespace paddle
