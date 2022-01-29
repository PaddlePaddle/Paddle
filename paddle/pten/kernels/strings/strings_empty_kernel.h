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

#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"

namespace pten {
namespace strings {

template <typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const ScalarArray& shape,
                 StringTensor* out);

template <typename Context>
void EmptyLikeKernel(const Context& dev_ctx, StringTensor* out);

// TODO(zhoushunjie): the tensor creation method need to be replaced later,
// all kernel api call Empty here instead of making tensor self
template <typename Context>
StringTensor Empty(const Context& dev_ctx, StringTensorMeta&& meta) {
  pten::StringTensor string_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(meta));
  return string_out;
}

template <typename Context>
StringTensor Empty(const Context& dev_ctx) {
  return Empty<Context>(dev_ctx, {{-1}});
}

template <typename Context>
StringTensor Empty(const Context& dev_ctx, const ScalarArray& shape) {
  auto out_meta = CreateInferMeta(shape);
  auto string_out = Empty<Context>(dev_ctx, std::move(out_meta));
  EmptyKernel<Context>(dev_ctx, shape, &string_out);
  return string_out;
}

template <typename Context>
StringTensor EmptyLike(const Context& dev_ctx, const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  auto string_out = Empty<Context>(dev_ctx, std::move(out_meta));
  EmptyLikeKernel<Context>(dev_ctx, &string_out);
  return string_out;
}

}  // namespace strings
}  // namespace pten
