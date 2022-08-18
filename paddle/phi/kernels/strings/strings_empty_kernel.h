// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/strings/nullary.h"
#include "paddle/phi/infermeta/strings/unary.h"

namespace phi {
namespace strings {

template <typename Context>
void EmptyKernel(const Context& dev_ctx,
                 const IntArray& shape,
                 StringTensor* out);

template <typename Context>
void EmptyLikeKernel(const Context& dev_ctx, StringTensor* out);

// TODO(zhoushunjie): the tensor creation method need to be replaced later,
// all kernel api call Empty here instead of making tensor self
template <typename Context>
StringTensor Empty(const Context& dev_ctx, StringTensorMeta&& meta) {
  auto allocator = std::make_unique<paddle::experimental::DefaultAllocator>(
      dev_ctx.GetPlace());
  phi::StringTensor string_out(allocator.get(), std::move(meta));
  return string_out;
}

template <typename Context>
StringTensor Empty(const Context& dev_ctx) {
  return Empty<Context>(dev_ctx, {{-1}});
}

template <typename Context>
StringTensor Empty(const Context& dev_ctx, const IntArray& shape) {
  StringTensor string_out;
  MetaTensor meta_out(&string_out);
  phi::strings::CreateInferMeta(shape, &meta_out);
  EmptyKernel<Context>(dev_ctx, shape, &string_out);
  return string_out;
}

template <typename Context>
StringTensor EmptyLike(const Context& dev_ctx, const StringTensor& x) {
  StringTensor string_out;
  MetaTensor meta_out(&string_out);
  phi::strings::UnchangedInferMeta(x.meta(), &meta_out);
  EmptyLikeKernel<Context>(dev_ctx, &string_out);
  return string_out;
}

}  // namespace strings
}  // namespace phi
