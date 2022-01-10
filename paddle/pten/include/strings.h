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
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/cpu/strings.h"

namespace pten {

template <typename ContextT>
StringTensor StringLower(const ContextT& dev_ctx,
                         const std::string& encoding,
                         const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  const auto alloc = std::make_shared<paddle::experimental::StringAllocator>(
      dev_ctx.GetPlace());
  pten::StringTensor dense_out(alloc, std::move(out_meta));
  StringLower(dev_ctx, x, encoding, &dense_out);
  return dense_out;
}

template <typename ContextT>
StringTensor StringUpper(const ContextT& dev_ctx,
                         const std::string& encoding,
                         const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  const auto alloc = std::make_shared<paddle::experimental::StringAllocator>(
      dev_ctx.GetPlace());
  pten::StringTensor dense_out(alloc, std::move(out_meta));
  StringUpper(dev_ctx, x, encoding, &dense_out);
  return dense_out;
}

}  // namespace pten
