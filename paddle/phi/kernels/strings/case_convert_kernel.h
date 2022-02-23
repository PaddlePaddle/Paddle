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
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/strings/strings_empty_kernel.h"

using pstring = ::phi::dtype::pstring;

namespace phi {
namespace strings {

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out);

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out);

template <typename ContextT>
StringTensor StringLower(const ContextT& dev_ctx,
                         const std::string& encoding,
                         const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  auto string_out = phi::strings::Empty<ContextT>(dev_ctx, std::move(out_meta));
  StringLowerKernel(dev_ctx, x, encoding, &string_out);
  return string_out;
}

template <typename ContextT>
StringTensor StringUpper(const ContextT& dev_ctx,
                         const std::string& encoding,
                         const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  auto string_out = phi::strings::Empty<ContextT>(dev_ctx, std::move(out_meta));
  StringUpperKernel(dev_ctx, x, encoding, &string_out);
  return string_out;
}

}  // namespace strings
}  // namespace phi
