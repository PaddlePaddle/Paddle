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
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/infermeta/unary.h"

using pstring = ::pten::dtype::pstring;

namespace pten {
namespace strings {
template <typename AsciiCoverter, typename UTF8Converter, typename ContextT>
struct StringCaseConvertKernel {
  void operator()(const ContextT& dev_ctx,
                  const StringTensor& x,
                  const std::string& encoding,
                  StringTensor* out) {
    AsciiCoverter ascii_converter;
    UTF8Converter utf8_converter;
    const pstring* in_ptr = x.data();
    pstring* out_ptr = out->mutable_data();
    auto num = x.numel();
    if (encoding.empty()) {
      for (int64_t i = 0; i < num; ++i) {
        ascii_converter(dev_ctx, in_ptr[i], out_ptr + i);
      }
    } else {
      for (int64_t i = 0; i < num; ++i) {
        utf8_converter(dev_ctx, in_ptr[i], out_ptr + i);
      }
    }
  }
};

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
  pten::StringTensor string_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  StringLowerKernel(dev_ctx, x, encoding, &string_out);
  return string_out;
}

template <typename ContextT>
StringTensor StringUpper(const ContextT& dev_ctx,
                         const std::string& encoding,
                         const StringTensor& x) {
  auto out_meta = UnchangedInferMeta(x.meta());
  pten::StringTensor string_out(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(out_meta));
  StringUpperKernel(dev_ctx, x, encoding, &string_out);
  return string_out;
}

}  // namespace strings
}  // namespace pten
