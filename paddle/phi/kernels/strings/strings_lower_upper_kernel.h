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

#pragma once
#include <algorithm>
#include <vector>

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/strings/unary.h"
#include "paddle/phi/kernels/strings/case_utils.h"

using pstring = ::phi::dtype::pstring;

namespace phi {
namespace strings {

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out);

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out);

template <typename ContextT>
StringTensor StringLower(const ContextT& dev_ctx,
                         const StringTensor& x,
                         bool use_utf8_encoding) {
  StringTensor string_out;
  MetaTensor meta_out(&string_out);
  UnchangedInferMeta(x.meta(), &meta_out);
  StringLowerKernel(dev_ctx, x, use_utf8_encoding, &string_out);
  return string_out;
}

template <typename ContextT>
StringTensor StringUpper(const ContextT& dev_ctx,
                         const StringTensor& x,
                         bool use_utf8_encoding) {
  StringTensor string_out;
  MetaTensor meta_out(&string_out);
  UnchangedInferMeta(x.meta(), &meta_out);
  StringUpperKernel(dev_ctx, x, use_utf8_encoding, &string_out);
  return string_out;
}

template <typename AsciiCoverter, typename UTF8Converter, typename ContextT>
struct StringCaseConvertKernel {
  void operator()(const ContextT& dev_ctx,
                  const StringTensor& x,
                  bool use_utf8_encoding,
                  StringTensor* out) {
    AsciiCoverter ascii_converter;
    UTF8Converter utf8_converter;
    const pstring* in_ptr = x.data();
    pstring* out_ptr = dev_ctx.template Alloc<pstring>(out);
    auto num = x.numel();
    if (!use_utf8_encoding) {
      ascii_converter(dev_ctx, in_ptr, out_ptr, num);
    } else {
      utf8_converter(dev_ctx, in_ptr, out_ptr, num);
    }
  }
};

template <typename DeviceContext, typename CharConverter>
struct AsciiCaseConverter {
  void operator()(const DeviceContext& dev_ctx UNUSED,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    for (size_t i = 0; i < num; ++i) {
      std::transform(
          in[i].begin(), in[i].end(), out[i].mdata(), CharConverter());
    }
  }
};

template <typename DeviceContext,
          template <typename DeviceContextT>
          class CharConverter>
struct UTF8CaseConverter {
  void operator()(const DeviceContext& dev_ctx UNUSED,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    auto unicode_flag_map = GetUniFlagMap();
    auto cases_map = GetCharcasesMap();
    for (size_t i = 0; i < num; ++i) {
      uint32_t unicode_len = GetUnicodeStrLen(in[i].data(), in[i].size());
      std::vector<uint32_t> unicode_in(unicode_len, 0);
      GetUnicodeStr(in[i].data(), unicode_in.data(), unicode_len);
      std::transform(unicode_in.begin(),
                     unicode_in.end(),
                     unicode_in.begin(),
                     CharConverter<DeviceContext>(unicode_flag_map, cases_map));
      uint32_t utf8_len = GetUTF8StrLen(unicode_in.data(), unicode_len);
      std::vector<char> result(utf8_len, 0);
      GetUTF8Str(unicode_in.data(), result.data(), unicode_len);
      out[i] = result.data();
    }
  }
};

}  // namespace strings
}  // namespace phi
