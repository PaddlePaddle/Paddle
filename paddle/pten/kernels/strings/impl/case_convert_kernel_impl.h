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
#include "paddle/pten/kernels/strings/case_utils.h"

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
      ascii_converter(dev_ctx, in_ptr, out_ptr, num);
    } else {
      utf8_converter(dev_ctx, in_ptr, out_ptr, num);
    }
  }
};

// For CPUContext
template <typename DeviceContext, typename CharConverter>
struct AsciiCaseConverter {
  void operator()(const DeviceContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    paddle::platform::Transform<DeviceContext> trans;
    for (size_t i = 0; i < num; ++i) {
      trans(
          dev_ctx, in[i].begin(), in[i].end(), out[i].mdata(), CharConverter());
    }
  }
};

template <typename DeviceContext,
          template <typename DeviceContextT> typename CharConverter>
struct UTF8CaseConverter {
  void operator()(const DeviceContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    paddle::platform::Transform<DeviceContext> trans;
    auto unicode_flag_map =
        strings::UnicodeFlagMap<DeviceContext, uint8_t>::Instance()->data();
    auto cases_map =
        strings::UnicodeFlagMap<DeviceContext, uint16_t>::Instance()->data();
    for (size_t i = 0; i < num; ++i) {
      uint32_t unicode_len = get_unicode_str_len(in[i].data(), in[i].size());
      std::vector<uint32_t> unicode_in(unicode_len, 0);
      get_unicode_str(in[i].data(), unicode_in.data(), unicode_len);
      trans(dev_ctx,
            unicode_in.begin(),
            unicode_in.end(),
            unicode_in.begin(),
            CharConverter<DeviceContext>(unicode_flag_map, cases_map));
      uint32_t utf8_len = get_utf8_str_len(unicode_in.data(), unicode_len);
      std::vector<char> result(utf8_len, 0);
      get_utf8_str(unicode_in.data(), result.data(), unicode_len);
      out[i] = result.data();
    }
  }
};

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToLower>,
                          UTF8CaseConverter<ContextT, UTF8ToLower>,
                          ContextT>()(dev_ctx, x, encoding, out);
}

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToUpper>,
                          UTF8CaseConverter<ContextT, UTF8ToUpper>,
                          ContextT>()(dev_ctx, x, encoding, out);
}

}  // namespace strings
}  // namespace pten
