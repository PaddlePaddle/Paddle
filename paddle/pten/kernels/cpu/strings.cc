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

#include "paddle/pten/kernels/cpu/strings.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/platform/pstring.h"
#include "paddle/pten/kernels/hybird/strings/case_convert.h"

using pstring = ::pten::platform::pstring;

namespace pten {

using AsciiLowerConverter =
    pten::strings::AsciiCaseConverter<CPUContext, pten::strings::AsciiToLower>;
using AsciiUpperConverter =
    pten::strings::AsciiCaseConverter<CPUContext, pten::strings::AsciiToUpper>;

template <typename AsciiCoverter>
void StringCaseConvert(const CPUContext& dev_ctx,
                       const DenseTensor& x,
                       const std::string& encoding,
                       DenseTensor* out) {
  AsciiCoverter ascii_converter;
  const pstring* in_ptr = x.data<pstring>();
  pstring* out_ptr = out->mutable_data<pstring>();
  auto num = x.numel();
  for (int64_t i = 0; i < num; ++i) {
    if (encoding.empty()) {
      ascii_converter(dev_ctx, in_ptr[i], out_ptr + i);
    } else {
      // TODO(zhoushunjie): need to add utf-8 encoding converter
      ascii_converter(dev_ctx, in_ptr[i], out_ptr + i);
    }
  }
}

void StringLower(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const std::string& encoding,
                 DenseTensor* out) {
  StringCaseConvert<AsciiLowerConverter>(dev_ctx, x, encoding, out);
}

void StringUpper(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const std::string& encoding,
                 DenseTensor* out) {
  StringCaseConvert<AsciiUpperConverter>(dev_ctx, x, encoding, out);
}

}  // namespace pten

PT_REGISTER_NO_TEMPLATE_KERNEL(
    string_lower, CPU, ALL_LAYOUT, pten::StringLower, pstring) {}

PT_REGISTER_NO_TEMPLATE_KERNEL(
    string_upper, CPU, ALL_LAYOUT, pten::StringUpper, pstring) {}
