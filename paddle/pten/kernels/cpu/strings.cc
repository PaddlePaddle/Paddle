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

using UTF8LowerConverter =
    pten::strings::UTF8CaseConverter<CPUContext, pten::strings::UTF8ToLower>;
using UTF8UpperConverter =
    pten::strings::UTF8CaseConverter<CPUContext, pten::strings::UTF8ToUpper>;

void StringLower(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const std::string& encoding,
                 DenseTensor* out) {
  pten::strings::StringCaseConvert<CPUContext,
                                   AsciiLowerConverter,
                                   UTF8LowerConverter>(
      dev_ctx, x, encoding, out);
}

void StringUpper(const CPUContext& dev_ctx,
                 const DenseTensor& x,
                 const std::string& encoding,
                 DenseTensor* out) {
  pten::strings::StringCaseConvert<CPUContext,
                                   AsciiUpperConverter,
                                   UTF8UpperConverter>(
      dev_ctx, x, encoding, out);
}

}  // namespace pten

PT_REGISTER_NO_TEMPLATE_KERNEL(
    string_lower, CPU, ALL_LAYOUT, pten::StringLower, pstring) {}

PT_REGISTER_NO_TEMPLATE_KERNEL(
    string_upper, CPU, ALL_LAYOUT, pten::StringUpper, pstring) {}
