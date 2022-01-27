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

#include "paddle/pten/kernels/strings/case_convert_kernel.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/strings/case_utils.h"

using pstring = ::pten::dtype::pstring;

namespace pten {
namespace strings {

using AsciiLowerConverter =
    pten::strings::AsciiCaseConverter<CPUContext, pten::strings::AsciiToLower>;
using AsciiUpperConverter =
    pten::strings::AsciiCaseConverter<CPUContext, pten::strings::AsciiToUpper>;

using UTF8LowerConverter =
    pten::strings::UTF8CaseConverter<CPUContext, pten::strings::UTF8ToLower>;
using UTF8UpperConverter =
    pten::strings::UTF8CaseConverter<CPUContext, pten::strings::UTF8ToUpper>;

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiLowerConverter, UTF8LowerConverter, ContextT>()(
      dev_ctx, x, encoding, out);
}

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       const std::string& encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiUpperConverter, UTF8UpperConverter, ContextT>()(
      dev_ctx, x, encoding, out);
}

}  // namespace strings
}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(string_lower,
                           CPU,
                           ALL_LAYOUT,
                           pten::strings::StringLowerKernel<pten::CPUContext>,
                           pstring) {}

PT_REGISTER_GENERAL_KERNEL(string_upper,
                           CPU,
                           ALL_LAYOUT,
                           pten::strings::StringUpperKernel<pten::CPUContext>,
                           pstring) {}
