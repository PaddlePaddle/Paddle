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

#include "paddle/phi/kernels/strings/strings_lower_upper_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/kernel_registry.h"

using pstring = ::phi::dtype::pstring;

namespace phi {
namespace strings {

template <typename ContextT>
void StringLowerKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToLower>,
                          UTF8CaseConverter<ContextT, UTF8ToLower>,
                          ContextT>()(dev_ctx, x, use_utf8_encoding, out);
}

template <typename ContextT>
void StringUpperKernel(const ContextT& dev_ctx,
                       const StringTensor& x,
                       bool use_utf8_encoding,
                       StringTensor* out) {
  StringCaseConvertKernel<AsciiCaseConverter<ContextT, AsciiToUpper>,
                          UTF8CaseConverter<ContextT, UTF8ToUpper>,
                          ContextT>()(dev_ctx, x, use_utf8_encoding, out);
}

}  // namespace strings
}  // namespace phi

PD_REGISTER_GENERAL_KERNEL(strings_lower,
                           CPU,
                           ALL_LAYOUT,
                           phi::strings::StringLowerKernel<phi::CPUContext>,
                           pstring) {}

PD_REGISTER_GENERAL_KERNEL(strings_upper,
                           CPU,
                           ALL_LAYOUT,
                           phi::strings::StringUpperKernel<phi::CPUContext>,
                           pstring) {}
