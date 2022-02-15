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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/strings/case_utils.h"
#include "paddle/pten/kernels/strings/impl/case_convert_kernel_impl.h"

using pstring = ::pten::dtype::pstring;

PT_REGISTER_GENERAL_KERNEL(strings_lower,
                           CPU,
                           ALL_LAYOUT,
                           pten::strings::StringLowerKernel<pten::CPUContext>,
                           pstring) {}

PT_REGISTER_GENERAL_KERNEL(strings_upper,
                           CPU,
                           ALL_LAYOUT,
                           pten::strings::StringUpperKernel<pten::CPUContext>,
                           pstring) {}
