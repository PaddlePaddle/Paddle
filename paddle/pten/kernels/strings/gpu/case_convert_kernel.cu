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

#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/strings/case_convert_kernel.h"
#include "paddle/pten/kernels/strings/case_utils.h"
#include "paddle/pten/kernels/strings/unicode.h"

using pstring = ::pten::dtype::pstring;
namespace pten {
namespace strings {

using AsciiLowerConverter =
    pten::strings::AsciiCaseConverter<GPUContext, pten::strings::AsciiToLower>;
using AsciiUpperConverter =
    pten::strings::AsciiCaseConverter<GPUContext, pten::strings::AsciiToUpper>;

using UTF8LowerConverter =
    pten::strings::UTF8CaseConverter<GPUContext, pten::strings::UTF8ToLower>;
using UTF8UpperConverter =
    pten::strings::UTF8CaseConverter<GPUContext, pten::strings::UTF8ToUpper>;

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

__global__ void StringCaseConvertCUDAKernel(pstring* out,
                                            const pstring* in,
                                            size_t num) {
  CUDA_KERNEL_LOOP(i, num) {
    out[i] = pstring(in[i]);
    thrust::transform(thrust::device,
                      in[i].begin(),
                      in[i].end(),
                      out[i].mdata(),
                      pten::strings::AsciiToLower());
  }
}

template <typename AsciiCoverter, typename UTF8Converter>
struct StringCaseConvertKernel<AsciiCoverter, UTF8Converter, GPUContext> {
  void operator()(const GPUContext& dev_ctx,
                  const StringTensor& x,
                  const std::string& encoding,
                  StringTensor* out) {
    AsciiCoverter ascii_converter;
    UTF8Converter utf8_converter;
    const pstring* in_ptr = x.data();
    pstring* out_ptr = out->mutable_data();
    auto num = x.numel();
    VLOG(0) << "StringCaseConvertKernel GPUContext";
    if (encoding.empty()) {
      StringCaseConvertCUDAKernel<<<1, 32>>>(out_ptr, in_ptr, num);
    } else {
      StringCaseConvertCUDAKernel<<<1, 32>>>(out_ptr, in_ptr, num);
      // StringCaseConvertCUDAKernel<UTF8Converter><<<1, 32>>>(&dev_ctx,
      // out_ptr, in_ptr, num);
    }
  }
};

}  // namespace strings
}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(string_lower,
                           GPU,
                           ALL_LAYOUT,
                           pten::strings::StringLowerKernel<pten::GPUContext>,
                           pstring) {}

PT_REGISTER_GENERAL_KERNEL(string_upper,
                           GPU,
                           ALL_LAYOUT,
                           pten::strings::StringUpperKernel<pten::GPUContext>,
                           pstring) {}
