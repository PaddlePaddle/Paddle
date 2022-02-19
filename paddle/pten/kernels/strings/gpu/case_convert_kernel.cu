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

#include "paddle/fluid/platform/device/gpu/gpu_helper.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/backends/gpu/gpu_launch_config.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/strings/case_utils.h"
#include "paddle/pten/kernels/strings/impl/case_convert_kernel_impl.h"
#include "paddle/pten/kernels/strings/unicode.h"

using pstring = ::pten::dtype::pstring;
namespace pten {
namespace strings {

template <typename CharConverter>
__global__ void StringCaseConvertCUDAKernel(pstring* out,
                                            const pstring* in,
                                            size_t num) {
  CUDA_KERNEL_LOOP(i, num) {
    out[i] = pstring(in[i]);
    thrust::transform(thrust::device,
                      in[i].begin(),
                      in[i].end(),
                      out[i].mdata(),
                      CharConverter());
  }
}

template <typename CharConverter>
struct AsciiCaseConverter<pten::GPUContext, CharConverter> {
  void operator()(const pten::GPUContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3((num + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
    StringCaseConvertCUDAKernel<
        CharConverter><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
        out, in, num);
  }
};

template <template <typename DeviceContextT> typename CharConverter>
struct UTF8CaseConverter<pten::GPUContext, CharConverter> {
  void operator()(const pten::GPUContext& dev_ctx,
                  const pstring* in,
                  pstring* out,
                  size_t num) const {
    auto unicode_flag_map =
        strings::UnicodeFlagMap<pten::GPUContext, uint8_t>::Instance()->data();
    auto cases_map =
        strings::UnicodeFlagMap<pten::GPUContext, uint16_t>::Instance()->data();
    // paddle::platform::Transform<GPUContext> trans;
    // uint32_t unicode_len =
    //     pten::strings::get_unicode_str_len(in.data(), in.size());
    // thrust::device_vector<uint32_t> unicode_in(unicode_len, 0);
    // uint32_t* unicode_raw_ptr = thrust::raw_pointer_cast(unicode_in.data());
    // pten::strings::get_unicode_str(in.data(), unicode_raw_ptr, unicode_len);
    // auto unicode_flag_map =
    //     strings::UnicodeFlagMap<GPUContext, uint8_t>::Instance()->data();
    // auto cases_map =
    //     strings::UnicodeFlagMap<GPUContext, uint16_t>::Instance()->data();
    // trans(dev_ctx,
    //       unicode_in.begin(),
    //       unicode_in.end(),
    //       unicode_in.begin(),
    //       CharConverter<GPUContext>(unicode_flag_map, cases_map));
    // uint32_t utf8_len =
    //     pten::strings::get_utf8_str_len(unicode_raw_ptr, unicode_len);
    // thrust::device_vector<char> result(utf8_len, 0);
    // char* result_ptr = thrust::raw_pointer_cast(result.data());
    // pten::strings::get_utf8_str(unicode_raw_ptr, result_ptr, unicode_len);
    // *out = result_ptr;
  }
};

}  // namespace strings
}  // namespace pten

PT_REGISTER_GENERAL_KERNEL(strings_lower,
                           GPU,
                           ALL_LAYOUT,
                           pten::strings::StringLowerKernel<pten::GPUContext>,
                           pstring) {}

PT_REGISTER_GENERAL_KERNEL(strings_upper,
                           GPU,
                           ALL_LAYOUT,
                           pten::strings::StringUpperKernel<pten::GPUContext>,
                           pstring) {}
