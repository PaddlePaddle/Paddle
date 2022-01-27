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
#include <string>

#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/common/pstring.h"
#include "paddle/pten/kernels/strings/unicode.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "paddle/pten/backends/gpu/gpu_context.h"
#endif
namespace pten {
namespace strings {

using pstring = dtype::pstring;
struct AsciiToLower {
  HOSTDEVICE char operator()(char in) const {
    return ('A' <= in && in <= 'Z') ? in - ('Z' - 'z') : in;
  }
};

struct AsciiToUpper {
  HOSTDEVICE char operator()(char in) const {
    return ('a' <= in && in <= 'z') ? in ^ 0x20 : in;
  }
};

template <typename DeviceContext, typename CharConverter>
struct AsciiCaseConverter {
  void operator()(const DeviceContext& dev_ctx,
                  const pstring& in,
                  pstring* out) const {
    paddle::platform::Transform<DeviceContext> trans;
    trans(dev_ctx, in.begin(), in.end(), out->mdata(), CharConverter());
  }
};

template <typename DeviceContext>
struct UTF8ToLower {
  HOSTDEVICE UTF8ToLower(uint8_t* unicode_flag_map, uint16_t* cases_map)
      : unicode_flag_map_(unicode_flag_map), cases_map_(cases_map) {}

  HOSTDEVICE uint32_t operator()(uint32_t in) const {
    uint32_t flg = (in <= 0x00FFFF ? unicode_flag_map_[in] : 0);
    return (strings::isupper(flg) ? cases_map_[in] : in);
  }

  uint8_t* unicode_flag_map_;
  uint16_t* cases_map_;
};

template <typename DeviceContext>
struct UTF8ToUpper {
  HOSTDEVICE UTF8ToUpper(uint8_t* unicode_flag_map, uint16_t* cases_map)
      : unicode_flag_map_(unicode_flag_map), cases_map_(cases_map) {}

  HOSTDEVICE uint32_t operator()(uint32_t in) const {
    uint32_t flg = (in <= 0x00FFFF ? unicode_flag_map_[in] : 0);
    return (strings::islower(flg) ? cases_map_[in] : in);
  }

  uint8_t* unicode_flag_map_;
  uint16_t* cases_map_;
};

template <typename DeviceContext,
          template <typename DeviceContextT> typename CharConverter>
struct UTF8CaseConverter {
  void operator()(const DeviceContext& dev_ctx,
                  const pstring& in,
                  pstring* out) const {
    paddle::platform::Transform<DeviceContext> trans;
    uint32_t unicode_len =
        pten::strings::get_unicode_str_len(in.data(), in.size());
    std::vector<uint32_t> unicode_in(unicode_len, 0);
    pten::strings::get_unicode_str(in.data(), unicode_in.data(), unicode_len);
    auto unicode_flag_map =
        strings::UnicodeFlagMap<DeviceContext, uint8_t>::Instance()->data();
    auto cases_map =
        strings::UnicodeFlagMap<DeviceContext, uint16_t>::Instance()->data();
    trans(dev_ctx,
          unicode_in.begin(),
          unicode_in.end(),
          unicode_in.begin(),
          CharConverter<DeviceContext>(unicode_flag_map, cases_map));
    uint32_t utf8_len =
        pten::strings::get_utf8_str_len(unicode_in.data(), unicode_len);
    std::vector<char> result(utf8_len, 0);
    pten::strings::get_utf8_str(unicode_in.data(), result.data(), unicode_len);
    *out = result.data();
  }
};

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename CharConverter>
struct AsciiCaseConverter<GPUContext, CharConverter> {
  void operator()(const GPUContext& dev_ctx,
                  const pstring& in,
                  pstring* out) const {
    paddle::platform::Transform<GPUContext> trans;
    trans(dev_ctx, in.begin(), in.end(), out->mdata(), CharConverter());
  }
};

template <template <typename DeviceContextT> typename CharConverter>
struct UTF8CaseConverter<GPUContext, CharConverter> {
  void operator()(const GPUContext& dev_ctx,
                  const pstring& in,
                  pstring* out) const {
    paddle::platform::Transform<GPUContext> trans;
    uint32_t unicode_len =
        pten::strings::get_unicode_str_len(in.data(), in.size());
    thrust::device_vector<uint32_t> unicode_in(unicode_len, 0);
    uint32_t* unicode_raw_ptr = thrust::raw_pointer_cast(unicode_in.data());
    pten::strings::get_unicode_str(in.data(), unicode_raw_ptr, unicode_len);
    auto unicode_flag_map =
        strings::UnicodeFlagMap<GPUContext, uint8_t>::Instance()->data();
    auto cases_map =
        strings::UnicodeFlagMap<GPUContext, uint16_t>::Instance()->data();
    trans(dev_ctx,
          unicode_in.begin(),
          unicode_in.end(),
          unicode_in.begin(),
          CharConverter<GPUContext>(unicode_flag_map, cases_map));
    uint32_t utf8_len =
        pten::strings::get_utf8_str_len(unicode_raw_ptr, unicode_len);
    thrust::device_vector<char> result(utf8_len, 0);
    char* result_ptr = thrust::raw_pointer_cast(result.data());
    pten::strings::get_utf8_str(unicode_raw_ptr, result_ptr, unicode_len);
    *out = result_ptr;
  }
};
#endif

}  // namespace strings
}  // namespace pten
