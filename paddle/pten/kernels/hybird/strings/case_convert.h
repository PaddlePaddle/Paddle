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
#include <utf8proc.h>

#include <codecvt>
#include <string>

#include "paddle/fluid/platform/transform.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/platform/pstring.h"

namespace pten {
namespace strings {

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
  HOSTDEVICE AsciiCaseConverter(const DeviceContext& dev_ctx)
      : dev_ctx_(dev_ctx) {}
  HOSTDEVICE pten::platform::pstring operator()(
      const pten::platform::pstring& in) const {
    paddle::platform::Transform<DeviceContext> trans;
    pten::platform::pstring output = in;

    trans(dev_ctx_, in.begin(), in.end(), output.mdata(), CharConverter());
    return output;
  }

 private:
  const DeviceContext& dev_ctx_;
};

struct UTF8ToLower {
  HOSTDEVICE char32_t operator()(char32_t in) const {
    return utf8proc_tolower(in);
  }
};

struct UTF8ToUpper {
  HOSTDEVICE char32_t operator()(char32_t in) const {
    return utf8proc_toupper(in);
  }
};

template <typename DeviceContext, typename CharConverter>
struct UTF8CaseConverter {
  HOSTDEVICE UTF8CaseConverter(const DeviceContext& dev_ctx)
      : dev_ctx_(dev_ctx) {}
  HOSTDEVICE pten::platform::pstring operator()(
      const pten::platform::pstring& in) const {
    paddle::platform::Transform<DeviceContext> trans;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> utf32conv;
    std::u32string unicode_in = utf32conv.from_bytes(in.begin(), in.end());
    trans(dev_ctx_,
          unicode_in.begin(),
          unicode_in.end(),
          unicode_in.begin(),
          CharConverter());
    return utf32conv.to_bytes(unicode_in);
  }

 private:
  const DeviceContext& dev_ctx_;
};

template <typename DeviceContext,
          typename AsciiConverter,
          typename UTF8Converter>
void StringCaseConvert(const DeviceContext& dev_ctx,
                       const DenseTensor& x,
                       const std::string& encoding,
                       DenseTensor* out) {
  AsciiConverter ascii_converter(dev_ctx);
  UTF8Converter utf8_converter(dev_ctx);
  auto* in_ptr = x.data<pten::platform::pstring>();
  auto* out_ptr = out->mutable_data<pten::platform::pstring>();
  auto num = x.numel();
  paddle::platform::Transform<DeviceContext> trans;
  if (encoding.empty()) {
    trans(dev_ctx, in_ptr, in_ptr + num, out_ptr, ascii_converter);
  } else {
    trans(dev_ctx, in_ptr, in_ptr + num, out_ptr, utf8_converter);
  }
}

}  // namespace strings
}  // namespace pten
