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
  HOSTDEVICE void operator()(const DeviceContext& dev_ctx,
                             const pten::platform::pstring& in,
                             pten::platform::pstring* out) const {
    paddle::platform::Transform<DeviceContext> trans;
    trans(dev_ctx, in.begin(), in.end(), out->data(), CharConverter());
  }
};

}  // namespace strings
}  // namespace pten
