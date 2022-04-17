// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/hostdevice.h"

#include "math.h"  // NOLINT

namespace paddle {
namespace operators {

inline HOSTDEVICE platform::float16 real_exp(platform::float16 x) {
  return static_cast<platform::float16>(::expf(static_cast<float>(x)));
}

inline HOSTDEVICE float real_exp(float x) { return ::expf(x); }

inline HOSTDEVICE double real_exp(double x) { return ::exp(x); }

inline HOSTDEVICE platform::float16 real_log(platform::float16 x) {
  return static_cast<platform::float16>(::logf(static_cast<float>(x)));
}

inline HOSTDEVICE float real_log(float x) { return ::logf(x); }

inline HOSTDEVICE double real_log(double x) { return ::log(x); }

inline HOSTDEVICE float real_min(float x, float y) { return ::fminf(x, y); }

inline HOSTDEVICE double real_min(double x, double y) { return ::fmin(x, y); }

}  // namespace operators
}  // namespace paddle
