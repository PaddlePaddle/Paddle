// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/pstring.h"

namespace ap {

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;
using float16 = ::phi::dtype::float16;
using bfloat16 = ::phi::dtype::bfloat16;
using float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
using float8_e5m2 = ::phi::dtype::float8_e5m2;
using pstring = ::phi::dtype::pstring;

#define AP_FOR_EACH_INT_TYPE(_) \
  _(int8)                       \
  _(uint8)                      \
  _(int16)                      \
  _(uint16)                     \
  _(int32)                      \
  _(uint32)                     \
  _(int64)                      \
  _(uint64)

}  // namespace ap
