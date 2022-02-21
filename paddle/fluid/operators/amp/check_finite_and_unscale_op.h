/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>

#include "paddle/fluid/operators/isfinite_op.h"
#include "paddle/phi/core/hostdevice.h"

namespace paddle {
namespace operators {

template <typename T>
inline HOSTDEVICE T Inverse(T s) {
  return 1.0 / s;
}

}  // namespace operators
}  // namespace paddle
