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

#pragma once

#include <string>

#include "paddle/phi/backends/all_context.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
class FCFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const int M,
                  const int N,
                  const int K,
                  const T* X,
                  const T* W,
                  T* Y,
                  const T* B = nullptr,
                  bool relu = false,
                  bool weight_pass = false);
};

}  // namespace funcs
}  // namespace phi
