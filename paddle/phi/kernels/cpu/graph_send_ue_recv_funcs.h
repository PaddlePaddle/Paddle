// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct GraphAddFunctor {
  inline T operator()(const T a, const T b) const { return a + b; }
};

template <typename T>
struct GraphMulFunctor {
  inline T operator()(const T a, const T b) const { return a * b; }
};

template <typename T>
struct GraphMaxFunctor {
  inline T operator()(const T a, const T b) const { return a < b ? b : a; }
};

template <typename T>
struct GraphMinFunctor {
  inline T operator()(const T a, const T b) const { return a < b ? a : b; }
};

}  // namespace phi
