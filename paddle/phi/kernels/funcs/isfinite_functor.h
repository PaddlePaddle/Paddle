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

namespace phi {
namespace funcs {

template <typename T>
struct IsNanFunctor {
  HOSTDEVICE bool operator()(const T a) const { return std::isnan(a); }
};

template <typename T>
struct IsInfFunctor {
  HOSTDEVICE bool operator()(const T a) const { return std::isinf(a); }
};

template <typename T>
struct IsFiniteFunctor {
  HOSTDEVICE bool operator()(const T a) const { return std::isfinite(a); }
};

template <>
struct IsFiniteFunctor<phi::dtype::float16> {
  HOSTDEVICE bool operator()(const phi::dtype::float16 a) const {
    return isfinite(a);
  }
};

}  // namespace funcs
}  // namespace phi
