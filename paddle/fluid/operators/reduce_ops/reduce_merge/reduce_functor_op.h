// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

// template <typename T>
// struct IdentityFunctor {
//   HOSTDEVICE explicit inline IdentityFunctor() {}
//
//   HOSTDEVICE inline T operator()(const T& x) const { return x; }
// };
//
// template <typename T>
// struct DivideFunctor {
//   HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}
//
//   HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }
//
//  private:
//   T n_inv;
// };

}  // namespace operators
}  // namespace paddle
