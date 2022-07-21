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

#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/dense_tensor.h"

// TODO(paddle-dev): Remove this file when we can call related Kernel directly

namespace phi {
namespace funcs {

using DDim = phi::DDim;
static DDim UDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return phi::make_ddim(x_vec);
}

static DDim VHDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return phi::make_ddim(x_vec);
}

static DDim SDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  x_vec.erase(x_vec.end() - 1);  // rank - 1
  return phi::make_ddim(x_vec);
}

}  // namespace funcs
}  // namespace phi
