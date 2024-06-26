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

extern "C" {
#include <xxhash.h>
}
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/hash_utils.h"

namespace phi {

template <typename T, typename Context>
void HashKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int num_hash,
                int64_t mod_by,
                DenseTensor* out) {
  auto* out_t = out;
  auto* in_t = &x;

  auto in_dims = in_t->dims();

  std::vector<int64_t> out_dims;
  phi::funcs::HashOutputSize(in_dims, out_dims, num_hash);
  out_t->Resize(common::make_ddim(out_dims));
  auto* output = dev_ctx.template Alloc<T>(out_t);

  auto seq_length = in_dims[0];
  auto last_dim = in_dims[in_dims.size() - 1];
  auto* input = in_t->data<T>();
  for (int idx = 0; idx < seq_length; ++idx) {
    for (int ihash = 0; ihash != num_hash; ++ihash) {
      output[idx * num_hash + ihash] =
          XXH64(input, sizeof(T) * last_dim, ihash) % mod_by;
    }
    input += last_dim;
  }

  out_t->set_lod(in_t->lod());
}

}  // namespace phi
