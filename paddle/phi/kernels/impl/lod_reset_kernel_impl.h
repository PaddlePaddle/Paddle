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

#include <algorithm>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void LodResetKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const paddle::optional<DenseTensor>& y,
                    const std::vector<int>& target_lod,
                    bool append,
                    DenseTensor* out) {
  auto* in = &x;
  auto* lod_t = y.get_ptr();

  phi::Copy(dev_ctx, *in, in->place(), false, out);

  std::vector<int> level0;
  if (lod_t) {
    if (lod_t->lod().size() > 0) {
      auto y_lod = lod_t->lod();
      auto last_level = y_lod[y_lod.size() - 1];
      PADDLE_ENFORCE_EQ(
          static_cast<int64_t>(last_level.back()),
          in->dims()[0],
          common::errors::InvalidArgument(
              "The last value of Input(Y)'s last level LoD should be equal "
              "to the first dimension of Input(X). But received the last "
              "value of Input(Y)'s last level LoD is %d, the first dimension "
              "of Input(X) is %d.",
              static_cast<int64_t>(last_level.back()),
              in->dims()[0]));
      out->set_lod(y_lod);
      return;  // early return, since lod already set
    } else {
      auto* lod = lod_t->data<int>();
      phi::DenseTensor lod_cpu;
      if (lod_t->place().GetType() == phi::AllocationType::GPU) {
        phi::Copy(dev_ctx, *lod_t, phi::CPUPlace(), true, &lod_cpu);
        lod = lod_cpu.data<int>();
      }
      level0 = std::vector<int>(lod, lod + lod_t->numel());
    }
  } else {
    level0 = target_lod;
  }

  PADDLE_ENFORCE_GT(
      level0.size(),
      1UL,
      common::errors::InvalidArgument(
          "The size of target LoD should be greater than 1. But received the "
          "size of target LoD is %d.",
          level0.size()));
  PADDLE_ENFORCE_EQ(static_cast<int64_t>(level0[0]),
                    0,
                    common::errors::InvalidArgument(
                        "Target LoD should be a vector starting from 0. But "
                        "target LoD starts from %d.",
                        static_cast<int64_t>(level0[0])));
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(level0.back()),
      in->dims()[0],
      common::errors::InvalidArgument(
          "The last value of 'Target LoD''s last level LoD should be equal "
          "to the first dimension of Input(X). But received the 'Target LoD' "
          "is %s, Input(X)'s shape is %s.",
          common::make_ddim(level0),
          in->dims()));
  for (size_t i = 0; i < level0.size() - 1; ++i) {
    PADDLE_ENFORCE_GE(level0[i + 1],
                      level0[i],
                      common::errors::InvalidArgument(
                          "'Target LoD' should be an ascending "
                          "vector. But received the Target LoD is %s.",
                          common::make_ddim(level0)));
  }

  // cast level0 to size_t
  std::vector<size_t> ulevel0(level0.size(), 0);
  std::transform(level0.begin(), level0.end(), ulevel0.begin(), [](int a) {
    return static_cast<size_t>(a);
  });
  if (append) {
    auto* out_lod = out->mutable_lod();
    out_lod->push_back(ulevel0);
  } else {
    phi::LegacyLoD target_lod;
    target_lod.push_back(ulevel0);
    out->set_lod(target_lod);
  }
}

template <typename T, typename Context>
void LodResetGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& out_grad,
                        const std::vector<int>& target_lod,
                        bool append,
                        DenseTensor* x_grad) {
  auto* d_out = &out_grad;
  auto* d_x = x_grad;

  phi::Copy(dev_ctx, *d_out, d_out->place(), false, d_x);
}

}  // namespace phi
