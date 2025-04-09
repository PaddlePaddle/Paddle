/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/sequence_pooling.h"

namespace phi {

template <typename T, typename Context>
void SequencePoolKernel(const Context& ctx,
                        const DenseTensor& x,
                        bool is_test,
                        const std::string& pooltype,
                        float pad_value,
                        DenseTensor* out,
                        DenseTensor* max_index) {
  T pad_value_ = static_cast<T>(pad_value);

  auto dims = x.dims();
  auto lod = x.lod();
  auto lod_level = lod.size();
  // InferShape by lod
  PADDLE_ENFORCE_GT(
      lod_level,
      0,
      errors::InvalidArgument("Input(X) phi::DenseTensor of SequencePoolOp "
                              "does not contain LoD information."));
  PADDLE_ENFORCE_LE(
      lod_level,
      2UL,
      errors::InvalidArgument("The lod level of input shall be no more than 2."
                              "Received lod level is %d.",
                              lod_level));
  PADDLE_ENFORCE_GE(
      dims[0],
      /*batch size = */ static_cast<int64_t>(lod[lod_level - 1].size() - 1),
      errors::InvalidArgument(
          "The first dimension of Input(X) must be large than batch size."
          "But received first dimension of Input(X) is %d, while batch"
          "size is %d.",
          dims[0],
          static_cast<int64_t>(lod[lod_level - 1].size() - 1)));
  if (lod_level > 1UL) {
    PADDLE_ENFORCE_EQ(
        lod[0][lod[0].size() - 1],
        lod[1].size() - 1,
        errors::InvalidArgument("The input lod information is illegal."));
    phi::LegacyLoD out_lod;
    out_lod.push_back(lod[0]);
    out->set_lod(out_lod);
  }
  dims[0] = lod[lod_level - 1].size() - 1;
  out->Resize({dims});
  ctx.template Alloc<T>(out);
  phi::DenseTensor* index = nullptr;

  // Do not create index buffer for inference mode
  if (pooltype == "MAX" &&
      (is_test == false || (ctx.GetPlace() == phi::CPUPlace()) == false)) {
    index = max_index;
    index->Resize({dims});
    ctx.template Alloc<int32_t>(index);
  }
  phi::funcs::SequencePoolFunctor<Context, T> pool;
  pool(ctx, pooltype, pad_value_, x, out, is_test, index);
}

}  // namespace phi
