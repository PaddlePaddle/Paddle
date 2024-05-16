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
#include <string>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/detection/bbox_util.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void BoxClipKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& im_info,
                   DenseTensor* output) {
  auto* input_box = &input;
  auto* im_info_p = &im_info;
  auto* output_box = output;
  dev_ctx.template Alloc<T>(output_box);

  if (input_box->lod().size()) {
    PADDLE_ENFORCE_EQ(input_box->lod().size(),
                      1UL,
                      phi::errors::InvalidArgument(
                          "Input(Input) of BoxClip only supports 1 level "
                          "of LoD. But received the "
                          "level = %d",
                          input_box->lod().size()));
  }
  auto box_lod = input_box->lod().back();
  int64_t n = static_cast<int64_t>(box_lod.size() - 1);
  for (int i = 0; i < n; ++i) {
    phi::DenseTensor im_info_slice = im_info_p->Slice(i, i + 1);
    phi::DenseTensor box_slice = input_box->Slice(box_lod[i], box_lod[i + 1]);
    phi::DenseTensor output_slice =
        output_box->Slice(box_lod[i], box_lod[i + 1]);
    phi::funcs::ClipTiledBoxes<T>(
        dev_ctx, im_info_slice, box_slice, &output_slice);
  }
}
}  // namespace phi
