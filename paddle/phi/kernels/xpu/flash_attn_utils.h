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

#ifdef PADDLE_WITH_XPU

#include <vector>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "xfa/flash_api.h"

namespace xfa = baidu::xpu::xfa;
namespace phi {

#ifdef PADDLE_WITH_XPU_XRE5
using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
using XPUTypeBF16 = typename XPUTypeTrait<phi::dtype::bfloat16>::Type;

enum XPU_FA_TGEMM {
  FA_FLOAT = 0,
  FA_TFLOAT32,
  FA_FLOAT16,
};

template <typename T>
XPU_FA_TGEMM get_flash_attn_tgemm() {
  const char* xpu_paddle_fa_float16 =
      std::getenv("XPU_PADDLE_FA_TGEMM_FLOAT16");
  if (xpu_paddle_fa_float16 != nullptr &&
      (std::is_same<phi::dtype::float16, T>::value ||
       std::is_same<XPUTypeFP16, T>::value)) {
    return XPU_FA_TGEMM::FA_FLOAT16;
  } else if ((std::is_same<phi::dtype::bfloat16, T>::value ||
              std::is_same<XPUTypeBF16, T>::value) &&
             std::getenv("XPU_PADDLE_FA_BFLOAT16_XTE") != nullptr) {
    return XPU_FA_TGEMM::FA_FLOAT16;
  } else if (std::getenv("XPU_PADDLE_FA_TGEMM_FLOAT") != nullptr) {
    return XPU_FA_TGEMM::FA_FLOAT;
  } else {
    return XPU_FA_TGEMM::FA_TFLOAT32;
  }
}

static void GenerateRNGState(
    const XPUContext& ctx,
    const paddle::optional<DenseTensor>& fixed_seed_offset,
    int64_t* seed_offset_data,
    const std::string& rng_name,
    const int64_t batch_size,
    const int64_t num_heads) {
  if (fixed_seed_offset.get_ptr()) {
    if ((fixed_seed_offset->place()).GetType() == phi::AllocationType::XPU) {
      memory_utils::Copy(phi::CPUPlace(),
                         seed_offset_data,
                         fixed_seed_offset->place(),
                         fixed_seed_offset->data<int64_t>(),
                         sizeof(int64_t) * 2);
    } else {
      const int64_t* fixed_seed_offset_data =
          fixed_seed_offset->data<int64_t>();
      seed_offset_data[0] = fixed_seed_offset_data[0];
      seed_offset_data[1] = fixed_seed_offset_data[1];
    }
  } else {
    std::pair<uint64_t, uint64_t> seed_offset_pair;
    uint64_t inc = batch_size * num_heads * 32;
    if (rng_name != "") {
      auto gen = phi::GetRandomSeedGenerator(rng_name);
      seed_offset_pair = gen->IncrementOffset(inc);
    } else {
      auto* gen = ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed_offset_data[0] = static_cast<int64_t>(seed_offset_pair.first);
    seed_offset_data[1] = static_cast<int64_t>(seed_offset_pair.second);
  }
}

#endif

}  // namespace phi
#endif
