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
#include <vector>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "xfa/flash_api.h"

namespace xfa = baidu::xpu::xfa;
namespace phi {

using XPUTypeFP16 = typename XPUTypeTrait<phi::float16>::Type;
using XPUTypeBF16 = typename XPUTypeTrait<phi::bfloat16>::Type;

enum XPU_FA_DTYPE {
  FA_FLOAT = 0,
  FA_TFLOAT32,
  FA_FLOAT16,
  FA_BFLOAT16,
  FA_INT32,
};

inline bool get_env_flag(const char* env_name) {
  const char* env = std::getenv(env_name);
  return env != nullptr && std::strcmp(env, "1") == 0;
}

template <typename T>  // for bfloat16 and float16
XPU_FA_DTYPE get_flash_attn_tgemm() {
  if (get_env_flag("XPU_PADDLE_FA_TGEMM_FLOAT16")) {
    return XPU_FA_DTYPE::FA_FLOAT16;
  } else if (get_env_flag("XPU_PADDLE_FA_TGEMM_FLOAT")) {
    return XPU_FA_DTYPE::FA_FLOAT;
  } else {
    return XPU_FA_DTYPE::FA_TFLOAT32;
  }
}

template <>  // for float
inline XPU_FA_DTYPE get_flash_attn_tgemm<float>() {
  if (get_env_flag("XPU_PADDLE_FA_TGEMM_FLOAT")) {
    return XPU_FA_DTYPE::FA_FLOAT;
  } else {
    return XPU_FA_DTYPE::FA_TFLOAT32;
  }
}

template <typename T>  // for bfloat16 and float16
XPU_FA_DTYPE get_flash_attn_taccum() {
  if (get_env_flag("XPU_PADDLE_FA_TACCUM_FLOAT16")) {
    return XPU_FA_DTYPE::FA_FLOAT16;
  } else {
    return XPU_FA_DTYPE::FA_FLOAT;
  }
}

template <>  // for float
inline XPU_FA_DTYPE get_flash_attn_taccum<float>() {
  return XPU_FA_DTYPE::FA_FLOAT;
}

static void GenerateRNGState(
    const XPUContext& dev_ctx,
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
      auto* gen = dev_ctx.GetGenerator();
      seed_offset_pair = gen->IncrementOffset(inc);
    }
    seed_offset_data[0] = static_cast<int64_t>(seed_offset_pair.first);
    seed_offset_data[1] = static_cast<int64_t>(seed_offset_pair.second);
  }
}
}  // namespace phi
