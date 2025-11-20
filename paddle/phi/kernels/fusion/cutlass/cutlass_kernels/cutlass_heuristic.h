/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/ft_gemm_configs.h"

namespace phi {

static std::vector<CutlassTileConfig> get_candidate_tiles(
    const bool is_weight_only,
    const bool simt_configs_only,
    const int sm,
    const int group_size,
    const bool is_moe) {
  VLOG(3) << "get_candidate_tiles sm: " << sm;
  if (simt_configs_only) {
    std::vector<CutlassTileConfig> simt_configs{
        CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};
    return simt_configs;
  } else if (!is_weight_only) {
    std::vector<CutlassTileConfig> square_configs{
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
        CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64,
        CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
    };
    return square_configs;
  }

  std::vector<CutlassTileConfig> quant_B_configs_sm70{
      CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape64x64x64,
  };
  std::vector<CutlassTileConfig> quant_B_configs_sm80{
      CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
      CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
      CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      CutlassTileConfig::CtaShape64x64x64_WarpShape32x32x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
      CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64,
      CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
  };
  if (is_moe) {
    quant_B_configs_sm80.push_back(
        CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64);
  } else {
    quant_B_configs_sm80.push_back(
        CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64);
  }
  std::vector<CutlassTileConfig> quant_B_configs_sm80_finegrained{
      CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
      CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
      CutlassTileConfig::CtaShape64x128x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64,
      CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64,
  };
  std::vector<CutlassTileConfig> quant_B_configs;
  switch (sm) {
    case 90:
    case 86:
    case 80: {
      quant_B_configs = group_size > 0 ? quant_B_configs_sm80_finegrained
                                       : quant_B_configs_sm80;
      break;
    }
    case 75:
    case 70:
      quant_B_configs = quant_B_configs_sm70;
      break;
    default:
      quant_B_configs = quant_B_configs_sm70;
      break;
  }
  return quant_B_configs;
}

static std::vector<CutlassGemmConfig> get_candidate_configs(
    const int sm,
    const int group_size,
    const bool is_weight_only,
    const bool simt_configs_only,
    const bool is_moe) {
  std::vector<CutlassTileConfig> tiles = get_candidate_tiles(
      is_weight_only, simt_configs_only, sm, group_size, is_moe);

  std::vector<CutlassGemmConfig> candidate_configs;
  int min_stages = 2;
  // Note(yuanlehome): max_stages must smaller than 5!
  int max_stages = sm >= 80 ? 4 : 2;
  if (is_moe) {
    max_stages = 5;
  }
  // NOTE: (changwenbin)
  // Support enabling stream_k by setting the environment
  // variable `export CUTLASS_GEMM_STREAM_K=1`.
  SplitKStyle env_split_k = SplitKStyle::NO_SPLIT_K;
  const char* env_stream_k = std::getenv("CUTLASS_GEMM_STREAM_K");
  if (env_stream_k != nullptr && !is_moe) {
    env_split_k = SplitKStyle::SPLIT_K_SERIAL;
  }
  for (const auto& tile_config : tiles) {
    for (int stages = min_stages; stages <= max_stages; ++stages) {
      CutlassGemmConfig config{tile_config, env_split_k, 1, stages};
      candidate_configs.push_back(config);
    }
  }

  return candidate_configs;
}

}  // namespace phi
