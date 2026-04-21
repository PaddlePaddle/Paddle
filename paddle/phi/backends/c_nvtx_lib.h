// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#pragma once

#include <cstdint>

namespace phi {
namespace nvtx {
enum class NvtxRangeColor : uint32_t {
  Black = 0x00000000,
  Red = 0x00ff0000,
  Green = 0x0000ff00,
  Blue = 0x000000ff,
  White = 0x00ffffff,
  Yellow = 0x00ffff00,
};
}  // namespace nvtx
}  // namespace phi
