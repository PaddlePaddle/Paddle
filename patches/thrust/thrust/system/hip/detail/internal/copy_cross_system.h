#pragma once

// Shadow wrapper around rocThrust's `thrust/system/hip/detail/internal/copy_cross_system.h`.
// The upstream implementation is correct; the fragile part is `NV_IF_TARGET` under HIP-clang.
// See `patches/hip/fix_nv_if_target.h`.

#include "hip/fix_nv_if_target.h"

#include <thrust/detail/config.h>

#include_next <thrust/system/hip/detail/internal/copy_cross_system.h>

