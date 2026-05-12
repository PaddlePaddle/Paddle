#pragma once

// Shadow wrapper around rocThrust's `thrust/system/hip/detail/util.h`.
// See `patches/hip/fix_nv_if_target.h` for the HIP-friendly `NV_IF_TARGET` override.

#include "hip/fix_nv_if_target.h"

#include <thrust/detail/config.h>

// Pull in the real rocThrust header after overriding NV_IF_TARGET.
#include_next <thrust/system/hip/detail/util.h>

