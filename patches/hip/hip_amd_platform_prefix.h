#pragma once

// Included **only** via `cmake/hip.cmake` `-include...` for HIP compile lines.
// Do not `#include` from normal headers (e.g. Thrust shims may pull
// `fix_nv_if_target.h` in host C++).
//
// HIP 7+ public headers require exactly one of __HIP_PLATFORM_AMD__ or
// __HIP_PLATFORM_NVIDIA__. Some HIP-Clang/CMake paths omit or mis-set these.
#undef __HIP_PLATFORM_AMD__
#undef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_AMD__ 1
#undef __HIP_PLATFORM_HCC__
#undef __HIP_PLATFORM_NVCC__
#define __HIP_PLATFORM_HCC__ 1
