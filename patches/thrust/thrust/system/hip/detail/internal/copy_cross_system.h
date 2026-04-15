#pragma once

// Shadow header to fix rocThrust/Thrust host-device dispatch under HIP-clang.
// rocThrust's copy_cross_system.h uses NV_IF_TARGET(...) to guard host-only
// trivial_device_copy calls inside a THRUST_HOST THRUST_DEVICE function.
// If NV_IF_TARGET does not correctly select the device branch at device compile
// time, HIP-clang attempts to compile the host branch for device and fails with:
//   reference to __host__ function 'trivial_device_copy' in __host__ __device__ function
//
// We override NV_IF_TARGET to select branches using __HIP_DEVICE_COMPILE__ and
// unwrap the "(...)" code blocks.

#include <thrust/detail/config.h>

#ifdef NV_IF_TARGET
#  undef NV_IF_TARGET
#endif

#ifdef NV_IS_HOST
#  undef NV_IS_HOST
#endif
#ifdef NV_IS_DEVICE
#  undef NV_IS_DEVICE
#endif
#define NV_IS_HOST PADDLE_NV_IS_HOST
#define NV_IS_DEVICE PADDLE_NV_IS_DEVICE

#define PADDLE_THRUST_REMOVE_PARENS_IMPL(...) __VA_ARGS__
#define PADDLE_THRUST_REMOVE_PARENS(X) PADDLE_THRUST_REMOVE_PARENS_IMPL X

#define PADDLE_THRUST_GET_4TH_ARG(_1, _2, _3, _4, ...) _4
#define PADDLE_THRUST_NV_IF_TARGET_DISPATCH(...)                                              \
  PADDLE_THRUST_GET_4TH_ARG(__VA_ARGS__, PADDLE_THRUST_NV_IF_TARGET_3, PADDLE_THRUST_NV_IF_TARGET_2)

#if defined(__HIP_DEVICE_COMPILE__)
#  define PADDLE_THRUST_NV_IF_TARGET_2_PADDLE_NV_IS_HOST(code) /* stripped in device compile */
#  define PADDLE_THRUST_NV_IF_TARGET_2_PADDLE_NV_IS_DEVICE(code) PADDLE_THRUST_REMOVE_PARENS(code)
#  define PADDLE_THRUST_NV_IF_TARGET_2(cond, code) PADDLE_THRUST_NV_IF_TARGET_2_##cond(code)
#  define PADDLE_THRUST_NV_IF_TARGET_3(cond, host_code, device_code) PADDLE_THRUST_REMOVE_PARENS(device_code)
#else
#  define PADDLE_THRUST_NV_IF_TARGET_2_PADDLE_NV_IS_HOST(code) PADDLE_THRUST_REMOVE_PARENS(code)
#  define PADDLE_THRUST_NV_IF_TARGET_2_PADDLE_NV_IS_DEVICE(code) /* stripped in host compile */
#  define PADDLE_THRUST_NV_IF_TARGET_2(cond, code) PADDLE_THRUST_NV_IF_TARGET_2_##cond(code)
#  define PADDLE_THRUST_NV_IF_TARGET_3(cond, host_code, device_code) PADDLE_THRUST_REMOVE_PARENS(host_code)
#endif

#define NV_IF_TARGET(...) PADDLE_THRUST_NV_IF_TARGET_DISPATCH(__VA_ARGS__)(__VA_ARGS__)

#include_next <thrust/system/hip/detail/internal/copy_cross_system.h>

