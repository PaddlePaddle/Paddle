#pragma once

// Shadow header to fix rocThrust/Thrust host-device dispatch under HIP-clang.
// ROCm 7's rocThrust uses NV_IF_TARGET(...) in THRUST_HOST_DEVICE functions.
// Under some HIP-clang modes this macro path does not reliably strip host-only
// code from device compilation, triggering errors like:
// - reference to __host__ function hipGetLastError / system_error / hip_category
// - cannot use 'throw' in __host__ __device__ function
//
// We override NV_IF_TARGET to select host/device branches using
// __HIP_DEVICE_COMPILE__ and to correctly unwrap the "(...)" code blocks.

#include <thrust/detail/config.h>

#ifdef NV_IF_TARGET
#  undef NV_IF_TARGET
#endif

// rocThrust defines NV_IS_HOST / NV_IS_DEVICE as numeric macros in some builds.
// That breaks token-pasting based dispatch. Redefine them to stable tokens for
// the duration of the real header include.
#ifdef NV_IS_HOST
#  undef NV_IS_HOST
#endif
#ifdef NV_IS_DEVICE
#  undef NV_IS_DEVICE
#endif
#define NV_IS_HOST PADDLE_NV_IS_HOST
#define NV_IS_DEVICE PADDLE_NV_IS_DEVICE

// Unwrap "( ... )" to "...".
#define PADDLE_THRUST_REMOVE_PARENS_IMPL(...) __VA_ARGS__
#define PADDLE_THRUST_REMOVE_PARENS(X) PADDLE_THRUST_REMOVE_PARENS_IMPL X

// Dispatch to the 2-arg or 3-arg form of NV_IF_TARGET.
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

// Pull in the real rocThrust header after overriding NV_IF_TARGET.
#include_next <thrust/system/hip/detail/util.h>

