/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

// #include <mtrtc.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"
#include "paddle/phi/core/enforce.h"

// TODO(MTAI): The following musa runtime compiling functions are not supported
// now. Here empty implementations are given temporarily. When compiler MCC
// supports these functions, we will replace them.
typedef struct _mtrtcProgram *mtrtcProgram;

typedef enum {
  MTRTC_SUCCESS = 0,
  MTRTC_ERROR_OUT_OF_MEMORY = 1,
  MTRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  MTRTC_ERROR_INVALID_INPUT = 3,
  MTRTC_ERROR_INVALID_PROGRAM = 4,
  MTRTC_ERROR_INVALID_OPTION = 5,
  MTRTC_ERROR_COMPILATION = 6,
  MTRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  MTRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  MTRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  MTRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  MTRTC_ERROR_INTERNAL_ERROR = 11
} mtrtcResult;

inline mtrtcResult mtrtcVersion(int *major, int *minor) {
  PADDLE_THROW(
      phi::errors::Unimplemented("mtrtcVersion is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline const char *mtrtcGetErrorString(mtrtcResult result) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcGetErrorString is not supported on MUSA now!"));
  return "mtrtcGetErrorString is not supported on MUSA now!";
}

inline mtrtcResult mtrtcCompileProgram(mtrtcProgram prog,
                                       int numOptions,
                                       const char *const *options) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcCompileProgram is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcCreateProgram(mtrtcProgram *prog,
                                      const char *src,
                                      const char *name,
                                      int numHeaders,
                                      const char *const *headers,
                                      const char *const *includeNames) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcCreateProgram is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcDestroyProgram(mtrtcProgram *prog) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcDestroyProgram is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcGetMUSA(mtrtcProgram prog, char *musa) {
  PADDLE_THROW(
      phi::errors::Unimplemented("mtrtcGetMUSA is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcGetMUSASize(mtrtcProgram prog, size_t *musaSizeRet) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcGetMUSASize is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcGetProgramLog(mtrtcProgram prog, char *log) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcGetProgramLog is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

inline mtrtcResult mtrtcGetProgramLogSize(mtrtcProgram prog,
                                          size_t *logSizeRet) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "mtrtcGetProgramLogSize is not supported on MUSA now!"));
  return mtrtcResult::MTRTC_ERROR_INTERNAL_ERROR;
}

namespace phi {
namespace dynload {

extern std::once_flag musartc_dso_flag;
extern void *musartc_dso_handle;
extern bool HasNVRTC();

#define DECLARE_DYNAMIC_LOAD_NVRTC_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using musartc_func = decltype(&::__name);                      \
      std::call_once(musartc_dso_flag, []() {                        \
        musartc_dso_handle = phi::dynload::GetNVRTCDsoHandle();      \
      });                                                            \
      static void *p_##__name = dlsym(musartc_dso_handle, #__name);  \
      return reinterpret_cast<musartc_func>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

/**
 * include all needed musartc functions
 **/
#define MUSARTC_ROUTINE_EACH(__macro) \
  __macro(mtrtcVersion);              \
  __macro(mtrtcGetErrorString);       \
  __macro(mtrtcCompileProgram);       \
  __macro(mtrtcCreateProgram);        \
  __macro(mtrtcDestroyProgram);       \
  __macro(mtrtcGetMUSA);              \
  __macro(mtrtcGetMUSASize);          \
  __macro(mtrtcGetProgramLog);        \
  __macro(mtrtcGetProgramLogSize)

MUSARTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NVRTC_WRAP);

#undef DECLARE_DYNAMIC_LOAD_NVRTC_WRAP

}  // namespace dynload
}  // namespace phi
