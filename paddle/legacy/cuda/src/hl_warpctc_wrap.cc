/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_warpctc_wrap.h"
#include <mutex>
#include "paddle/utils/DynamicLoader.h"
#include "paddle/utils/Logging.h"

namespace dynload {

std::once_flag warpctc_dso_flag;
void* warpctc_dso_handle = nullptr;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load warpctc routine
 * via operator overloading. When PADDLE_USE_DSO is
 * false, you need to add the path of libwarp-ctc.so to
 * the linked-libs of paddle or to LD_PRELOAD.
 */
#define DYNAMIC_LOAD_WARPCTC_WRAP(__name)                              \
  struct DynLoad__##__name {                                           \
    template <typename... Args>                                        \
    auto operator()(Args... args) -> decltype(__name(args...)) {       \
      using warpctcFunc = decltype(__name(args...)) (*)(Args...);      \
      std::call_once(                                                  \
          warpctc_dso_flag, GetWarpCTCDsoHandle, &warpctc_dso_handle); \
      void* p_##_name = dlsym(warpctc_dso_handle, #__name);            \
      return reinterpret_cast<warpctcFunc>(p_##_name)(args...);        \
    }                                                                  \
  } __name;  // struct DynLoad__##__name

// include all needed warp-ctc functions
DYNAMIC_LOAD_WARPCTC_WRAP(get_warpctc_version)
DYNAMIC_LOAD_WARPCTC_WRAP(ctcGetStatusString)
DYNAMIC_LOAD_WARPCTC_WRAP(compute_ctc_loss)
DYNAMIC_LOAD_WARPCTC_WRAP(get_workspace_size)

#undef DYNAMIC_LOAD_WARPCTC_WRAP

} /* namespace dynload */

#define WARPCTC_GET_VERSION dynload::get_warpctc_version
#define WARPCTC_GET_STATUS_STRING dynload::ctcGetStatusString

static int g_warpctcVersion = -1;
#ifndef PADDLE_TYPE_DOUBLE
#define WARPCTC_COMPUTE_LOSS dynload::compute_ctc_loss
#define WARPCTC_GET_WORKSPACE_SIZE dynload::get_workspace_size
#else
hl_warpctc_status_t fatal(...) {
  LOG(FATAL) << "warp-ctc [version " << g_warpctcVersion
             << "] Error: not support double precision.";
  // both of get_warpctc_version() and get_workspace_size() return an ctcStatus
  // type value
  return CTC_STATUS_EXECUTION_FAILED;
}
#define WARPCTC_COMPUTE_LOSS fatal
#define WARPCTC_GET_WORKSPACE_SIZE fatal
#endif

/**
 * Check build-in warp-ctc function using glog and it also
 * support << operator for more details error info.
 */
#define CHECK_WARPCTC(warpctcStat)                \
  CHECK_EQ(CTC_STATUS_SUCCESS, warpctcStat)       \
      << "warp-ctc [version " << g_warpctcVersion \
      << "] Error: " << WARPCTC_GET_STATUS_STRING(warpctcStat) << " "

void hl_warpctc_init(const size_t blank,
                     bool useGpu,
                     hl_warpctc_options_t* options) {
  CHECK_NOTNULL(options);

  g_warpctcVersion = WARPCTC_GET_VERSION();

  if (useGpu) {
#ifdef __NVCC__
    options->loc = CTC_GPU;
    options->stream = STREAM_DEFAULT;
#else
    LOG(FATAL) << "[warpctc init] GPU is not enabled.";
#endif
  } else {
    options->loc = CTC_CPU;
    options->num_threads = 1;
  }

  options->blank_label = blank;
}

void hl_warpctc_compute_loss(const real* batchInput,
                             real* batchGrad,
                             const int* cpuLabels,
                             const int* cpuLabelLengths,
                             const int* cpuInputLengths,
                             const size_t numClasses,
                             const size_t numSequences,
                             real* cpuCosts,
                             void* workspace,
                             hl_warpctc_options_t* options) {
  CHECK_NOTNULL(batchInput);
  CHECK_NOTNULL(cpuLabels);
  CHECK_NOTNULL(cpuLabelLengths);
  CHECK_NOTNULL(cpuInputLengths);
  CHECK_NOTNULL(cpuCosts);
  CHECK_NOTNULL(workspace);
  CHECK_NOTNULL(options);

  CHECK_WARPCTC(WARPCTC_COMPUTE_LOSS(batchInput,
                                     batchGrad,
                                     cpuLabels,
                                     cpuLabelLengths,
                                     cpuInputLengths,
                                     numClasses,
                                     numSequences,
                                     cpuCosts,
                                     workspace,
                                     *options));
}

void hl_warpctc_get_workspace_size(const int* cpuLabelLengths,
                                   const int* cpuInputLengths,
                                   const size_t numClasses,
                                   const size_t numSequences,
                                   hl_warpctc_options_t* options,
                                   size_t* bytes) {
  CHECK_NOTNULL(cpuLabelLengths);
  CHECK_NOTNULL(cpuInputLengths);
  CHECK_NOTNULL(options);
  CHECK_NOTNULL(bytes);

  CHECK_WARPCTC(WARPCTC_GET_WORKSPACE_SIZE(cpuLabelLengths,
                                           cpuInputLengths,
                                           numClasses,
                                           numSequences,
                                           *options,
                                           bytes));
}
