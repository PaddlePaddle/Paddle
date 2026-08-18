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
#pragma once

#include <nccl.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

#ifdef __cplusplus
extern "C" {
#endif
ncclResult_t ncclCommInitRank2(ncclComm_t* newcomm,
                               int nranks,
                               ncclUniqueId commId,
                               int myrank,
                               int param);

#if NCCL_VERSION_CODE < 21400
typedef struct ncclConfig_v21400 ncclConfig_t;
#endif

typedef struct ncclMemOptConfig ncclMemOptConfig_t;

ncclResult_t ncclCommInitRankConfigMemOpt(ncclComm_t* comm,
                                          int nranks,
                                          ncclUniqueId commId,
                                          int myrank,
                                          ncclConfig_t* config,
                                          ncclMemOptConfig_t* memopt_config);

ncclMemOptConfig_t* ncclCommGenMemOptConfig(const char* commName,
                                            int ll_buffsize,
                                            int ll128_buffsize,
                                            int simple_buffsize,
                                            int buffsize_align,
                                            int nchannels,
                                            const char* algoStr,
                                            const char* protoStr);

ncclResult_t ncclCommFreeMemOptConfig(ncclMemOptConfig_t* config);
#ifdef __cplusplus
}
#endif

namespace phi {
namespace dynload {

extern std::once_flag nccl_dso_flag;
extern void* nccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    static auto GetNCCLFunc() {                                  \
      using nccl_func = decltype(&::__name);                     \
      std::call_once(nccl_dso_flag, []() {                       \
        nccl_dso_handle = phi::dynload::GetNCCLDsoHandle();      \
      });                                                        \
      static void* p_##__name = dlsym(nccl_dso_handle, #__name); \
      return reinterpret_cast<nccl_func>(p_##__name);            \
    }                                                            \
                                                                 \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return GetNCCLFunc()(args...);                             \
    }                                                            \
                                                                 \
    static bool IsValid() { return GetNCCLFunc() != nullptr; }   \
  };                                                             \
  extern DynLoad__##__name __name

#define NCCL_RAND_ROUTINE_EACH(__macro)  \
  __macro(ncclCommInitAll);              \
  __macro(ncclGetUniqueId);              \
  __macro(ncclCommInitRank);             \
  __macro(ncclCommInitRank2);            \
  __macro(ncclCommInitRankConfigMemOpt); \
  __macro(ncclCommGenMemOptConfig);      \
  __macro(ncclCommFreeMemOptConfig);     \
  __macro(ncclCommAbort);                \
  __macro(ncclCommDestroy);              \
  __macro(ncclCommCount);                \
  __macro(ncclCommCuDevice);             \
  __macro(ncclCommUserRank);             \
  __macro(ncclAllReduce);                \
  __macro(ncclBcast);                    \
  __macro(ncclAllGather);                \
  __macro(ncclGroupStart);               \
  __macro(ncclGroupEnd);                 \
  __macro(ncclReduce);                   \
  __macro(ncclReduceScatter);            \
  __macro(ncclCommGetAsyncError);        \
  __macro(ncclGetErrorString);

NCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)

#if NCCL_VERSION_CODE >= 2212
#define NCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(ncclBroadcast);
NCCL_RAND_ROUTINE_EACH_AFTER_2212(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2304
#define NCCL_RAND_ROUTINE_EACH_AFTER_2304(__macro) __macro(ncclGetVersion);
NCCL_RAND_ROUTINE_EACH_AFTER_2304(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2703
#define NCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(ncclSend);                               \
  __macro(ncclRecv);
NCCL_RAND_ROUTINE_EACH_AFTER_2703(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 21100
#define NCCL_RAND_ROUTINE_EACH_AFTER_21100(__macro) \
  __macro(ncclRedOpCreatePreMulSum);                \
  __macro(ncclRedOpDestroy);
NCCL_RAND_ROUTINE_EACH_AFTER_21100(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 21400
#define NCCL_RAND_ROUTINE_EACH_AFTER_21400(__macro) \
  __macro(ncclCommInitRankConfig);
NCCL_RAND_ROUTINE_EACH_AFTER_21400(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

// User buffer registration, required by the zero-copy and the zero-SM
// (NCCL_CTA_POLICY_ZERO) communication paths.
#if NCCL_VERSION_CODE >= 21900
#define NCCL_RAND_ROUTINE_EACH_AFTER_21900(__macro) \
  __macro(ncclMemAlloc);                            \
  __macro(ncclMemFree);                             \
  __macro(ncclCommRegister);                        \
  __macro(ncclCommDeregister);
NCCL_RAND_ROUTINE_EACH_AFTER_21900(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

// Symmetric memory window registration. The hierarchical zero-SM collectives
// (NCCL 2.30.7+) operate on buffers registered through this API.
#if NCCL_VERSION_CODE >= 22700
#define NCCL_RAND_ROUTINE_EACH_AFTER_22700(__macro) \
  __macro(ncclCommWindowRegister);                  \
  __macro(ncclCommWindowDeregister);
NCCL_RAND_ROUTINE_EACH_AFTER_22700(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

// Single-call all-to-all. Unlike a group of ncclSend/ncclRecv this is a
// symmetric collective, which is what makes the zero-SM path reachable for
// all-to-all. Present in 2.29.7; the exact version it appeared in is unknown,
// so the gate matches the one used for ncclConfig_t::CTAPolicy.
#if NCCL_VERSION_CODE >= 22900
#define NCCL_RAND_ROUTINE_EACH_AFTER_22900(__macro) __macro(ncclAlltoAll);
NCCL_RAND_ROUTINE_EACH_AFTER_22900(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
#endif

}  // namespace dynload
}  // namespace phi
