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

#include "paddle/phi/backends/dynload/nccl.h"
#include "glog/logging.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {

std::once_flag nccl_dso_flag;
void *nccl_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

NCCL_RAND_ROUTINE_EACH(DEFINE_WRAP);

#if NCCL_VERSION_CODE >= 2212
NCCL_RAND_ROUTINE_EACH_AFTER_2212(DEFINE_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2304
NCCL_RAND_ROUTINE_EACH_AFTER_2304(DEFINE_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2703
NCCL_RAND_ROUTINE_EACH_AFTER_2703(DEFINE_WRAP)
#endif

#if NCCL_VERSION_CODE >= 21100
NCCL_RAND_ROUTINE_EACH_AFTER_21100(DEFINE_WRAP)
#endif

int ncclSetMaxUserChannels(int max_channels) {
  using FuncPtr = ncclResult_t (*)(int, int *);
  static FuncPtr func_ptr = [] {
    std::call_once(nccl_dso_flag, []() {
      nccl_dso_handle = phi::dynload::GetNCCLDsoHandle();
    });
    const char *kFuncName = "ncclUpdateMaxUserChannels";
    auto ptr = reinterpret_cast<FuncPtr>(dlsym(nccl_dso_handle, kFuncName));
    if (ptr != nullptr) {
      VLOG(10) << "Get " << kFuncName << " successfully at " << ptr;
    } else {
      VLOG(10) << "Get " << kFuncName << " failed because: " << dlerror();
    }
    return ptr;
  }();

  if (func_ptr) {
    int old_max_channels;
    PADDLE_ENFORCE_GPU_SUCCESS(func_ptr(max_channels, &old_max_channels));
    return old_max_channels;
  } else {
    return -1;
  }
}

}  // namespace dynload
}  // namespace phi
