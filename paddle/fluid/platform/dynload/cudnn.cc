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

#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace dynload {
std::once_flag cudnn_dso_flag;
void* cudnn_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUDNN_DNN_ROUTINE_EACH(DEFINE_WRAP);
CUDNN_DNN_ROUTINE_EACH_R2(DEFINE_WRAP);

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R3
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_AFTER_R4
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R5
CUDNN_DNN_ROUTINE_EACH_R5(DEFINE_WRAP);
#endif

#ifdef CUDNN_DNN_ROUTINE_EACH_R7
CUDNN_DNN_ROUTINE_EACH_R7(DEFINE_WRAP);
#endif

#ifdef PADDLE_USE_DSO
bool HasCUDNN() {
  std::call_once(cudnn_dso_flag,
                 []() { cudnn_dso_handle = GetCUDNNDsoHandle(); });
  return cudnn_dso_handle != nullptr;
}

void EnforceCUDNNLoaded(const char* fn_name) {
  PADDLE_ENFORCE(cudnn_dso_handle != nullptr,
                 "Cannot load cudnn shared library. Cannot invoke method %s",
                 fn_name);
}
#else
bool HasCUDNN() { return true; }
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
