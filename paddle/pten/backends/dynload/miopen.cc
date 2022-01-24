/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/backends/dynload/miopen.h"
#include "paddle/fluid/platform/enforce.h"

namespace pten {
namespace dynload {
std::once_flag miopen_dso_flag;
void* miopen_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MIOPEN_DNN_ROUTINE_EACH(DEFINE_WRAP);
MIOPEN_DNN_ROUTINE_EACH_R2(DEFINE_WRAP);

#ifdef MIOPEN_DNN_ROUTINE_EACH_AFTER_R3
MIOPEN_DNN_ROUTINE_EACH_AFTER_R3(DEFINE_WRAP);
#endif

#ifdef MIOPEN_DNN_ROUTINE_EACH_AFTER_R4
MIOPEN_DNN_ROUTINE_EACH_AFTER_R4(DEFINE_WRAP);
#endif

#ifdef MIOPEN_DNN_ROUTINE_EACH_R5
MIOPEN_DNN_ROUTINE_EACH_R5(DEFINE_WRAP);
#endif

#ifdef MIOPEN_DNN_ROUTINE_EACH_R6
MIOPEN_DNN_ROUTINE_EACH_R6(DEFINE_WRAP);
#endif

#ifdef MIOPEN_DNN_ROUTINE_EACH_R7
MIOPEN_DNN_ROUTINE_EACH_R7(DEFINE_WRAP);
#endif

#ifdef MIOPEN_DNN_ROUTINE_EACH_AFTER_R7
MIOPEN_DNN_ROUTINE_EACH_AFTER_R7(DEFINE_WRAP);
#endif

bool HasCUDNN() {
  std::call_once(miopen_dso_flag,
                 []() { miopen_dso_handle = GetCUDNNDsoHandle(); });
  return miopen_dso_handle != nullptr;
}

void EnforceCUDNNLoaded(const char* fn_name) {
  PADDLE_ENFORCE_NOT_NULL(
      miopen_dso_handle,
      paddle::platform::errors::PreconditionNotMet(
          "Cannot load miopen shared library. Cannot invoke method %s.",
          fn_name));
}

}  // namespace dynload
}  // namespace pten
