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

#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace platform {
namespace dynload {

std::once_flag mklml_dso_flag;
void* mklml_dso_handle = nullptr;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

MKLML_ROUTINE_EACH(DEFINE_WRAP);

#if !defined(_WIN32)
DEFINE_WRAP(mkl_scsrmm);
DEFINE_WRAP(mkl_dcsrmm);
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
