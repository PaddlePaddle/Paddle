<<<<<<< HEAD
/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
=======
/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/dynload/rccl.h"

namespace paddle {
namespace platform {
namespace dynload {

<<<<<<< HEAD
std::once_flag rccl_dso_flag;
void *rccl_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

RCCL_RAND_ROUTINE_EACH(DEFINE_WRAP);
=======
std::once_flag nccl_dso_flag;
void *nccl_dso_handle;

#define DEFINE_WRAP(__name) DynLoad__##__name __name

NCCL_RAND_ROUTINE_EACH(DEFINE_WRAP);
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
