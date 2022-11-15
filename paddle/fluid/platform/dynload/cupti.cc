/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_CUPTI

#include "paddle/fluid/platform/dynload/cupti.h"

namespace paddle {
namespace platform {
namespace dynload {

#define DEFINE_WRAP(__name) DynLoad__##__name __name

CUPTI_ROUTINE_EACH(DEFINE_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_WITH_CUPTI
