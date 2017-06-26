/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/frameowork/place.h"

namespace paddle {
namespace memory {

void* Alloc(paddle::framework::Place, size_t);
void Free(paddle::framework::Place, void*);
size_t Used(paddle::framework::Place);

}  // namespace memory
}  // namespace paddle
