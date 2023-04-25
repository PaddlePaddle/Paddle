/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

// All paddle apis in C++ frontend
#include "paddle/phi/api/all.h"
#if !defined(PADDLE_ON_INFERENCE) && !defined(PADDLE_NO_PYTHON)
// Python bindings for the C++ frontend (includes Python.h)
#include "paddle/utils/pybind.h"
#endif
// For initialization of DeviceContextPool and MemoryMethod
#include "paddle/fluid/platform/init_phi.h"

static paddle::InitPhi g_init_phi;
