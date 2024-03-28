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

#if defined(__clang__) || defined(__GNUC__)
#define CPP_STANDARD __cplusplus
#elif defined(_MSC_VER)
#define CPP_STANDARD _MSVC_LANG
#endif

#ifndef CUSTOM_OP_WITH_SPMD
#define CUSTOM_OP_WITH_SPMD
#endif

// All paddle apis in C++ frontend
// phi headers
#include "paddle/phi/api/all.h"
// common headers
#include "paddle/common/ddim.h"
#include "paddle/common/exception.h"
#include "paddle/common/layout.h"

#if CPP_STANDARD >= 201703L && !defined(__clang__)
// pir&pass headers
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#endif

#if !defined(PADDLE_ON_INFERENCE) && !defined(PADDLE_NO_PYTHON)
// Python bindings for the C++ frontend (includes Python.h)
#include "paddle/utils/pybind.h"
#endif
// For initialization of DeviceContextPool and MemoryMethod
#include "paddle/fluid/platform/init_phi.h"

static paddle::InitPhi g_init_phi;
