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

#if !defined(_MSC_VER) && __cplusplus < 201402L
#error C++14 or later compatible compiler is required to use Paddle.
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#endif

// new pten apis
#include "paddle/pten/api/include/api.h"
#include "paddle/pten/api/include/manual_api.h"
#include "paddle/pten/api/include/sparse_api.h"
#include "paddle/pten/api/include/tensor.h"

// pten common headers
#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"

// original custom op headers
#include "paddle/pten/api/ext/dispatch.h"
#include "paddle/pten/api/ext/dll_decl.h"
#include "paddle/pten/api/ext/exception.h"
#include "paddle/pten/api/ext/op_kernel_info.h"
#include "paddle/pten/api/ext/op_meta_info.h"
#include "paddle/pten/api/ext/place.h"
#include "paddle/pten/api/ext/tensor_compat.h"
