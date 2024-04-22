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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#endif

// new phi apis
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/api/include/fused_api.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/include/tensor_utils.h"

// phi common headers
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"

// original custom op headers
#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/ext/tensor_compat.h"
