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

#if !defined(_MSC_VER) && __cplusplus < 199711L
#error C++11 or later compatible compiler is required to use Paddle.
#endif

#include "paddle/fluid/extension/include/dispatch.h"
#include "paddle/fluid/extension/include/dtype.h"
#include "paddle/fluid/extension/include/op_meta_info.h"
#include "paddle/fluid/extension/include/place.h"
#include "paddle/fluid/extension/include/tensor.h"
