// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/utils/test_macros.h"

#define IR_API TEST_API
#if defined(_WIN32)
#ifndef STATIC_IR
#ifdef IR_LIBRARY
#define IR_API __declspec(dllexport)
#else
#define IR_API __declspec(dllimport)
#endif  // IR_LIBRARY
#endif  // STATIC_IR
#endif  // _WIN32
