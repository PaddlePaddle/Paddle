// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdint.h>
#include <stdio.h>

#include "pd_common.h"  // NOLINT

typedef struct PD_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} PD_OneDimArrayInt32;  // std::vector<int32_t>

typedef struct PD_OneDimArraySize {
  size_t size;
  size_t* data;
} PD_OneDimArraySize;  // std::vector<size_t>

typedef struct PD_OneDimArrayCstr {
  size_t size;
  char** data;
} PD_OneDimArrayCstr;  // std::vector<std::string>

typedef struct PD_Cstr {
  size_t size;
  char* data;
} PD_Cstr;  // std::string

typedef struct PD_TwoDimArraySize {
  size_t size;
  PD_OneDimArraySize** data;
} PD_TwoDimArraySize;  // std::vector<std::vector<size_t>>
