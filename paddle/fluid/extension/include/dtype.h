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
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {

enum DataType {
  FLOAT32,
  FLOAT64,
  BFLOAT16,
  COMPLEX128,
  COMPLEX64,
  FLOAT16,
  INT64,
  INT32,
  INT16,
  UINT8,
  INT8,
  BOOL,
  // TODO(JiabinYang) support more data types if needed.
};

}  // namespace paddle
