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

namespace paddle {
namespace experimental {

enum class DataType {
  // pod types.
  INVALID = 0,
  INT8,   // Char
  UINT8,  // BYte
  INT16,
  UINT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  // COMPLEX64,
  // COMPLEX128,
  // composite types.
  // STRING = 100,
};

}  // namespace experimental
}  // namespace paddle
