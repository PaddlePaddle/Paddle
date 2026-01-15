// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "ck/ck.hpp"

namespace ck {

struct BatchedMatrixCoord {
  int batch;
  int row;
  int column;
  bool is_valid;

  __host__ __device__
  BatchedMatrixCoord() : batch(0), row(0), column(0), is_valid(false) {}

  __host__ __device__
  BatchedMatrixCoord(int b, int r, int c) : batch(b), row(r), column(c), is_valid(true) {}

  __host__ __device__
  BatchedMatrixCoord(int b, int r, int c, bool valid) : batch(b), row(r), column(c), is_valid(valid) {}
};

};  // namespace ck
