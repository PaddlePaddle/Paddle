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

#include <cassert>

#include "paddle/extension.h"

int main() {
  float data[] = {1., 2., 3., 4.};
  auto tensor = paddle::from_blob(data, {2, 2}, phi::DataType::FLOAT32);
  auto gpu_tensor =
      paddle::experimental::copy_to(tensor, phi::GPUPlace(), false);
  assert(gpu_tensor.is_gpu());
}
