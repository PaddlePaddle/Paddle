// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/chunk_eval_kernel.h"
#include "paddle/phi/kernels/impl/chunk_eval_kernel_impl.h"

PD_REGISTER_KERNEL(
    chunk_eval, CPU, ALL_LAYOUT, phi::ChunkEvalKernel, float, int64_t) {
  kernel->InputAt(0).SetDataType(phi::DataType::INT64);
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);

  kernel->OutputAt(3).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(4).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(5).SetDataType(phi::DataType::INT64);
}
