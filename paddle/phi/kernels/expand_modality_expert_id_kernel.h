// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void ExpandModalityExpertIDKernel(const Context& dev_ctx,
                                  const DenseTensor& expert_id,
                                  int64_t num_expert_per_modality,
                                  int64_t group_size,
                                  int64_t modality_offset,
                                  bool is_group_expert,
                                  DenseTensor* expert_id_out);

}  // namespace phi
