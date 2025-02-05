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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void ChunkEvalKernel(const Context& dev_ctx,
                     const DenseTensor& inference,
                     const DenseTensor& label,
                     const paddle::optional<DenseTensor>& seq_length,
                     int num_chunk_types,
                     const std::string& chunk_scheme,
                     const std::vector<int>& excluded_chunk_types,
                     DenseTensor* precision,
                     DenseTensor* recall,
                     DenseTensor* f1_score,
                     DenseTensor* num_infer_chunks,
                     DenseTensor* num_label_chunks,
                     DenseTensor* num_correct_chunks);

}  // namespace phi
