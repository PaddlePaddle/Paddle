// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

template <typename Context>
void GetAccumulators(const Context& dev_ctx,
                     const DenseTensor& in_num_accumulates,
                     const DenseTensor& in_old_num_accumulates,
                     const DenseTensor& in_num_updates,
                     int64_t* num_updates,
                     int64_t* num_accumulates,
                     int64_t* old_num_accumulates);

template <typename Context>
void SetAccumulators(const Context& dev_ctx,
                     int64_t num_updates,
                     int64_t num_accumulates,
                     int64_t old_num_accumulates,
                     DenseTensor* out_num_accumulates,
                     DenseTensor* out_old_num_accumulates,
                     DenseTensor* out_num_updates);

template <typename T, typename Context>
void AverageAccumulatesKernel(const Context& dev_ctx,
                              const DenseTensor& param,
                              const DenseTensor& in_sum_1,
                              const DenseTensor& in_sum_2,
                              const DenseTensor& in_sum_3,
                              const DenseTensor& in_num_accumulates,
                              const DenseTensor& in_old_num_accumulates,
                              const DenseTensor& in_num_updates,
                              float average_window,
                              int64_t max_average_window,
                              int64_t min_average_window,
                              DenseTensor* out_sum_1,
                              DenseTensor* out_sum_2,
                              DenseTensor* out_sum_3,
                              DenseTensor* out_num_accumulates,
                              DenseTensor* out_old_num_accumulates,
                              DenseTensor* out_num_updates);
}  // namespace phi
