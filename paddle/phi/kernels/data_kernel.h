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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename Context>
void DataKernel(const Context& ctx,
                const std::string& name,
                const phi::IntArray& shape,
                phi::DataType data_type,
                DenseTensor* out);

template <typename Context>
void ShadowOutputKernel(const Context& ctx,
                        const DenseTensor& x,
                        DenseTensor* out);

template <typename Context>
void ShadowFeedKernel(const Context& ctx,
                      const DenseTensor& x,
                      DenseTensor* out);

template <typename Context>
void ShadowFeedTensorsKernel(const Context& ctx,
                             const std::vector<const DenseTensor*>& xs,
                             std::vector<DenseTensor*> outs);

template <typename Context>
void PrintKernel(const Context& ctx,
                 const DenseTensor& x,
                 int first_n,
                 const std::string& message,
                 int summarize,
                 bool print_tensor_name,
                 bool print_tensor_type,
                 bool print_tensor_shape,
                 bool print_tensor_layout,
                 bool print_tensor_lod,
                 const std::string& print_phase,
                 bool is_forward,
                 DenseTensor* out);

}  // namespace phi
