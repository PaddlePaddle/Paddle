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

#include "paddle/top/api/include/tensor.h"

namespace egr {

void ScaleAPI(const pt::Tensor& x, float scale, float bias, bool bias_after_scale, std::vector<pt::Tensor>& outs);
void FillConstAPI(double value, const pt::DDim& ddim, const pt::Backend& backend, 
                  const pt::DataType& dtype, const pt::DataLayout& layout,
                  pt::Tensor& target);
void AccumulateTensorsAPI(pt::Tensor& t0, const pt::Tensor& t1);

} // namespace egr
