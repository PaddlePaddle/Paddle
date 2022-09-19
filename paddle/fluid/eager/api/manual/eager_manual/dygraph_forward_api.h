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

#include "paddle/phi/api/include/tensor.h"

paddle::experimental::Tensor add_n_ad_func(
    const std::vector<paddle::experimental::Tensor>& x);

paddle::experimental::Tensor conv2d_ad_func(
    const paddle::experimental::Tensor& input,
    const paddle::experimental::Tensor& filter,
    std::vector<int> strides,
    std::vector<int> paddings,
    std::string paddding_algorithm,
    int groups,
    std::vector<int> dilations,
    std::string data_format,
    bool use_addto,
    int workspace_size_MB,
    bool exhaustive_search);
