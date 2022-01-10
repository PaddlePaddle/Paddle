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

// TODO(yuanrisheng): this file may need to be removed
#pragma once

namespace pten {

// the key is kernel_name in fluid, the value is the kernel_name in pten
// the key is sorted by key's alphabet
const std::unordered_map<std::string, std::string> kernel_alias_name_map = {
    {"elementwise_add", "add"},
    {"elementwise_div", "divide"},
    {"elementwise_mul", "muliply"},
    {"elementwise_sub", "subtract"},
    {"fill_any_like", "full_like"},
    {"fill_constant", "full"},
    {"flatten_contiguous_range", "flatten"},
    {"flatten_contiguous_range_grad", "flatten_grad"},
    {"matmul_v2", "matmul"},
    {"reduce_mean", "mean"},
    {"reduce_sum", "sum"},
    {"reshape2", "reshape"},
    // fluid kernel "mean/reshape/matmul/flatten/sum" should be deprecated
    {"flatten", "deprecated"},
    {"flatten_grad", "deprecated"},
    {"matmul", "deprecated"},
    {"mean", "deprecated"},
    {"reshape", "deprecated"},
    {"sum", "deprecated"}};

}  // namespace pten
