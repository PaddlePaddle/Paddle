/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace sparse {

void Conv3dInferMeta(const MetaTensor& x,
                     const MetaTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     MetaTensor* out,
                     MetaTensor* rulebook,
                     MetaTensor* counter);

void Pool3dInferMeta(const MetaTensor& x,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     MetaTensor* out,
                     MetaTensor* rulebook,
                     MetaTensor* counter);

void SparseCooTensorInferMeta(const MetaTensor& values,
                              const MetaTensor& indices,
                              const IntArray& dense_shape,
                              MetaTensor* out);

}  // namespace sparse
}  // namespace phi
