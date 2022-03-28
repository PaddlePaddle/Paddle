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

#pragma once

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/scalar_array.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {
namespace strings {
// Common InferMeta Functions of StringTensor for unary operators:
void UnchangedInferMeta(const StringTensorMeta& x_meta, MetaTensor* out);

void CreateLikeInferMeta(const MetaTensor& x, MetaTensor* out);

}  // namespace strings
}  // namespace phi
