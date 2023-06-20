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

#include "paddle/fluid/framework/phi_tensor_base_vector.h"

namespace paddle {
namespace framework {

template <>
struct PhiVectorType<const phi::DenseTensor*> {
  const char* type_name = "PhiTensorRefArray";
};

using TensorRefArray = PhiVector<const phi::DenseTensor*>;

}  // namespace framework
}  // namespace paddle
