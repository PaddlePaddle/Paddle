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

#include "paddle/fluid/jit/function.h"

#include <string>
#include <vector>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/jit/engine/base_engine.h"
#include "paddle/fluid/jit/function_utils.h"

namespace paddle {
namespace jit {

Function::Function(BaseEngine* engine) : engine_(engine) {}

std::vector<Tensor> Function::operator()(
    const std::vector<Tensor>& inputs) const {
  PADDLE_ENFORCE_EQ(IsValid(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Funtion engine ptr is nullptr, please check it."));
  auto dense_tensors = utils::ToDenseTensors(inputs);
  return utils::ToTensors(this->operator()(dense_tensors));
}

std::vector<DenseTensor> Function::operator()(
    const std::vector<DenseTensor>& inputs) const {
  PADDLE_ENFORCE_EQ(IsValid(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Funtion engine ptr is nullptr, please check it."));
  return (*engine_)(inputs);
}

}  // namespace jit
}  // namespace paddle
