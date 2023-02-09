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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/operants_base.h"

namespace paddle {

namespace operants {

class StaticTensorOperants : public TensorOperantsBase {
 public:
  StaticTensorOperants() = default;

  Tensor multiply(const Tensor& x, const Tensor& y) override;

 private:
  DISABLE_COPY_AND_ASSIGN(StaticTensorOperants);
};

}  // namespace operants
}  // namespace paddle
