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

#include "paddle/cinn/adt/adt.h"

namespace cinn::adt {

struct ArgDimPosDescriptor {
  ArgDimPosDescriptor(std::size_t t_idx, std::size_t d_idx)
      : tensor_idx(t_idx), dim_idx(d_idx) {}

  std::size_t tensor_idx;
  std::size_t dim_idx;
};

DEFINE_ADT_UNION(OpArgPos, Undefined, tIn<std::size_t>, tOut<std::size_t>);
DEFINE_ADT_UNION(OpArgDimPos,
                 Undefined,
                 tIn<ArgDimPosDescriptor>,
                 tOut<ArgDimPosDescriptor>);

}  // namespace cinn::adt
