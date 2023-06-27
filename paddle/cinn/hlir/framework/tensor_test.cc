// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/tensor.h"

#include <gtest/gtest.h>

namespace cinn {
namespace hlir {
namespace framework {

TEST(Tensor, basic) {
  _Tensor_ tensor;
  tensor.Resize(Shape{{3, 2}});

  auto* data = tensor.mutable_data<float>(common::DefaultHostTarget());

  for (int i = 0; i < tensor.shape().numel(); i++) {
    data[i] = i;
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
