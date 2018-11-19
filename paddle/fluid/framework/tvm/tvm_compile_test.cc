// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/tvm/tvm_compile.h"
#include <gtest/gtest.h>
#include <iostream>

namespace paddle {
namespace framework {

TEST(tvm, compile) {
  int batch_size = 64;
  int feature_size = 8;
  int N = 128;
  int K = 17;
  auto A = tvm::compile::placeholder({batch_size, feature_size, 1, K},
                                     tvm::compile::Float(32), "A");
  auto B = tvm::compile::placeholder({batch_size, feature_size, K, N},
                                     tvm::compile::Float(32), "B");
  auto k = tvm::compile::reduce_axis(tvm::compile::Range(0, K), "k");

  auto C = tvm::compile::compute(
      {batch_size, feature_size, 1, N},
      [&](tvm::compile::Var x1, tvm::compile::Var x2, tvm::compile::Var x3,
          tvm::compile::Var x4) {
        return tvm::compile::sum(A(x1, x2, x3, k) * B(x1, x2, k, x4), {k});
      },
      "C");

  auto schedule = tvm::compile::create_schedule({C->op});

  std::cout << (&C) << " " << (&schedule) << std::endl;
}

}  // namespace framework
}  // namespace paddle
