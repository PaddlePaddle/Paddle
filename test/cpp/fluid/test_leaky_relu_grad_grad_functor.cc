// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "test/cpp/fluid/test_leaky_relu_grad_grad_functor.h"

namespace paddle {
namespace operators {

TEST(leaky_relu_grad_grad, test_cpu) {
  ASSERT_TRUE(
      TestLeakyReluGradGradMain<float>({32, 64}, phi::CPUPlace(), 0.02));
}

TEST(leaky_relu_grad_grad, test_cpu_zero_alpha) {
  ASSERT_TRUE(TestLeakyReluGradGradMain<float>({32, 64}, phi::CPUPlace(), 0.0));
}

}  // namespace operators
}  // namespace paddle
