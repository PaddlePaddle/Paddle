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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/elementwise/test_elementwise_op_grad_grad.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

USE_OP(elementwise_add);

namespace paddle {
namespace operators {

template <typename T>
class TestElementwiseAddGradGradWithoutDout
    : public TestElementwiseOpGradGrad<T> {
 public:
  TestElementwiseAddGradGradWithoutDout(const platform::Place &place,
                                        const framework::DDim &dims)
      : TestElementwiseOpGradGrad<T>("elementwise_add_grad_grad", place, dims,
                                     {"Y", "DOut", "DDY"}, {"DDOut"}) {}

  using TestElementwiseOpGradGrad<T>::feed_datas_;
  using TestElementwiseOpGradGrad<T>::expected_outs_;
  using TestElementwiseOpGradGrad<T>::dims_;
  void ComputeExpectedOuts() override {
    size_t numel = static_cast<size_t>(framework::product(dims_));
    std::vector<T> dy(numel);
    std::vector<T> ddout(numel);
    for (size_t i = 0; i < numel; ++i) {
      // ddOut = ddX + ddY = ddY if ddX empty
      ddout[i] = feed_datas_["DDY"][i];
    }
    expected_outs_["DDOut"] = ddout;
  }

  std::unique_ptr<framework::OperatorBase> CreateTestOp() override {
    auto op = framework::OpRegistry::CreateOp(
        this->op_type_, {{"Y", {"Y"}}, {"DOut", {"DOut"}}, {"DDY", {"DDY"}}},
        {{"DDOut", {"DDOut"}}}, {{"use_mkldnn", false}, {"axis", 0}});
    return op;
  }
};

TEST(test_elementwise_add_grad_grad_without_ddx, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  TestElementwiseAddGradGradWithoutDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}
#ifdef PADDLE_WITH_CUDA
TEST(test_elementwise_add_grad_grad_without_ddx, gpu_place) {
  framework::DDim dims({32, 64});
  platform::CUDAPlace p(0);
  TestElementwiseAddGradGradWithoutDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}
#endif

}  // namespace operators
}  // namespace paddle
