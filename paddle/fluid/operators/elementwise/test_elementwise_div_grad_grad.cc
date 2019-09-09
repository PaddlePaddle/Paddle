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

USE_OP(elementwise_div);

namespace paddle {
namespace operators {

template <typename T>
class TestElementwiseDivGradGradWithoutDout
    : public TestElementwiseOpGradGrad<T> {
 public:
  TestElementwiseDivGradGradWithoutDout(const platform::Place &place_,
                                        const framework::DDim &dims_)
      : TestElementwiseOpGradGrad<T>("elementwise_div_grad_grad", place_, dims_,
                                     {"Y", "Out", "DDX", "DDY", "DX"},
                                     {"Y@GRAD", "DDOut"}) {}

  using TestElementwiseOpGradGrad<T>::feed_datas;
  using TestElementwiseOpGradGrad<T>::expected_outs;
  using TestElementwiseOpGradGrad<T>::dims;
  void ComputeExpectedOuts() override {
    size_t numel = static_cast<size_t>(framework::product(this->dims));
    std::vector<T> dy(numel);
    std::vector<T> ddout(numel);
    for (size_t i = 0; i < numel; ++i) {
      // dY(Y@GRAD) = Out * dX * ddY / Y - dX * ddX / Y
      dy[i] =
          (feed_datas["DX"][i] / feed_datas["Y"][i]) *
          (feed_datas["Out"][i] * feed_datas["DDY"][i] - feed_datas["DDX"][i]);
      // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
      ddout[i] =
          (feed_datas["DDX"][i] - feed_datas["Out"][i] * feed_datas["DDY"][i]) /
          (feed_datas["Y"][i]);
    }
    expected_outs["Y@GRAD"] = dy;
    expected_outs["DDOut"] = ddout;
  }

  std::unique_ptr<framework::OperatorBase> CreateTestOp() override {
    auto op = framework::OpRegistry::CreateOp(
        this->op_type, {{"Y", {"Y"}},
                        {"Out", {"Out"}},
                        {"DDX", {"DDX"}},
                        {"DDY", {"DDY"}},
                        {"DX", {"DX"}}},
        {{"Y@GRAD", {"Y@GRAD"}}, {"DDOut", {"DDOut"}}},
        {{"use_mkldnn", false}, {"axis", 0}});
    return op;
  }
};

TEST(test_elementwise_div_grad_grad_without_dout, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  TestElementwiseDivGradGradWithoutDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}

#ifdef PADDLE_WITH_CUDA
TEST(test_elementwise_div_grad_grad_without_dout, gpu_place) {
  framework::DDim dims({32, 64});
  platform::CUDAPlace p(0);
  TestElementwiseDivGradGradWithoutDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}
#endif

}  // namespace operators
}  // namespace paddle
