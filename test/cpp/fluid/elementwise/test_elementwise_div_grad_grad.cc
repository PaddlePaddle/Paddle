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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "test/cpp/fluid/elementwise/test_elementwise_op_grad_grad.h"

USE_OP_ITSELF(elementwise_div);

PD_DECLARE_KERNEL(divide_double_grad, CPU, ALL_LAYOUT);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(divide_double_grad, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace operators {

template <typename T>
class TestElementwiseDivGradGradWithDout : public TestElementwiseOpGradGrad<T> {
 public:
  TestElementwiseDivGradGradWithDout(const phi::Place &place,
                                     const phi::DDim &dims)
      : TestElementwiseOpGradGrad<T>(
            "elementwise_div_grad_grad",
            place,
            dims,
            {"Y", "Out", "Out@GRAD", "DDX", "DDY", "DX"},
            {"Y@GRAD", "DDOut", "DOut"}) {}

  using TestElementwiseOpGradGrad<T>::feed_datas_;
  using TestElementwiseOpGradGrad<T>::expected_outs_;
  using TestElementwiseOpGradGrad<T>::dims_;
  void ComputeExpectedOuts() override {
    size_t numel = static_cast<size_t>(common::product(dims_));
    std::vector<T> dy(numel);
    std::vector<T> ddout(numel);
    std::vector<T> dout(numel);
    for (size_t i = 0; i < numel; ++i) {
      // dY(Y@GRAD) = Out * dX * ddY / Y - dX * ddX / Y
      dy[i] = (feed_datas_["DX"][i] / feed_datas_["Y"][i]) *
              (feed_datas_["Out"][i] * feed_datas_["DDY"][i] -
               feed_datas_["DDX"][i]);
      // ddOut = ddX / Y - Out * ddY / Y = (ddX - Out * ddY) / Y
      ddout[i] = (feed_datas_["DDX"][i] -
                  feed_datas_["Out"][i] * feed_datas_["DDY"][i]) /
                 (feed_datas_["Y"][i]);
      // dOut = - DX * DDy
      dout[i] = -feed_datas_["DX"][i] * feed_datas_["DDY"][i];
    }
    expected_outs_["Y@GRAD"] = dy;
    expected_outs_["DDOut"] = ddout;
    expected_outs_["DOut"] = dout;
  }

  std::unique_ptr<framework::OperatorBase> CreateTestOp() override {
    auto op = framework::OpRegistry::CreateOp(
        this->op_type_,
        {{"Y", {"Y"}},
         {"Out", {"Out"}},
         {"Out@GRAD", {"Out@GRAD"}},
         {"DDX", {"DDX"}},
         {"DDY", {"DDY"}},
         {"DX", {"DX"}}},
        {{"Y@GRAD", {"Y@GRAD"}}, {"DDOut", {"DDOut"}}, {"DOut", {"DOut"}}},
        {{"use_mkldnn", false}, {"axis", 0}});
    return op;
  }
};

TEST(test_elementwise_div_grad_grad, cpu_place) {
  phi::DDim dims({32, 64});
  phi::CPUPlace p;
  TestElementwiseDivGradGradWithDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(test_elementwise_div_grad_grad, gpu_place) {
  phi::DDim dims({32, 64});
  phi::GPUPlace p(0);
  TestElementwiseDivGradGradWithDout<float> test(p, dims);
  ASSERT_TRUE(test.Check());
}
#endif

}  // namespace operators
}  // namespace paddle
