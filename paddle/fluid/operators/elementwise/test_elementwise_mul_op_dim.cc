/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"

USE_OP(elementwise_mul);

namespace paddle {
namespace operators {
#ifdef PADDLE_WITH_MKLDNN

using framework::Scope;
using framework::LoDTensor;
using framework::OpRegistry;
using framework::OperatorBase;
using framework::RuntimeContext;
using framework::ExecutionContext;

struct TestData {
  int64_t channel_num;
  MKLDNNMemoryFormat format;
  framework::DDim y_dims;
  bool supposed_to_fail;
};

void MainTest(const TestData& test_data) {
  auto place = platform::CPUPlace();
  Scope scope;

  auto* x = scope.Var("x")->GetMutable<LoDTensor>();
  auto* y = scope.Var("y")->GetMutable<LoDTensor>();
  scope.Var("out")->GetMutable<LoDTensor>();

  x->Resize({1, test_data.channel_num, 3, 3});
  y->Resize(test_data.y_dims);

  x->set_format(test_data.format);
  y->set_format(MKLDNNMemoryFormat::nc);

  std::unique_ptr<OperatorBase> op = OpRegistry::CreateOp(
      "elementwise_mul", {{"X", {"x"}}, {"Y", {"y"}}}, {{"Out", {"out"}}}, {});

  auto& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = dynamic_cast<platform::MKLDNNDeviceContext*>(pool.Get(place));

  RuntimeContext runtime_ctx =
      RuntimeContext(op->Inputs(), op->Outputs(), scope);
  ExecutionContext ctx = ExecutionContext(*op, scope, *dev_ctx, runtime_ctx);
  bool result = ElementwiseMulOp::AreDimsAndFormatCorrect(
      ctx, 16, MKLDNNMemoryFormat::nChw16c);
  if (test_data.supposed_to_fail)
    ASSERT_FALSE(result);
  else
    ASSERT_TRUE(result);
}

// Checks if AreDimsAndFormatCorrect returns true when supplied with expected
// data
TEST(ElementwiseMulOpTester, correct_dims) {
  TestData test_data;
  test_data.channel_num = 16;
  test_data.format = MKLDNNMemoryFormat::nChw16c;
  test_data.y_dims = {1, test_data.channel_num};
  test_data.supposed_to_fail = false;
  MainTest(test_data);
}

// Checks if AreDimsAndFormatCorrect fails when channel_num is not devisable by
// 16
TEST(ElementwiseMulOpTester, incorrect_channel_num) {
  TestData test_data;
  test_data.channel_num = 17;
  test_data.format = MKLDNNMemoryFormat::nChw16c;
  test_data.y_dims = {1, test_data.channel_num};
  test_data.supposed_to_fail = true;
  MainTest(test_data);
}

// Checks if AreDimsAndFormatCorrect fails when x format is different from
// nchw16c
TEST(ElementwiseMulOpTester, incorrect_format) {
  TestData test_data;
  test_data.channel_num = 16;
  test_data.format = MKLDNNMemoryFormat::nchw;
  test_data.y_dims = {1, test_data.channel_num};
  test_data.supposed_to_fail = true;
  MainTest(test_data);
}

// Checks if AreDimsAndFormatCorrect fails when y input is not 2-dimensional
TEST(ElementwiseMulOpTester, incorrect_y_dims) {
  TestData test_data;
  test_data.channel_num = 16;
  test_data.format = MKLDNNMemoryFormat::nChw16c;
  test_data.y_dims = {1, test_data.channel_num, 1};
  test_data.supposed_to_fail = true;
  MainTest(test_data);
}
#endif
}  // namespace operators
}  // namespace paddle
