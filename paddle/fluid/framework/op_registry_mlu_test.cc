/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace pd = paddle::framework;

namespace paddle {
namespace framework {

class OpKernelTestMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() { AddComment("NoGradOp, same input output. no Grad"); }
};

class OpWithKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::FP32, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class OpKernelTest : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const {}
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(op_with_kernel,
                             paddle::framework::OpWithKernelTest,
                             paddle::framework::OpKernelTestMaker);

REGISTER_OP_MLU_KERNEL(op_with_kernel,
                       paddle::framework::OpKernelTest<
                           paddle::platform::MLUDeviceContext, float>);

TEST(MLUOperatorRegistrar, MLU) {
  paddle::framework::proto::OpDesc op_desc;
  paddle::platform::MLUPlace mlu_place(0);
  paddle::framework::Scope scope;

  op_desc.set_type("op_with_kernel");
  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);

  op->Run(scope, mlu_place);
}

static int op_fallback_test_value = 0;
static int op_test_value = 0;

using paddle::platform::CPUDeviceContext;
using paddle::platform::DeviceContext;
using paddle::platform::MLUDeviceContext;

namespace paddle {
namespace framework {

class OpFallbackKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::FP32, ctx.device_context());
  }
  // framework::OpKernelType GetExpectedKernelType(
  //     const framework::ExecutionContext& ctx) const override {
  //   return framework::OpKernelType(proto::VarType::FP32, platform::MLUPlace(0),
  //                                  DataLayout::kAnyLayout,
  //                                  framework::LibraryType::kPlain);
  // }
};

template <typename DeviceContext, typename T>
class FallbackKernelTest : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const;
};

template <typename T>
class FallbackKernelTest<CPUDeviceContext, T>
    : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const {
    ++op_fallback_test_value;
  }
};

class OpWithMultiKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(InferShapeContext* ctx) const override {}

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(proto::VarType::FP32, platform::MLUPlace(0),
                                   DataLayout::kAnyLayout,
                                   framework::LibraryType::kPlain);
  }
};

template <typename DeviceContext, typename T>
class OpMultiKernelTest : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const;
};

template <typename T>
class OpMultiKernelTest<CPUDeviceContext, T>
    : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const {
    op_test_value += 10;
  }
};

template <typename T>
class OpMultiKernelTest<MLUDeviceContext, T>
    : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const {
    op_test_value -= 10;
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(fallback_kernel,
                             paddle::framework::OpFallbackKernelTest,
                             paddle::framework::OpKernelTestMaker);
REGISTER_OP_KERNEL(
    fallback_kernel, CPU, paddle::platform::CPUPlace,
    paddle::framework::FallbackKernelTest<CPUDeviceContext, float>);

TEST(MLUOperatorRegistrar, OpFallbackKernel) {
  paddle::framework::proto::OpDesc op_desc;;
  paddle::platform::MLUPlace mlu_place;
  paddle::framework::Scope scope;

  op_desc.set_type("fallback_kernel");
  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);

  op->Run(scope, mlu_place);
  EXPECT_EQ(op_fallback_test_value, 1);
}

REGISTER_OP_WITHOUT_GRADIENT(op_with_multi_kernel,
                             paddle::framework::OpWithMultiKernelTest,
                             paddle::framework::OpKernelTestMaker);
REGISTER_OP_KERNEL(
    op_with_multi_kernel, CPU, paddle::platform::CPUPlace,
    paddle::framework::OpMultiKernelTest<CPUDeviceContext, float>);
REGISTER_OP_KERNEL(
    op_with_multi_kernel, MLU, paddle::platform::MLUPlace,
    paddle::framework::OpMultiKernelTest<MLUDeviceContext, float>);

TEST(MLUOperatorRegistrar, OpWithMultiKernel) {
  paddle::framework::proto::OpDesc op_desc;;
  paddle::platform::MLUPlace mlu_place;
  paddle::framework::Scope scope;

  op_desc.set_type("op_with_multi_kernel");
  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);

  op->Run(scope, mlu_place);
  EXPECT_EQ(op_test_value, -10);
}
