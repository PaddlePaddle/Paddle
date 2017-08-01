/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/operator.h"
#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

static int op_run_num = 0;

class OpWithoutKernelTest : public OperatorBase {
 public:
  void Init() override { x = 1; }
  void InferShape(const Scope& scope) const override {}
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    op_run_num++;
    ASSERT_EQ((int)inputs_.size(), 1);
    ASSERT_EQ((int)outputs_.size(), 1);
    ASSERT_EQ(scope.FindVar(inputs_[0]), nullptr);
    ASSERT_EQ(x, 1);
    ASSERT_NE(scope.FindVar(outputs_[0]), nullptr);
  }

 public:
  float x = 0;
};

class OpeWithoutKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OpeWithoutKernelTestProtoAndCheckerMaker(OpProto* proto,
                                           OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op");
    AddComment("This is test op");
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP(test_operator, paddle::framework::OpWithoutKernelTest,
            paddle::framework::OpeWithoutKernelTestProtoAndCheckerMaker);

TEST(OperatorBase, all) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("test_operator");
  *op_desc.mutable_inputs()->Add() = "IN1";
  *op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUDeviceContext device_context;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  scope.NewVar("OUT1");
  ASSERT_EQ(paddle::framework::op_run_num, 0);
  op->InferShape(scope);
  op->Run(scope, device_context);
  ASSERT_EQ(paddle::framework::op_run_num, 1);
}

namespace paddle {
namespace framework {

class OpKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OpKernelTestProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("x", "input of test op");
    AddOutput("y", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is test op");
  }
};

static int cpu_kernel_run_num = 0;

class OpWithKernelTest : public OperatorWithKernel {
 protected:
  void InferShape(const framework::InferShapeContext& ctx) const override {}
};

template <typename T1, typename T2>
class CPUKernelTest : public OpKernel {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << "this is cpu kernel" << std::endl;
    std::cout << ctx.op_.DebugString() << std::endl;
    cpu_kernel_run_num++;
    ASSERT_EQ(ctx.op_.Input("x"), "IN1");
    ASSERT_EQ(ctx.op_.Output("y"), "OUT1");
  }
};

// multiple inputs test
class OperatorMultiInputsTest : public OperatorBase {
 public:
  void Init() override { x = 1; }
  void InferShape(const Scope& scope) const override {}
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    ASSERT_EQ(scope.FindVar(inputs_[0]), nullptr);
    ASSERT_EQ(x, 1);
    ASSERT_NE(scope.FindVar(outputs_[0]), nullptr);
    ASSERT_EQ(Input("x"), "IN1");
    ASSERT_EQ(Input("y"), "OUT1");
  }

 public:
  float x = 0;
};

class OpKernelTestMultiInputsProtoAndCheckerMaker
    : public OpProtoAndCheckerMaker {
 public:
  OpKernelTestMultiInputsProtoAndCheckerMaker(OpProto* proto,
                                              OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("xs", "inputs of test op").SetMultiple();
    AddInput("k", "input of test op");
    AddOutput("ys", "outputs of test op").SetMultiple();
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .LargerThan(0.0);
    AddComment("This is test op");
  }
};

class CPUKernalMultiInputsTest : public OpKernel {
 public:
  void Compute(const ExecutionContext& ctx) const {
    auto xs = ctx.op_.Inputs("xs");
    ASSERT_EQ(xs.size(), 3UL);
    ASSERT_EQ(xs[0], "x0");
    ASSERT_EQ(xs[1], "x1");
    ASSERT_EQ(xs[2], "x2");

    auto inVar0 = ctx.MultiInputVar("xs");
    ASSERT_EQ(inVar0.size(), 3);

    auto intVar1 = ctx.InputVar("k");
    ASSERT_NE(intVar1, nullptr);

    auto outVar0 = ctx.MultiOutputVar("ys");
    ASSERT_EQ(outVar0.size(), 2);

    auto inTensor0 = ctx.MultiInput<Tensor>("xs");
    ASSERT_EQ(inTensor0.size(), 3);

    auto intTensor1 = ctx.Input<Tensor>("k");
    ASSERT_NE(intTensor1, nullptr);

    auto outTensor0 = ctx.MultiOutput<Tensor>("ys");
    ASSERT_EQ(outTensor0.size(), 2);

    auto k = ctx.op_.Input("k");
    ASSERT_EQ(k, "k0");

    auto ys = ctx.op_.Outputs("ys");
    ASSERT_EQ(ys.size(), 2UL);
    ASSERT_EQ(ys[0], "y0");
    ASSERT_EQ(ys[1], "y1");
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP(op_with_kernel, paddle::framework::OpWithKernelTest,
            paddle::framework::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(op_with_kernel,
                       paddle::framework::CPUKernelTest<float, float>);

// test with single input
TEST(OpKernel, all) {
  paddle::framework::OpDesc op_desc;
  op_desc.set_type("op_with_kernel");
  *op_desc.mutable_inputs()->Add() = "IN1";
  *op_desc.mutable_outputs()->Add() = "OUT1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUDeviceContext cpu_device_context;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 0);
  op->Run(scope, cpu_device_context);
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 1);
}

REGISTER_OP(op_multi_inputs_with_kernel, paddle::framework::OpWithKernelTest,
            paddle::framework::OpKernelTestMultiInputsProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(op_multi_inputs_with_kernel,
                       paddle::framework::CPUKernalMultiInputsTest);

// test with multi inputs
TEST(OpKernel, multi_inputs) {
  using namespace paddle::framework;

  OpDesc op_desc;
  op_desc.set_type("op_multi_inputs_with_kernel");
  *op_desc.mutable_inputs()->Add() = "x0";
  *op_desc.mutable_inputs()->Add() = "x1";
  *op_desc.mutable_inputs()->Add() = "x2";
  *op_desc.mutable_inputs()->Add() = "k0";
  *op_desc.mutable_outputs()->Add() = "y0";
  *op_desc.mutable_outputs()->Add() = "y1";
  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::AttrType::FLOAT);
  attr->set_f(3.14);

  auto attr0 = op_desc.mutable_attrs()->Add();
  attr0->set_name("input_format");
  attr0->set_type(paddle::framework::AttrType::INTS);
  auto input_format = attr0->mutable_ints();
  input_format->Add(0);  // x0
  input_format->Add(3);  // k
  input_format->Add(4);  // end

  auto attr1 = op_desc.mutable_attrs()->Add();
  attr1->set_name("output_format");
  attr1->set_type(paddle::framework::AttrType::INTS);
  auto output_format = attr1->mutable_ints();
  output_format->Add(0);  // y0
  output_format->Add(2);  // y1

  paddle::platform::CPUDeviceContext cpu_device_context;
  paddle::framework::Scope scope;
  scope.NewVar("x0")->GetMutable<Tensor>();
  scope.NewVar("x1")->GetMutable<Tensor>();
  scope.NewVar("x2")->GetMutable<Tensor>();
  scope.NewVar("k0")->GetMutable<Tensor>();
  scope.NewVar("y0")->GetMutable<Tensor>();
  scope.NewVar("y1")->GetMutable<Tensor>();

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  op->Run(scope, cpu_device_context);
}
