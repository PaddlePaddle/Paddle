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

#include "gtest/gtest.h"

#include "paddle/framework/init.h"
#include "paddle/framework/op_info.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class OpKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  OpKernelTestProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddComment("This is test op");
  }
};

static int cpu_kernel_run_num = 0;
static int op_run_num = 0;

class CPUOpWithKernel : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetActualKernelType(const ExecutionContext& ctx) const override {
    return OpKernelType(proto::DataType::FP32, ctx.GetPlace());
  }
};

class CPUKernel : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << ctx.op().DebugString() << std::endl;
    cpu_kernel_run_num++;
    ASSERT_EQ(ctx.op().Input("input"), "IN1");
    ASSERT_EQ(ctx.op().Output("output"), "OUT1");
  }
};

class GeneralOpWithKernel : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetActualKernelType(const ExecutionContext& ctx) const override {
    return OpKernelType(proto::DataType::FP32, ctx.GetPlace());
  }
};

class GeneralKernel : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << ctx.op().DebugString() << std::endl;
    op_run_num++;
    ASSERT_EQ(ctx.op().Input("input"), "IN2");
    ASSERT_EQ(ctx.op().Output("output"), "OUT2");
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    cpu_op, paddle::framework::CPUOpWithKernel,
    paddle::framework::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(cpu_op, paddle::framework::CPUKernel);

REGISTER_OP_WITHOUT_GRADIENT(
    general_op, paddle::framework::GeneralOpWithKernel,
    paddle::framework::OpKernelTestProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(general_op, paddle::framework::GeneralKernel);

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    *var->mutable_arguments()->Add() = arg_name;
  }
}

TEST(Operator, MulitDevice) {
  paddle::framework::InitDevices({"CPU"});

  paddle::framework::Scope scope;

  // create a op to run on CPU
  paddle::framework::proto::OpDesc cpu_op_desc;
  cpu_op_desc.set_type("cpu_op");
  BuildVar("input", {"IN1"}, cpu_op_desc.add_inputs());
  BuildVar("output", {"OUT1"}, cpu_op_desc.add_outputs());

  paddle::platform::CPUPlace cpu_place;

  auto cpu_op = paddle::framework::OpRegistry::CreateOp(cpu_op_desc);
  scope.Var("OUT1");
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 0);
  cpu_op->Run(scope, cpu_place);
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 1);

  // create a op to run on GPU
  paddle::framework::proto::OpDesc general_op_desc;
  general_op_desc.set_type("general_op");
  BuildVar("input", {"IN2"}, general_op_desc.add_inputs());
  BuildVar("output", {"OUT2"}, general_op_desc.add_outputs());

  paddle::platform::CUDAPlace cuda_place(0);

  auto general_op = paddle::framework::OpRegistry::CreateOp(general_op_desc);
  scope.Var("OUT2");
  ASSERT_EQ(paddle::framework::op_run_num, 0);
  general_op->Run(scope, cuda_place);
  ASSERT_EQ(paddle::framework::op_run_num, 1);
}
