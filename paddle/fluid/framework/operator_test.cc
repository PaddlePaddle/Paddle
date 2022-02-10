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
#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/init.h"

DECLARE_bool(enable_unused_var_check);

namespace paddle {
namespace framework {

static int op_run_num = 0;

class OpWithoutKernelTest : public OperatorBase {
 public:
  OpWithoutKernelTest(const std::string& type, const VariableNameMap& inputs,
                      const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs), x(1) {}

 private:
  void RunImpl(const Scope& scope,
               const platform::Place& place) const override {
    ++op_run_num;
    ASSERT_EQ(static_cast<int>(inputs_.size()), 1);
    ASSERT_EQ(static_cast<int>(outputs_.size()), 1);
    ASSERT_EQ(scope.FindVar(inputs_.at("input")[0]), nullptr);
    ASSERT_EQ(x, 1);
    ASSERT_NE(scope.FindVar(outputs_.at("output")[0]), nullptr);
  }

 public:
  int x{0};
};

class OpWithoutKernelCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("input", "input of test op");
    AddOutput("output", "output of test op");
    AddAttr<float>("scale", "scale of cosine op");
    AddAttr<int>("kernel_sub_type", "kernels with different implementations.")
        .SetDefault(0);
    AddComment("This is test op");
  }
};

}  // namespace framework
}  // namespace paddle

static void BuildVar(const std::string& param_name,
                     std::initializer_list<const char*> arguments,
                     paddle::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    *var->mutable_arguments()->Add() = arg_name;
  }
}

REGISTER_OP_WITHOUT_GRADIENT(test_operator,
                             paddle::framework::OpWithoutKernelTest,
                             paddle::framework::OpWithoutKernelCheckerMaker);

TEST(OperatorBase, all) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("test_operator");
  BuildVar("input", {"IN1"}, op_desc.add_inputs());
  BuildVar("output", {"OUT1"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  scope.Var("OUT1");
  ASSERT_EQ(paddle::framework::op_run_num, 0);
  op->Run(scope, cpu_place);
  ASSERT_EQ(paddle::framework::op_run_num, 1);
}

namespace paddle {
namespace framework {

static int special_type_value = 1;

class OpKernelTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("x", "input of test op");
    AddOutput("y", "output of test op");
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .GreaterThan(0.0);
    AddAttr<int>("kernel_sub_type", "kernels with different implementations.")
        .SetDefault(0);
    AddComment("This is test op");
  }
};

static int cpu_kernel_run_num = 0;
static int cpu_kernel2_run_num = 0;

class OpWithKernelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    int sub_type = ctx.Attr<int>("kernel_sub_type");
    return OpKernelType(proto::VarType::FP32, ctx.GetPlace(),
                        framework::DataLayout::kAnyLayout,
                        framework::LibraryType::kPlain, sub_type);
  }
};

template <typename T1, typename T2>
class CPUKernelTest : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << ctx.DebugString() << std::endl;
    cpu_kernel_run_num++;
    ASSERT_EQ(ctx.InputName("x"), "IN1");
    ASSERT_EQ(ctx.OutputName("y"), "OUT1");
    auto* x = ctx.Input<Tensor>("X");
    ASSERT_EQ(x, nullptr);
  }
};

template <typename T1, typename T2>
class CPUKernel2Test : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    std::cout << ctx.DebugString() << std::endl;
    cpu_kernel2_run_num++;
    ASSERT_EQ(ctx.InputName("x"), "IN1");
    ASSERT_EQ(ctx.OutputName("y"), "OUT1");
  }
};

class OpKernelTestMultiInputsProtoAndCheckerMaker
    : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("xs", "inputs of test op").AsDuplicable();
    AddInput("k", "input of test op");
    AddOutput("ys", "outputs of test op").AsDuplicable();
    AddAttr<float>("scale", "scale of cosine op")
        .SetDefault(1.0)
        .GreaterThan(0.0);
    AddAttr<int>("kernel_sub_type", "kernels with different implementations.")
        .SetDefault(0);
    AddComment("This is test op");
  }
};

class CPUKernalMultiInputsTest : public OpKernel<float> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    auto xs = ctx.InputNames("xs");
    ASSERT_EQ(xs.size(), 3UL);
    ASSERT_EQ(xs[0], "x0");
    ASSERT_EQ(xs[1], "x1");
    ASSERT_EQ(xs[2], "x2");

    auto inVar0 = ctx.MultiInputVar("xs");
    ASSERT_EQ(inVar0.size(), 3U);

    auto intVar1 = ctx.InputVar("k");
    ASSERT_NE(intVar1, nullptr);

    auto outVar0 = ctx.MultiOutputVar("ys");
    ASSERT_EQ(outVar0.size(), 2U);

    auto inTensor0 = ctx.MultiInput<Tensor>("xs");
    ASSERT_EQ(inTensor0.size(), 3U);

    auto intTensor1 = ctx.Input<Tensor>("k");
    ASSERT_NE(intTensor1, nullptr);

    auto outTensor0 = ctx.MultiOutput<Tensor>("ys");
    ASSERT_EQ(outTensor0.size(), 2U);

    auto k = ctx.InputName("k");
    ASSERT_EQ(k, "k0");

    auto ys = ctx.OutputNames("ys");
    ASSERT_EQ(ys.size(), 2UL);
    ASSERT_EQ(ys[0], "y0");
    ASSERT_EQ(ys[1], "y1");
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    op_with_kernel, paddle::framework::OpWithKernelTest,
    paddle::framework::OpKernelTestProtoAndCheckerMaker);

REGISTER_OP_CPU_KERNEL(op_with_kernel,
                       paddle::framework::CPUKernelTest<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    op_with_kernel, CPU, paddle::platform::CPUPlace, MY_SPECIAL_NAME,
    paddle::framework::special_type_value,
    paddle::framework::CPUKernel2Test<float, float>);

// test with single input
TEST(OpKernel, all) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("op_with_kernel");
  BuildVar("x", {"IN1"}, op_desc.add_inputs());
  BuildVar("y", {"OUT1"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 0);
  op->Run(scope, cpu_place);
  // kerne_sub_type = 0, hence cpu_kernel is called, cpu_kernel2 is not called.
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 1);
  ASSERT_EQ(paddle::framework::cpu_kernel2_run_num, 0);

  attr = op_desc.mutable_attrs()->Add();
  attr->set_name("kernel_sub_type");
  attr->set_type(paddle::framework::proto::AttrType::INT);
  attr->set_i(1);
  auto op2 = paddle::framework::OpRegistry::CreateOp(op_desc);
  op2->Run(scope, cpu_place);
  // kerne_sub_type = 1, hence cpu_kernel2 is called, cpu_kernel is not called.
  ASSERT_EQ(paddle::framework::cpu_kernel_run_num, 1);
  ASSERT_EQ(paddle::framework::cpu_kernel2_run_num, 1);
}

REGISTER_OP_WITHOUT_GRADIENT(
    op_multi_inputs_with_kernel, paddle::framework::OpWithKernelTest,
    paddle::framework::OpKernelTestMultiInputsProtoAndCheckerMaker);
REGISTER_OP_CPU_KERNEL(op_multi_inputs_with_kernel,
                       paddle::framework::CPUKernalMultiInputsTest);

// test with multi inputs
TEST(OpKernel, multi_inputs) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;

  op_desc.set_type("op_multi_inputs_with_kernel");
  BuildVar("xs", {"x0", "x1", "x2"}, op_desc.add_inputs());
  BuildVar("k", {"k0"}, op_desc.add_inputs());
  BuildVar("ys", {"y0", "y1"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;
  scope.Var("x0")->GetMutable<paddle::framework::LoDTensor>();
  scope.Var("x1")->GetMutable<paddle::framework::LoDTensor>();
  scope.Var("x2")->GetMutable<paddle::framework::LoDTensor>();
  scope.Var("k0")->GetMutable<paddle::framework::LoDTensor>();
  scope.Var("y0")->GetMutable<paddle::framework::LoDTensor>();
  scope.Var("y1")->GetMutable<paddle::framework::LoDTensor>();

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  op->Run(scope, cpu_place);
}

TEST(VarNameTest, all) {
  std::string var_name("X");
  std::string grad_var_name = paddle::framework::GradVarName(var_name);
  ASSERT_EQ(grad_var_name, "X@GRAD");
  std::string original_var_name =
      paddle::framework::GradOriginalVarName(grad_var_name);
  ASSERT_EQ(original_var_name, "X");
  original_var_name = paddle::framework::GradOriginalVarName(original_var_name);
  ASSERT_EQ(original_var_name, "X");

  std::string var_name_2("XYZ");
  grad_var_name = paddle::framework::GradVarName(var_name_2);
  ASSERT_EQ(grad_var_name, "XYZ@GRAD");
  original_var_name = paddle::framework::GradOriginalVarName(grad_var_name);
  ASSERT_EQ(original_var_name, "XYZ");
  original_var_name = paddle::framework::GradOriginalVarName(original_var_name);
  ASSERT_EQ(original_var_name, "XYZ");

  std::string var_name_3("");
  grad_var_name = paddle::framework::GradVarName(var_name_3);
  ASSERT_EQ(grad_var_name, "@GRAD");
  original_var_name = paddle::framework::GradOriginalVarName(grad_var_name);
  ASSERT_EQ(original_var_name, "");
  original_var_name = paddle::framework::GradOriginalVarName(original_var_name);
  ASSERT_EQ(original_var_name, "");
}

namespace paddle {
namespace framework {

class IndicateLoDTensorDataTypeTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "LoDTensor");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class IndicateLoDTensorDataTypeTestProtoMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("LoDTensor", "Input of Tensor type Variable.");
    AddComment("This Op is only for IndicateVarDataType interface test.");
  }
};

class IndicateSelectedRowsDataTypeTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    auto data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "SelectedRows");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};
class IndicateSelectedRowsDataTypeTestProtoMaker
    : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("SelectedRows", "Input of SelectedRows type Variable.");
    AddComment("This Op is only for IndicateVarDataType interface test.");
  }
};

class IndicateOtherDataTypeTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Other");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};
class IndicateOtherDataTypeTestProtoMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Other", "Input of Other type Variable");
    AddComment("This Op is only for IndicateVarDataType interface test.");
  }
};

template <typename DeviceContext, typename T>
class EmptyTestKernel : public OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const {}
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    indicate_lod_tensor_data_type_test,
    paddle::framework::IndicateLoDTensorDataTypeTest,
    paddle::framework::IndicateLoDTensorDataTypeTestProtoMaker);
REGISTER_OP_WITHOUT_GRADIENT(
    indicate_selected_rows_data_type_test,
    paddle::framework::IndicateSelectedRowsDataTypeTest,
    paddle::framework::IndicateSelectedRowsDataTypeTestProtoMaker);
REGISTER_OP_WITHOUT_GRADIENT(
    indicate_other_data_type_test, paddle::framework::IndicateOtherDataTypeTest,
    paddle::framework::IndicateOtherDataTypeTestProtoMaker);

REGISTER_OP_CPU_KERNEL(indicate_lod_tensor_data_type_test,
                       paddle::framework::EmptyTestKernel<
                           paddle::platform::CPUDeviceContext, int>);
REGISTER_OP_CPU_KERNEL(indicate_selected_rows_data_type_test,
                       paddle::framework::EmptyTestKernel<
                           paddle::platform::CPUDeviceContext, int>);
REGISTER_OP_CPU_KERNEL(indicate_other_data_type_test,
                       paddle::framework::EmptyTestKernel<
                           paddle::platform::CPUDeviceContext, int>);

TEST(IndicateVarDataTypeTest, lodtensor) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("indicate_lod_tensor_data_type_test");
  BuildVar("LoDTensor", {"lodtensor_1"}, op_desc.add_inputs());

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  auto* var = scope.Var("lodtensor_1");
  var->GetMutable<paddle::framework::LoDTensor>();

  bool caught = false;
  try {
    op->Run(scope, cpu_place);
  } catch (paddle::platform::EnforceNotMet& err) {
    caught = true;
    std::string ex_msg = err.what();
    EXPECT_TRUE(
        ex_msg.find(
            "The indicate_lod_tensor_data_type_test Op's Input Variable "
            "`LoDTensor` contains uninitialized Tensor.") != std::string::npos);
  }
  ASSERT_TRUE(caught);
}

TEST(IndicateVarDataTypeTest, selectedrows) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("indicate_selected_rows_data_type_test");
  BuildVar("SelectedRows", {"selected_rows_1"}, op_desc.add_inputs());

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  auto* var = scope.Var("selected_rows_1");
  var->GetMutable<pten::SelectedRows>();

  bool caught = false;
  try {
    op->Run(scope, cpu_place);
  } catch (paddle::platform::EnforceNotMet& err) {
    caught = true;
    std::string ex_msg = err.what();
    EXPECT_TRUE(
        ex_msg.find("The indicate_selected_rows_data_type_test Op's "
                    "Input Variable `SelectedRows` contains uninitialized "
                    "Tensor.") != std::string::npos);
  }
  ASSERT_TRUE(caught);
}

TEST(IndicateVarDataTypeTest, other) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("indicate_other_data_type_test");
  BuildVar("Other", {"lod_rank_table_1"}, op_desc.add_inputs());

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  auto* var = scope.Var("lod_rank_table_1");
  var->GetMutable<paddle::framework::LoDRankTable>();

  bool caught = false;
  try {
    op->Run(scope, cpu_place);
  } catch (paddle::platform::EnforceNotMet& err) {
    caught = true;
    std::string ex_msg = err.what();
    EXPECT_TRUE(
        ex_msg.find(
            "The Input Variable(Other) of "
            "(indicate_other_data_type_test) Operator used to "
            "determine kernel data type "
            "is empty or not LoDTensor or SelectedRows or LoDTensorArray.") !=
        std::string::npos);
  }
  ASSERT_TRUE(caught);
}

TEST(ExecutionContextAttrAndInOut, new_api) {
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("test_operator");
  BuildVar("input", {"IN1"}, op_desc.add_inputs());
  BuildVar("output", {"OUT1"}, op_desc.add_outputs());

  auto attr = op_desc.mutable_attrs()->Add();
  attr->set_name("scale");
  attr->set_type(paddle::framework::proto::AttrType::FLOAT);
  attr->set_f(3.14);

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  auto* var = scope.Var("OUT1");
  var->GetMutable<paddle::framework::LoDTensorArray>();

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(cpu_place);

  paddle::framework::RuntimeContext ctx({}, {});
  paddle::framework::ExecutionContext exe_context(*(op.get()), scope, *dev_ctx,
                                                  ctx);

  ASSERT_EQ(exe_context.InputSize("input"), 1u);
  ASSERT_EQ(exe_context.OutputSize("output"), 1u);

  auto attr_map = exe_context.Attrs();
  ASSERT_EQ(BOOST_GET(float, attr_map["scale"]), 3.14f);
  ASSERT_EQ(exe_context.Type(), "test_operator");
}

namespace paddle {
namespace framework {

class GetLoDLevelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "GetLoDLevelTest");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "GetLoDLevelTest");

    auto lod_level = ctx->GetLoDLevel("X");
    PADDLE_ENFORCE_GT(lod_level, 0,
                      paddle::platform::errors::InvalidArgument(
                          "The LoD level Input(X) should be larger than 0."));
  }
};

class SetLoDLevelTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "SetLoDLevelTest");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SetLoDLevelTest");
    ctx->SetLoDLevel("Out", 1);
  }
};

class GetSetLoDLevelTestMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(LoDTensor) Input Variable.");
    AddOutput("Out", "(LoDTensor) Output Variable.");
    AddComment("This Op is only for Get/SetLoDLevel interface test.");
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(get_lod_level_test,
                             paddle::framework::GetLoDLevelTest,
                             paddle::framework::GetSetLoDLevelTestMaker);
REGISTER_OP_CPU_KERNEL(get_lod_level_test,
                       paddle::framework::EmptyTestKernel<
                           paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_WITHOUT_GRADIENT(set_lod_level_test,
                             paddle::framework::SetLoDLevelTest,
                             paddle::framework::GetSetLoDLevelTestMaker);
REGISTER_OP_CPU_KERNEL(set_lod_level_test,
                       paddle::framework::EmptyTestKernel<
                           paddle::platform::CPUDeviceContext, float>);

void SetGetLoDLevelTestMain(std::string op_type) {
  paddle::framework::InitDevices({});
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type(op_type);
  BuildVar("X", {"x.0"}, op_desc.add_inputs());
  BuildVar("Out", {"out.0"}, op_desc.add_outputs());

  paddle::platform::CPUPlace place;
  paddle::framework::Scope scope;

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  auto* x_var = scope.Var("x.0");
  auto* x = x_var->GetMutable<paddle::framework::LoDTensor>();
  x->mutable_data<float>(paddle::framework::make_ddim({64}), place);
  auto* out_var = scope.Var("out.0");
  out_var->GetMutable<paddle::framework::LoDTensor>();

  bool caught = false;
  std::string err_str =
      (op_type == "get_lod_level_test") ? "GetLoDLevel" : "SetLoDLevel";
  err_str +=
      " is only used in compile time. The calculation of output's actual lod "
      "is different among operators so that should be set in the runtime "
      "kernel.";
  try {
    op->Run(scope, place);
  } catch (paddle::platform::EnforceNotMet& err) {
    caught = true;
    std::string ex_msg = err.what();
    EXPECT_TRUE(ex_msg.find(err_str) != std::string::npos);
  }
  ASSERT_TRUE(caught);
}

TEST(GetLoDLevelTest, base) { SetGetLoDLevelTestMain("get_lod_level_test"); }

TEST(SetLoDLevelTest, base) { SetGetLoDLevelTestMain("set_lod_level_test"); }

namespace paddle {
namespace framework {

class OpUnusedVarTest : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext& ctx) const override {
    return OpKernelType(proto::VarType::FP32, ctx.GetPlace(),
                        framework::DataLayout::kAnyLayout);
  }
};

class OpUnusedVarTestProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "input of test op");
    AddOutput("Y", "output of test op");
    AddComment("This is test op for unused var check.");
  }
};

template <typename T>
class OpWithUnusedVarKernelTest : public OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    ASSERT_EQ(ctx.InputName("X"), "X");
    ASSERT_EQ(ctx.OutputName("Y"), "Y");
  }
};

template <typename T>
class OpWithoutUnusedVarKernelTest : public OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const {
    ASSERT_EQ(ctx.InputName("X"), "X");
    ASSERT_EQ(ctx.OutputName("Y"), "Y");
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    ASSERT_NE(x, y);
    ASSERT_NE(y, nullptr);
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    op_with_unused_var, paddle::framework::OpUnusedVarTest,
    paddle::framework::OpUnusedVarTestProtoAndCheckerMaker);

REGISTER_OP_CPU_KERNEL(op_with_unused_var,
                       paddle::framework::OpWithUnusedVarKernelTest<float>);

REGISTER_OP_WITHOUT_GRADIENT(
    op_without_unused_var, paddle::framework::OpUnusedVarTest,
    paddle::framework::OpUnusedVarTestProtoAndCheckerMaker);

REGISTER_OP_CPU_KERNEL(op_without_unused_var,
                       paddle::framework::OpWithoutUnusedVarKernelTest<float>);

// test with single input
TEST(OpWithUnusedVar, all) {
  // enable the unused_var_check
  FLAGS_enable_unused_var_check = true;
  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("op_with_unused_var");
  BuildVar("X", {"X"}, op_desc.add_inputs());
  BuildVar("Y", {"Y"}, op_desc.add_outputs());

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;
  auto* x = scope.Var("X")->GetMutable<paddle::framework::LoDTensor>();
  auto* y = scope.Var("Y")->GetMutable<paddle::framework::LoDTensor>();
  x->Resize({32, 64});
  y->Resize({32, 64});
  x->mutable_data<float>(cpu_place);
  y->mutable_data<float>(cpu_place);

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  // should throw exception
  ASSERT_THROW(op->Run(scope, cpu_place), paddle::platform::EnforceNotMet);
  FLAGS_enable_unused_var_check = false;
}

TEST(OpWithoutUnusedVar, all) {
  // enable the unused_var_check
  FLAGS_enable_unused_var_check = true;

  paddle::framework::InitDevices();
  paddle::framework::proto::OpDesc op_desc;
  op_desc.set_type("op_without_unused_var");
  BuildVar("X", {"X"}, op_desc.add_inputs());
  BuildVar("Y", {"Y"}, op_desc.add_outputs());

  paddle::platform::CPUPlace cpu_place;
  paddle::framework::Scope scope;
  auto* x = scope.Var("X")->GetMutable<paddle::framework::LoDTensor>();
  auto* y = scope.Var("Y")->GetMutable<paddle::framework::LoDTensor>();
  x->Resize({32, 64});
  y->Resize({32, 64});
  x->mutable_data<float>(cpu_place);
  y->mutable_data<float>(cpu_place);

  auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
  // should not throw exception
  ASSERT_NO_THROW(op->Run(scope, cpu_place));
  FLAGS_enable_unused_var_check = false;
}
