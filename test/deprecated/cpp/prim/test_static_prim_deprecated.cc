// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/prim/api/manual_prim/utils/utils.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/fluid/prim/utils/static/static_tensor_operants.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/phi/api/include/operants_manager.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_bool(prim_enabled);
COMMON_DECLARE_string(tensor_operants_mode);

namespace paddle::prim {

using Tensor = paddle::Tensor;
struct TestBaseProgram {
 public:
  const framework::ProgramDesc& main_program() { return program_; }

  std::string unique_name() { return "tmp_" + std::to_string(idx_++); }

  framework::VarDesc* lod_tensor(std::string name,
                                 std::vector<int64_t> shape = {},
                                 bool is_persistable = false,
                                 framework::proto::VarType::Type data_type =
                                     framework::proto::VarType::FP32) {
    auto* var = program_.MutableBlock(0)->Var(name);
    var->SetType(framework::proto::VarType::DENSE_TENSOR);
    var->SetDataType(data_type);
    var->SetShape(shape);
    var->SetPersistable(is_persistable);
    return var;
  }

  framework::VarDesc* unary_op(std::string type,
                               framework::VarDesc* x,
                               framework::VarDesc* out = nullptr,
                               const framework::AttributeMap* attrs = nullptr) {
    if (!out) {
      out = lod_tensor(unique_name());
    }
    framework::OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("X", {x->Name()});
    op->SetOutput("Out", {out->Name()});
    if (attrs) {
      for (auto& iter : *attrs) {
        op->SetAttr(iter.first, iter.second);
      }
    }
    op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(framework::OpRole::kForward));
    return out;
  }

  framework::VarDesc* tanh(framework::VarDesc* x,
                           framework::VarDesc* out = nullptr) {
    return unary_op("tanh", x, out);
  }

  framework::BlockDesc* GetBlock(std::size_t id) {
    return program_.MutableBlock(id);
  }

  void concat(std::vector<framework::VarDesc*> inputs,
              int axis,
              framework::VarDesc* out) {
    framework::OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("concat");
    std::vector<std::string> input_names(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_names[i] = inputs[i]->Name();
    }
    op->SetInput("X", input_names);
    op->SetOutput("Out", {out->Name()});
    op->SetAttr("axis", axis);
    op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(framework::OpRole::kForward));
  }

  void split(framework::VarDesc* input,
             int num,
             int axis,
             std::vector<framework::VarDesc*> outputs) {
    framework::OpDesc* op = program_.MutableBlock(0)->AppendOp();
    op->SetType("split");
    const std::string input_name = input->Name();
    std::vector<std::string> output_names(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      output_names[i] = outputs[i]->Name();
    }
    op->SetInput("X", {input_name});
    op->SetOutput("Out", output_names);
    op->SetAttr("num", num);
    op->SetAttr("axis", axis);
    op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
                static_cast<int>(framework::OpRole::kForward));
  }

 private:
  framework::ProgramDesc program_;
  int idx_{0};
};

class TestCompositeGradMaker : public CompositeGradOpMakerBase {
 public:
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;
  void Apply() override {}
};

TEST(StaticPrim, TanhBackwardComposite) {
  // Initialized environment
  FLAGS_tensor_operants_mode = "static";
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());

  TestBaseProgram base_program = TestBaseProgram();
  auto* target_block = base_program.GetBlock(0);
  // Prepare for forward tanh
  std::vector<int64_t> shape = {2, 2};
  StaticCompositeContext::Instance().SetBlock(target_block);
  Tensor x = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  Tensor out = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* x_desc =
      static_cast<prim::DescTensor*>(x.impl().get())->get_ptr();
  target_block->RenameVar(x_desc->Name(), "a");
  framework::VarDesc* out_desc =
      static_cast<prim::DescTensor*>(out.impl().get())->get_ptr();
  target_block->RenameVar(out_desc->Name(), "b");
  // TODO(jiabin): Grad out should be created by full, we can test it later
  base_program.tanh(target_block->FindVar("a"), target_block->FindVar("b"));

  ASSERT_EQ(target_block->AllOps().size(), static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Type(), "tanh");
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X")[0], "a");
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out").size(),
            std::size_t(1));
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out")[0], "b");
  ASSERT_EQ(target_block->AllVars().size(), static_cast<std::size_t>(2));
  ASSERT_EQ(target_block->AllVars()[0]->Name(), "a");
  ASSERT_EQ(target_block->AllVars()[1]->Name(), "b");
  auto* forward_opdesc = target_block->AllOps()[0];
  std::unordered_map<std::string, std::string> grad_to_var;
  std::vector<framework::BlockDesc*> grad_sub_block;
  Tensor out_grad = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* out_grad_desc =
      static_cast<prim::DescTensor*>(out_grad.impl().get())->get_ptr();
  target_block->RenameVar(out_grad_desc->Name(), "b@GRAD");
  std::vector<std::unique_ptr<framework::OpDesc>> grad_ops =
      framework::OpInfoMap::Instance()
          .Get(forward_opdesc->Type())
          .CompGradOpMaker()(*forward_opdesc,
                             std::unordered_set<std::string>(),
                             &grad_to_var,
                             target_block,
                             grad_sub_block);
  ASSERT_EQ(target_block->AllOps().size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops.size(), static_cast<std::size_t>(4));
  ASSERT_EQ(target_block->AllOps()[0]->Type(), "tanh");
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X")[0], "a");
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out")[0], "b");
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out")[0], "b");

  ASSERT_EQ(grad_ops[0]->Type(), "elementwise_mul");
  ASSERT_EQ(grad_ops[0]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[0]->Inputs().at("Y").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[0]->Inputs().at("Y")[0], "b");
  ASSERT_EQ(grad_ops[0]->Inputs().at("X")[0], "b");

  ASSERT_EQ(grad_ops[1]->Type(), "fill_constant");
  ASSERT_EQ(PADDLE_GET_CONST(int, grad_ops[1]->GetAttr("dtype")),
            static_cast<int>(5));  // ProtoDataType::FP32
  ASSERT_EQ(grad_ops[1]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));

  ASSERT_EQ(grad_ops[2]->Type(), "elementwise_sub");
  ASSERT_EQ(grad_ops[2]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[2]->Inputs().at("Y").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[2]->Inputs().at("X")[0],
            grad_ops[1]->Outputs().at("Out")[0]);
  ASSERT_EQ(grad_ops[2]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));

  ASSERT_EQ(grad_ops[3]->Type(), "elementwise_mul");
  ASSERT_EQ(grad_ops[3]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[3]->Inputs().at("Y").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[3]->Inputs().at("Y")[0],
            grad_ops[2]->Outputs().at("Out")[0]);
  ASSERT_EQ(grad_ops[3]->Inputs().at("X")[0], "b@GRAD");
  ASSERT_EQ(grad_ops[3]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));
}

TEST(StaticCompositeGradMaker, TestMultiInputMethod) {
  // Initialized environment
  FLAGS_tensor_operants_mode = "static";
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());

  TestBaseProgram base_program = TestBaseProgram();
  auto* target_block = base_program.GetBlock(0);
  std::vector<int64_t> shape = {2, 2};
  std::vector<int64_t> shape_out = {4, 2};
  StaticCompositeContext::Instance().SetBlock(target_block);
  Tensor x0 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  Tensor x1 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  Tensor out = prim::empty<prim::DescTensor>(
      shape_out, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* x0_desc =
      static_cast<prim::DescTensor*>(x0.impl().get())->get_ptr();
  target_block->RenameVar(x0_desc->Name(), "x0");
  framework::VarDesc* x1_desc =
      static_cast<prim::DescTensor*>(x1.impl().get())->get_ptr();
  target_block->RenameVar(x1_desc->Name(), "x1");
  framework::VarDesc* out_desc =
      static_cast<prim::DescTensor*>(out.impl().get())->get_ptr();
  target_block->RenameVar(out_desc->Name(), "out");
  std::vector<framework::VarDesc*> inputs = {target_block->FindVar("x0"),
                                             target_block->FindVar("x1")};
  framework::VarDesc* output = target_block->FindVar("out");
  base_program.concat(inputs, 0, output);
  auto* forward_opdesc = target_block->AllOps()[0];
  std::unordered_map<std::string, std::string> grad_to_var;
  std::vector<framework::BlockDesc*> grad_sub_block;
  Tensor out_grad = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* out_grad_desc =
      static_cast<prim::DescTensor*>(out_grad.impl().get())->get_ptr();
  target_block->RenameVar(out_grad_desc->Name(), "out@GRAD");
  auto test = TestCompositeGradMaker(*forward_opdesc,
                                     std::unordered_set<std::string>(),
                                     &grad_to_var,
                                     target_block,
                                     grad_sub_block);
  test();
  std::vector<paddle::Tensor> multi_fw_input = test.GetMultiForwardInput("X");
  paddle::optional<std::vector<paddle::Tensor>> opt_multi_fw_input =
      test.GetOptionalMultiForwardInput("X");
  std::vector<paddle::Tensor> opt_inner = opt_multi_fw_input.is_initialized()
                                              ? opt_multi_fw_input.get()
                                              : std::vector<paddle::Tensor>{};
  paddle::Tensor fw_out = test.GetSingleForwardOutput("Out");
  paddle::Tensor* fw_out_ptr = test.GetOutputPtr(&fw_out);
  std::string fw_out_name = test.GetOutputName(fw_out);

  ASSERT_EQ(multi_fw_input.size(), static_cast<std::size_t>(2));
  ASSERT_EQ(
      static_cast<prim::DescTensor*>(multi_fw_input[0].impl().get())->Name(),
      "x0");
  ASSERT_EQ(
      static_cast<prim::DescTensor*>(multi_fw_input[1].impl().get())->Name(),
      "x1");
  ASSERT_EQ(opt_inner.size(), static_cast<std::size_t>(2));
  ASSERT_EQ(static_cast<prim::DescTensor*>(opt_inner[0].impl().get())->Name(),
            "x0");
  ASSERT_EQ(static_cast<prim::DescTensor*>(opt_inner[1].impl().get())->Name(),
            "x1");
  ASSERT_EQ(&fw_out, fw_out_ptr);
  ASSERT_EQ(fw_out_name, "out");
}

TEST(StaticCompositeGradMaker, TestMultiOutputMethod) {
  // Initialized environment
  FLAGS_tensor_operants_mode = "static";
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());

  TestBaseProgram base_program = TestBaseProgram();
  auto* target_block = base_program.GetBlock(0);
  std::vector<int64_t> shape = {4, 2};
  std::vector<int64_t> shape_out = {2, 2};
  StaticCompositeContext::Instance().SetBlock(target_block);
  Tensor x = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  Tensor out1 = prim::empty<prim::DescTensor>(
      shape_out, phi::DataType::FLOAT32, paddle::Place());
  Tensor out2 = prim::empty<prim::DescTensor>(
      shape_out, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* x_desc =
      static_cast<prim::DescTensor*>(x.impl().get())->get_ptr();
  target_block->RenameVar(x_desc->Name(), "x");
  framework::VarDesc* out1_desc =
      static_cast<prim::DescTensor*>(out1.impl().get())->get_ptr();
  target_block->RenameVar(out1_desc->Name(), "out1");
  framework::VarDesc* out2_desc =
      static_cast<prim::DescTensor*>(out2.impl().get())->get_ptr();
  target_block->RenameVar(out2_desc->Name(), "out2");
  framework::VarDesc* input = target_block->FindVar("x");
  std::vector<framework::VarDesc*> outputs = {target_block->FindVar("out1"),
                                              target_block->FindVar("out2")};
  base_program.split(input, 2, 0, outputs);
  auto* forward_opdesc = target_block->AllOps()[0];
  std::unordered_map<std::string, std::string> grad_to_var;
  std::vector<framework::BlockDesc*> grad_sub_block;

  Tensor out1_grad = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* out1_grad_desc =
      static_cast<prim::DescTensor*>(out1_grad.impl().get())->get_ptr();
  target_block->RenameVar(out1_grad_desc->Name(), "out1@GRAD");

  Tensor out2_grad = prim::empty<prim::DescTensor>(
      shape, phi::DataType::FLOAT32, paddle::Place());
  framework::VarDesc* out2_grad_desc =
      static_cast<prim::DescTensor*>(out2_grad.impl().get())->get_ptr();
  target_block->RenameVar(out2_grad_desc->Name(), "out2@GRAD");

  auto test = TestCompositeGradMaker(*forward_opdesc,
                                     std::unordered_set<std::string>(),
                                     &grad_to_var,
                                     target_block,
                                     grad_sub_block);
  test();
  paddle::Tensor fw_input = test.GetSingleForwardInput("X");
  paddle::optional<paddle::Tensor> opt_fw_input =
      test.GetOptionalSingleForwardInput("X");
  std::vector<paddle::Tensor> fw_out = test.GetMultiForwardOutput("Out");
  std::vector<paddle::Tensor*> fw_out_ptr(fw_out.size());
  for (size_t i = 0; i < fw_out.size(); ++i) {
    fw_out_ptr[i] = &fw_out[i];
  }
  fw_out_ptr = test.GetOutputPtr(fw_out_ptr);
  std::vector<std::string> fw_out_name = test.GetOutputName(fw_out);
  ASSERT_EQ(static_cast<prim::DescTensor*>(fw_input.impl().get())->Name(), "x");
  ASSERT_EQ(static_cast<prim::DescTensor*>(opt_fw_input.get_ptr()->impl().get())
                ->Name(),
            "x");
  ASSERT_EQ(fw_out.size(), static_cast<std::size_t>(2));
  ASSERT_EQ(fw_out_ptr[0], &fw_out[0]);
  ASSERT_EQ(fw_out_ptr[1], &fw_out[1]);
  ASSERT_EQ(fw_out_name[0], "out1");
  ASSERT_EQ(fw_out_name[1], "out2");
}

TEST(StaticCompositeGradMaker, LogicalOperantsTest) {
  // Initialized environment
  FLAGS_tensor_operants_mode = "static";
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());

  TestBaseProgram base_program = TestBaseProgram();
  auto* target_block = base_program.GetBlock(0);
  std::vector<int64_t> shape = {2, 2};
  StaticCompositeContext::Instance().SetBlock(target_block);
  Tensor x0 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x0_name =
      std::static_pointer_cast<prim::DescTensor>(x0.impl())->Name();
  Tensor x1 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x1_name =
      std::static_pointer_cast<prim::DescTensor>(x1.impl())->Name();
  Tensor x2 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x2_name =
      std::static_pointer_cast<prim::DescTensor>(x2.impl())->Name();
  Tensor x3 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x3_name =
      std::static_pointer_cast<prim::DescTensor>(x3.impl())->Name();

  Tensor out_not = ~x0;
  Tensor out_and = out_not & x1;
  Tensor out_or = out_and | x2;
  Tensor out_xor = out_or ^ x3;

  ASSERT_EQ(target_block->AllOps().size(), static_cast<std::size_t>(4));
  ASSERT_EQ(target_block->AllOps()[0]->Type(), "bitwise_not");
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X")[0], x0_name);
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[1]->Type(), "bitwise_and");
  ASSERT_EQ(target_block->AllOps()[1]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[1]->Inputs().at("Y")[0], x1_name);
  ASSERT_EQ(target_block->AllOps()[1]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[2]->Type(), "bitwise_or");
  ASSERT_EQ(target_block->AllOps()[2]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[2]->Inputs().at("Y")[0], x2_name);
  ASSERT_EQ(target_block->AllOps()[2]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[3]->Type(), "bitwise_xor");
  ASSERT_EQ(target_block->AllOps()[3]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[3]->Inputs().at("Y")[0], x3_name);
  ASSERT_EQ(target_block->AllOps()[3]->Outputs().at("Out").size(),
            std::size_t(1));
}

TEST(StaticCompositeGradMaker, CompareOperantsTest) {
  // Initialized environment
  FLAGS_tensor_operants_mode = "static";
  paddle::OperantsManager::Instance().static_operants.reset(
      new paddle::prim::StaticTensorOperants());

  TestBaseProgram base_program = TestBaseProgram();
  auto* target_block = base_program.GetBlock(0);
  std::vector<int64_t> shape = {2, 2};
  StaticCompositeContext::Instance().SetBlock(target_block);
  Tensor x0 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x0_name =
      std::static_pointer_cast<prim::DescTensor>(x0.impl())->Name();
  Tensor x1 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x1_name =
      std::static_pointer_cast<prim::DescTensor>(x1.impl())->Name();
  Tensor x2 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x2_name =
      std::static_pointer_cast<prim::DescTensor>(x2.impl())->Name();
  Tensor x3 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x3_name =
      std::static_pointer_cast<prim::DescTensor>(x3.impl())->Name();
  Tensor x4 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x4_name =
      std::static_pointer_cast<prim::DescTensor>(x4.impl())->Name();
  Tensor x5 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x5_name =
      std::static_pointer_cast<prim::DescTensor>(x5.impl())->Name();
  Tensor x6 = prim::empty<prim::DescTensor>(
      shape, phi::DataType::INT32, phi::CPUPlace());
  std::string x6_name =
      std::static_pointer_cast<prim::DescTensor>(x6.impl())->Name();

  Tensor out_less = (x0 < x1);
  Tensor out_less_equal = (out_less <= x2);
  Tensor out_equal = (out_less_equal == x3);
  Tensor out_not_equal = (out_equal != x4);
  Tensor out_greater = (out_not_equal > x5);
  Tensor out_greater_equal = (out_greater >= x6);

  ASSERT_EQ(target_block->AllOps().size(), static_cast<std::size_t>(6));
  ASSERT_EQ(target_block->AllOps()[0]->Type(), "less_than");
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X")[0], x0_name);
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("Y")[0], x1_name);
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[1]->Type(), "less_equal");
  ASSERT_EQ(target_block->AllOps()[1]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[1]->Inputs().at("Y")[0], x2_name);
  ASSERT_EQ(target_block->AllOps()[1]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[2]->Type(), "equal");
  ASSERT_EQ(target_block->AllOps()[2]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[2]->Inputs().at("Y")[0], x3_name);
  ASSERT_EQ(target_block->AllOps()[2]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[3]->Type(), "not_equal");
  ASSERT_EQ(target_block->AllOps()[3]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[3]->Inputs().at("Y")[0], x4_name);
  ASSERT_EQ(target_block->AllOps()[3]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[4]->Type(), "greater_than");
  ASSERT_EQ(target_block->AllOps()[4]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[4]->Inputs().at("Y")[0], x5_name);
  ASSERT_EQ(target_block->AllOps()[4]->Outputs().at("Out").size(),
            std::size_t(1));

  ASSERT_EQ(target_block->AllOps()[5]->Type(), "greater_equal");
  ASSERT_EQ(target_block->AllOps()[5]->Inputs().at("Y").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[5]->Inputs().at("Y")[0], x6_name);
  ASSERT_EQ(target_block->AllOps()[5]->Outputs().at("Out").size(),
            std::size_t(1));
}

TEST(StaticPrim, TestFlags) {
  PrimCommonUtils::SetBwdPrimEnabled(true);
  ASSERT_TRUE(PrimCommonUtils::IsBwdPrimEnabled());
  PrimCommonUtils::SetBwdPrimEnabled(false);
  ASSERT_FALSE(PrimCommonUtils::IsBwdPrimEnabled());
}

}  // namespace paddle::prim
