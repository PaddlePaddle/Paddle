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

#include "paddle/framework/var_type_inference.h"
#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"

namespace paddle {
namespace framework {

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  SumOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(const OpDescBind &op_desc,
                  BlockDescBind *block) const override {
    auto &inputs = op_desc.Input("X");
    auto default_var_type = VarDesc::SELECTED_ROWS;

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [block](const std::string &name) {
          return block->Var(name)->GetType() == VarDesc::LOD_TENSOR;
        });
    if (any_input_is_lod_tensor) {
      default_var_type = VarDesc::LOD_TENSOR;
    }

    auto out_var_name = op_desc.Output("Out").front();
    block->Var(out_var_name)->SetType(default_var_type);
  }
};
}  // namespace framework
}  // namespace paddle

REGISTER_OPERATOR(sum, paddle::framework::NOP, paddle::framework::SumOpMaker,
                  paddle::framework::SumOpVarTypeInference);
REGISTER_OPERATOR(sum_without_infer_var_type, paddle::framework::NOP,
                  paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {

TEST(InferVarType, sum_op) {
  ProgramDescBind prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});

  prog.MutableBlock(0)->Var("test_a")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(VarDesc::SELECTED_ROWS,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  prog.MutableBlock(0)->Var("test_b")->SetType(VarDesc::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(VarDesc::LOD_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());
}

TEST(InferVarType, sum_op_without_infer_var_type) {
  ProgramDescBind prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum_without_infer_var_type");
  op->SetInput("X", {"test2_a", "test2_b", "test2_c"});
  op->SetOutput("Out", {"test2_out"});

  prog.MutableBlock(0)->Var("test2_a")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_b")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_c")->SetType(VarDesc::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(VarDesc_VarType_LOD_TENSOR,
            prog.MutableBlock(0)->Var("test2_out")->GetType());
}

}  // namespace framework
}  // namespace paddle