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

#include "paddle/fluid/framework/var_type_inference.h"
#include <string>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "");
    AddComment("");
  }
};

class SumOpVarTypeInference : public VarTypeInference {
 public:
  void operator()(const OpDesc &op_desc, BlockDesc *block) const override {
    auto &inputs = op_desc.Input("X");
    auto default_var_type = proto::VarType::SELECTED_ROWS;

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [block](const std::string &name) {
          return block->Var(name)->GetType() == proto::VarType::LOD_TENSOR;
        });
    if (any_input_is_lod_tensor) {
      default_var_type = proto::VarType::LOD_TENSOR;
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
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType::SELECTED_ROWS,
            prog.MutableBlock(0)->Var("test_out")->GetType());

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));
  ASSERT_EQ(proto::VarType::LOD_TENSOR,
            prog.MutableBlock(0)->Var("test_out")->GetType());
}

TEST(InferVarType, sum_op_without_infer_var_type) {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum_without_infer_var_type");
  op->SetInput("X", {"test2_a", "test2_b", "test2_c"});
  op->SetOutput("Out", {"test2_out"});

  prog.MutableBlock(0)->Var("test2_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test2_out");

  op->InferVarType(prog.MutableBlock(0));

  ASSERT_EQ(proto::VarType_Type_LOD_TENSOR,
            prog.MutableBlock(0)->Var("test2_out")->GetType());
}

}  // namespace framework
}  // namespace paddle
