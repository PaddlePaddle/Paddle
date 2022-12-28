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
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/prim/api/manual/utils/utils.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace prim {

using Tensor = paddle::experimental::Tensor;
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
    var->SetType(framework::proto::VarType::LOD_TENSOR);
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

 private:
  framework::ProgramDesc program_;
  int idx_{0};
};

TEST(StaticPrim, TanhBackwardComposite) {
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
  std::vector<std::unique_ptr<framework::OpDesc>> grad_ops =
      std::move(framework::OpInfoMap::Instance()
                    .Get(forward_opdesc->Type())
                    .GradCompOpMaker()(*forward_opdesc,
                                       std::unordered_set<std::string>(),
                                       &grad_to_var,
                                       target_block,
                                       grad_sub_block));
  ASSERT_EQ(target_block->AllOps().size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops.size(), static_cast<std::size_t>(3));
  ASSERT_EQ(target_block->AllOps()[0]->Type(), "tanh");
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Inputs().at("X")[0], "a");
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out")[0], "b");
  ASSERT_EQ(target_block->AllOps()[0]->Outputs().at("Out")[0], "b");

  ASSERT_EQ(grad_ops[0]->Type(), "pow");
  ASSERT_EQ(grad_ops[0]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[0]->Inputs().at("X")[0], "b");
  ASSERT_EQ(PADDLE_GET_CONST(float, grad_ops[0]->GetAttr("factor")),
            static_cast<float>(2.0));
  ASSERT_EQ(grad_ops[0]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));

  ASSERT_EQ(grad_ops[1]->Type(), "scale");
  ASSERT_EQ(grad_ops[1]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[1]->Inputs().at("X")[0],
            grad_ops[0]->Outputs().at("Out")[0]);
  ASSERT_EQ(PADDLE_GET_CONST(float, grad_ops[1]->GetAttr("scale")),
            static_cast<float>(-1.0));
  ASSERT_EQ(PADDLE_GET_CONST(float, grad_ops[1]->GetAttr("bias")),
            static_cast<float>(1.0));
  ASSERT_EQ(grad_ops[1]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));

  ASSERT_EQ(grad_ops[2]->Type(), "elementwise_mul");
  ASSERT_EQ(grad_ops[2]->Inputs().at("X").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[2]->Inputs().at("Y").size(), static_cast<std::size_t>(1));
  ASSERT_EQ(grad_ops[2]->Inputs().at("Y")[0],
            grad_ops[1]->Outputs().at("Out")[0]);
  ASSERT_EQ(grad_ops[2]->Inputs().at("X")[0], "b@GRAD");
  ASSERT_EQ(grad_ops[2]->Outputs().at("Out").size(),
            static_cast<std::size_t>(1));
}
}  // namespace prim
}  // namespace paddle
