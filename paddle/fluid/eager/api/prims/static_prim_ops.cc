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

#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

#include "paddle/fluid/eager/api/prims/composite_grad_desc_maker.h"
#include "paddle/fluid/eager/api/prims/prim_api.h"
#include "paddle/fluid/eager/api/prims/utils.h"
namespace paddle {
namespace prims {

template <>
void Pow(const framework::VarDesc& X,
         const paddle::optional<framework::VarDesc>& FactorTensor,
         float factor,
         framework::VarDesc* Out) {
  if (!Out) {
    Out = CreateVar<framework::VarDesc>("Out");
  }
  framework::OpDesc* op =
      StaticCompositeContext::Instance().GetBlock()->AppendOp();
  op->SetType("pow");
  op->SetInput("X", {X.Name()});
  if (FactorTensor.get_ptr()) {
    op->SetInput("FactorTensor", {FactorTensor->Name()});
  }
  op->SetOutput("Out", {Out->Name()});
  op->SetAttr("factor", factor);
  op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(framework::OpRole::kForward));
}

template <>
void Scale(const framework::VarDesc& X,
           const paddle::optional<framework::VarDesc>& ScaleTensor,
           float scale,
           float bias,
           bool bias_after_scale,
           framework::VarDesc* Out) {
  if (!Out) {
    Out = CreateVar<framework::VarDesc>("Out");
  }
  framework::OpDesc* op =
      StaticCompositeContext::Instance().GetBlock()->AppendOp();
  op->SetType("scale");
  op->SetInput("X", {X.Name()});
  if (ScaleTensor.get_ptr()) {
    op->SetInput("ScaleTensor", {ScaleTensor->Name()});
  }
  op->SetOutput("Out", {Out->Name()});
  op->SetAttr("scale", scale);
  op->SetAttr("bias", bias);
  op->SetAttr("bias_after_scale", bias_after_scale);
  op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(framework::OpRole::kForward));
}

template <>
void Mul(const paddle::framework::VarDesc& X,
         const paddle::framework::VarDesc& Y,
         paddle::framework::VarDesc* Out) {
  if (!Out) {
    Out = CreateVar<framework::VarDesc>("Out");
  }
  framework::OpDesc* op =
      StaticCompositeContext::Instance().GetBlock()->AppendOp();
  op->SetType("elementwise_mul");
  op->SetInput("X", {X.Name()});
  op->SetInput("Y", {Y.Name()});
  op->SetOutput("Out", {Out->Name()});
  op->SetAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(framework::OpRole::kForward));
}

}  // namespace prims
}  // namespace paddle
