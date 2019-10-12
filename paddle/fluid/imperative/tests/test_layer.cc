// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

//
// Created by Jiabin on 2019-08-16.
//

#include <paddle/fluid/framework/op_registry.h>
#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "paddle/fluid/imperative/layer.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;
namespace framework = paddle::framework;

namespace paddle {
namespace imperative {

using vb_vector = std::vector<std::shared_ptr<imperative::VarBase>>;

using var_pair = std::pair<std::string, vb_vector>;

TEST(test_layer, test_runtime_context) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  var_pair in_pair = var_pair("X", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {in_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap attrs;
  auto* ctx = new imperative::RuntimeInferVarTypeContext(ins, &outs, attrs);
  ASSERT_TRUE(ctx->HasVar("vin"));
  ASSERT_TRUE(ctx->HasInput("X"));
  ASSERT_TRUE(ctx->HasOutput("Out"));

  ASSERT_ANY_THROW(ctx->GetDataTypes("vin"));
  std::vector<framework::proto::VarType::Type> NullType;
  ASSERT_ANY_THROW(ctx->SetDataTypes("vin", NullType));
  ASSERT_ANY_THROW(ctx->GetShape("vin"));
  ASSERT_ANY_THROW(ctx->GetLoDLevel("vin"));
  ASSERT_ANY_THROW(ctx->SetLoDLevel("vin", 2));
}

std::string LayerDebugString(const std::string& op_type,
                             const NameVarBaseMap& ins,
                             const NameVarBaseMap& outs);

TEST(test_layer, test_debug_string_test_debug_Test) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vin_error(
      new imperative::VarBase(false, "vin_error"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  std::shared_ptr<imperative::VarBase> vout_error(
      new imperative::VarBase(false, "vout_error"));
  vin_error->MutableVar()->GetMutable<framework::LoDTensor>();
  vout->MutableVar()->GetMutable<framework::LoDTensor>();
  vout_error->MutableVar()->GetMutable<framework::SelectedRows>();
  var_pair in_pair = var_pair("X", vb_vector(1, vin));
  vb_vector vb_in_error = {vin_error, nullptr};
  var_pair vin_error_pair = var_pair("X", vb_in_error);
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  var_pair vout_error_pair = var_pair("Out2", vb_vector(1, vout_error));
  imperative::NameVarBaseMap ins = {in_pair};
  imperative::NameVarBaseMap ins_error = {vin_error_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  imperative::NameVarBaseMap outs_error = {vout_error_pair};
  ASSERT_NO_FATAL_FAILURE(LayerDebugString("test_op", ins, outs));
  std::string res = LayerDebugString("test_op", ins, outs_error);
  ASSERT_TRUE(res.find("UNRESOLVED_TYPE") != std::string::npos);
  std::string res2 = LayerDebugString("test_op", ins_error, outs_error);
  VLOG(3) << res2;
  ASSERT_TRUE(res2.find("NOT_INITED") != std::string::npos);
  ASSERT_TRUE(res2.find("NULL") != std::string::npos);
}

TEST(test_layer, test_clear_backward_info) {
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  std::shared_ptr<imperative::VarBase> vout(
      new imperative::VarBase(false, "vout"));
  framework::OpDesc desc;
  platform::CPUPlace place;
  var_pair x_pair = var_pair("X", vb_vector(1, vin));
  var_pair y_pair = var_pair("Y", vb_vector(1, vin));
  var_pair out_pair = var_pair("Out", vb_vector(1, vout));
  imperative::NameVarBaseMap ins = {x_pair, y_pair};
  imperative::NameVarBaseMap outs = {out_pair};
  framework::AttributeMap concat_att_map;
  concat_att_map["axis"] = 1;
  std::shared_ptr<imperative::OpBase> op(
      OpBase::Create(0, "mul", ins, outs, concat_att_map, place));
  std::shared_ptr<imperative::OpBase> preceding_op(
      OpBase::Create(0, "mul", ins, outs, concat_att_map, place));
  op->InsertGradPendingOps(preceding_op.get());
  *(op->GetMutableInsMap()) = ins;
  *(op->GetMutableOutsMap()) = outs;
  ASSERT_GT(op->GetInsMap().size(), 0);
  ASSERT_GT(op->GetOutsMap().size(), 0);
  ASSERT_GT(op->GradPendingOps().size(), 0);

  op->ClearBackwardTrace();

  ASSERT_EQ(op->GetInsMap().size(), 0);
  ASSERT_EQ(op->GetOutsMap().size(), 0);
  ASSERT_EQ(op->GradPendingOps().size(), 0);
}

TEST(test_layer, test_varbase_basic) {
  platform::CPUPlace place;
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  vin->MutableVar()->GetMutable<framework::LoDTensor>()->mutable_data<float>(
      place);
  std::shared_ptr<imperative::VarBase> vout(vin->NewVarBase(place, false));
  ASSERT_EQ(vout->Name(), "Itmp0");

  std::shared_ptr<imperative::VarBase> vin_with_grad(
      new imperative::VarBase(true, "vin"));
  ASSERT_ANY_THROW(vin->MutableGradVar());
  ASSERT_NO_THROW(ASSERT_TRUE(dynamic_cast<framework::Variable*>(
                                  vin_with_grad->MutableGradVar()) != 0));
  ASSERT_TRUE(
      dynamic_cast<framework::Variable*>(vin_with_grad->MutableGradVar()) != 0);
  vin_with_grad->SetOverridedStopGradient(false);
  ASSERT_FALSE(vin_with_grad->OverridedStopGradient());
  ASSERT_NO_FATAL_FAILURE(vin_with_grad->SetPersistable(true));
  ASSERT_FALSE(vin_with_grad->OverridedStopGradient());
  ASSERT_NO_FATAL_FAILURE(vin_with_grad->SetName("new_name"));
  ASSERT_EQ(vin_with_grad->Name(), "new_name");
}
// TODO(jiabin): Add more ut here for layer

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
