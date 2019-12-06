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

TEST(test_layer, test_debug_string) {
  platform::CPUPlace place;
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  var_pair in_pair = var_pair("X", vb_vector(1, vin));

  auto test_func = [&](std::shared_ptr<imperative::VarBase>& vout) {
    var_pair out_pair = var_pair("Out", vb_vector(1, vout));
    imperative::NameVarBaseMap ins = {in_pair};
    imperative::NameVarBaseMap outs = {out_pair};
    return LayerDebugString("test_op", ins, outs);
  };

  // 1. test null
  std::shared_ptr<imperative::VarBase> null_out(nullptr);
  std::string res_null = test_func(null_out);
  ASSERT_TRUE(res_null.find("NULL") != std::string::npos);

  // 2. test uninit var
  std::shared_ptr<imperative::VarBase> un_init_out(
      new imperative::VarBase(false, "un_init_out"));
  std::string res_un_init = test_func(un_init_out);
  ASSERT_TRUE(res_un_init.find("NOT_INITED_VAR") != std::string::npos);

  // 3. test unresolved type
  std::shared_ptr<imperative::VarBase> ut_out(
      new imperative::VarBase(false, "ut_out"));
  ut_out->MutableVar()->GetMutable<framework::LoDTensorArray>();
  std::string res_ut = test_func(ut_out);
  ASSERT_TRUE(res_ut.find("UNRESOLVED_TYPE") != std::string::npos);

  // 4. test uninit lod tensor
  std::shared_ptr<imperative::VarBase> lod_tensor(
      new imperative::VarBase(false, "lod_tensor"));
  auto tensor_l = lod_tensor->MutableVar()->GetMutable<framework::LoDTensor>();
  std::string res_ui_lod_t = test_func(lod_tensor);
  ASSERT_TRUE(res_ui_lod_t.find("NOT_INITED") != std::string::npos);

  // 5. test init lod tensor
  tensor_l->mutable_data<float>(place);
  std::string res_lod_t = test_func(lod_tensor);
  ASSERT_TRUE(res_lod_t.find("LoDTensor") != std::string::npos);

  // 6. test uninit selected rows
  std::shared_ptr<imperative::VarBase> selected_rows(
      new imperative::VarBase(false, "selected_rows"));
  auto tensor_sr = selected_rows->MutableVar()
                       ->GetMutable<framework::SelectedRows>()
                       ->mutable_value();
  std::string res_ui_sr = test_func(selected_rows);
  ASSERT_TRUE(res_ui_sr.find("NOT_INITED") != std::string::npos);

  // 7. test init selected rows
  tensor_sr->mutable_data<float>(place);
  std::string res_sr = test_func(selected_rows);
  ASSERT_TRUE(res_sr.find("SelectedRows") != std::string::npos);
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
  ASSERT_GT(op->GetInsMap().size(), 0UL);
  ASSERT_GT(op->GetOutsMap().size(), 0UL);
  ASSERT_GT(op->GradPendingOps().size(), 0UL);

  op->ClearBackwardTrace();

  ASSERT_EQ(op->GetInsMap().size(), 0UL);
  ASSERT_EQ(op->GetOutsMap().size(), 0UL);
  ASSERT_EQ(op->GradPendingOps().size(), 0UL);
}

TEST(test_layer, test_varbase_basic) {
  platform::CPUPlace place;
  std::shared_ptr<imperative::VarBase> vin(
      new imperative::VarBase(false, "vin"));
  vin->MutableVar()->GetMutable<framework::LoDTensor>()->mutable_data<float>(
      place);
  std::shared_ptr<imperative::VarBase> vout(vin->NewVarBase(place, false));
  ASSERT_EQ(vout->Name(), "vin0");

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

TEST(test_layer, test_dygraph_execution_context) {
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

  auto op = framework::OpRegistry::CreateOp("mul", {}, {}, {}, false);
  paddle::platform::CPUPlace cpu_place;

  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(cpu_place);
  paddle::framework::RuntimeContext ctx({}, {});
  framework::Scope scope;

  DygraphExecutionContext dy_exe_context(*(op.get()), scope, *dev_ctx, ctx,
                                         nullptr, ins, outs, &concat_att_map);

  ASSERT_EQ(dy_exe_context.InputSize("X"), 1u);
  ASSERT_EQ(dy_exe_context.InputName("X"), "vin");
  ASSERT_EQ(dy_exe_context.HasAttr("axis"), true);
  auto attr_map = dy_exe_context.Attrs();
  ASSERT_EQ(boost::get<int>(attr_map["axis"]), 1);
  ASSERT_EQ(dy_exe_context.OutputSize("Out"), 1u);
  ASSERT_EQ(dy_exe_context.HasOutput("Out"), true);
}

TEST(test_layer, test_dygraph_infershape_context) {
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

  DygraphInferShapeContext infer_shape_ctx(&ins, &outs, &concat_att_map);

  bool have_x = infer_shape_ctx.HasOutputs("Out");
  ASSERT_EQ(have_x, true);
  bool have_z = infer_shape_ctx.HasOutputs("Z");
  ASSERT_EQ(have_z, false);
}

}  // namespace imperative
}  // namespace paddle

USE_OP(mul);
