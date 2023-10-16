// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/paddle_model_convertor.h"

#include <glog/logging.h>

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "paddle/cinn/frontend/op_mappers/use_op_mappers.h"
#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"
#include "paddle/cinn/frontend/paddle/cpp/program_desc.h"
#include "paddle/cinn/frontend/paddle/model_parser.h"
#include "paddle/cinn/frontend/var_type_utils.h"
#include "paddle/cinn/hlir/op/use_ops.h"

PD_DECLARE_double(cinn_infer_model_version);

namespace cinn {
namespace frontend {

using cinn::utils::Attribute;

PaddleModelConvertor::PaddleModelConvertor()
    : PaddleModelConvertor(common::DefaultTarget(), nullptr, nullptr) {}

PaddleModelConvertor::PaddleModelConvertor(
    const common::Target& target,
    std::shared_ptr<NetBuilder> builder,
    std::shared_ptr<hlir::framework::Scope> scope)
    : target_(target), builder_(builder), scope_(scope) {
  if (!builder_) {
    // do not need scope
    builder_ =
        std::make_shared<NetBuilder>(cinn::UniqName("PaddleModelConvertor"));
  }
  if (!scope_) {
    // do not need scope
    scope_ = hlir::framework::Scope::Create();
  }
  ctx_ = std::make_unique<OpMapperContext>(*scope_,
                                           target_,
                                           builder_.get(),
                                           &var_map_,
                                           &var_model_to_program_map_,
                                           &fetch_var_names_);
}

void PaddleModelConvertor::PrepareRun(const paddle::cpp::BlockDesc& block_desc,
                                      OpMapperContext* ctx) {
  std::unordered_map<std::string, const paddle::cpp::VarDesc*> var_desc_map;
  // preserve var desc info lik shape and dtype
  for (int i = 0; i < block_desc.VarsSize(); i++) {
    const auto& var_desc = block_desc.GetConstVar<paddle::cpp::VarDesc>(i);
    var_desc_map[var_desc.Name()] = &var_desc;
  }

  for (int i = 0; i < block_desc.OpsSize(); i++) {
    const auto& op_desc = block_desc.GetConstOp<paddle::cpp::OpDesc>(i);

    if (op_desc.Type() == "feed") {
      for (const auto& var_name : op_desc.output_vars()) {
        CHECK(var_desc_map.count(var_name))
            << "Feed var [" << var_name << "] Not found in block";
        ctx->AddFeedInfo(var_name,
                         utils::GetFeedInfoFromDesc(*var_desc_map[var_name]));
      }
    }
  }
}

void PaddleModelConvertor::RunOp(const paddle::cpp::OpDesc& op_desc,
                                 const OpMapperContext& ctx) {
  const auto& op_type = op_desc.Type();
  auto kernel = OpMapperRegistry::Global()->Find(op_type);
  CHECK(kernel) << "Op [" << op_type << "] Not supported in OpMapper";
  VLOG(4) << "Running Op " << op_type;
  kernel->Run(op_desc, ctx);
}

std::unordered_map<std::string, Variable> PaddleModelConvertor::GetFetchList(
    const std::unordered_set<std::string>& fetch_name_list) const {
  // the return map's key is paddle variable name, the value is the cinn fetch
  // variable
  const std::unordered_set<std::string>* var_name_list = &fetch_name_list;
  if (fetch_name_list.empty()) {
    // if paddle var list is empty, fetch the program's fetch var instead
    CHECK(!fetch_var_names_.empty())
        << "Should not fetch empty variable in CINN.";
    var_name_list = &fetch_var_names_;
  }

  std::unordered_map<std::string, Variable> fetch_list;
  fetch_list.reserve(var_name_list->size());
  for (const auto& pd_name : *var_name_list) {
    CHECK(var_model_to_program_map_.count(pd_name))
        << "Cannot find cinn variable [" << pd_name
        << "] in var_model_to_program_map_";
    auto norm_pd_name = pd_name;
    // remove inplace output's suffix
    auto pos = pd_name.find(paddle::InplaceOutSuffix);
    if (pos != std::string::npos) {
      norm_pd_name.replace(pos, sizeof(paddle::InplaceOutSuffix), "");
    }
    fetch_list[pd_name] = var_map_.at(norm_pd_name);
  }
  return fetch_list;
}

Program PaddleModelConvertor::LoadModel(
    const std::string& model_dir,
    bool is_combined,
    const std::unordered_map<std::string, std::vector<int64_t>>& feed) {
  paddle::cpp::ProgramDesc program_desc;
  if (FLAGS_cinn_infer_model_version < 2.0) {
    paddle::LoadModelPb(model_dir,
                        "/__model__",
                        "/params",
                        scope_.get(),
                        &program_desc,
                        is_combined,
                        false,
                        target_);
  } else {
    paddle::LoadModelPb(model_dir,
                        ".pdmodel",
                        ".pdiparams",
                        scope_.get(),
                        &program_desc,
                        is_combined,
                        false,
                        target_);
  }
  CHECK_EQ(program_desc.BlocksSize(), 1)
      << "CINN can only support the model with a single block";
  auto* block_desc = program_desc.GetBlock<paddle::cpp::BlockDesc>(0);

  // Set feeds shape
  for (int i = 0; i < block_desc->VarsSize(); i++) {
    auto* var_desc = block_desc->GetVar<paddle::cpp::VarDesc>(i);
    const auto var_name = var_desc->Name();
    if (feed.count(var_name)) {
      const auto& var_shape = feed.at(var_name);
      VLOG(4) << "Update var " << var_name
              << "'s shape to: " << cinn::utils::Join(var_shape, ", ");
      var_desc->SetShape(var_shape);
    }
  }

  OpMapperContext ctx(*scope_,
                      target_,
                      builder_.get(),
                      &var_map_,
                      &var_model_to_program_map_,
                      &fetch_var_names_);

  PrepareRun(*block_desc, &ctx);
  for (int i = 0; i < block_desc->OpsSize(); i++) {
    auto* op_desc = block_desc->GetOp<paddle::cpp::OpDesc>(i);
    RunOp(*op_desc, ctx);
  }
  return builder_->Build();
}

void SetOpDescAttr(const std::string& attr_name,
                   const Attribute& attr_value,
                   paddle::cpp::OpDesc* op_desc) {
  class Visitor {
   public:
    Visitor(paddle::cpp::OpDesc* op_desc, const std::string& attr_name)
        : op_desc_(op_desc), attr_name_(attr_name) {}

#define VISITOR_EXPAND(TYPE) \
  void operator()(const TYPE& v) { op_desc_->SetAttr(attr_name_, v); }

    VISITOR_EXPAND(bool)
    VISITOR_EXPAND(float)
    VISITOR_EXPAND(int)
    VISITOR_EXPAND(std::string)
    VISITOR_EXPAND(std::vector<bool>)
    VISITOR_EXPAND(std::vector<int>)
    VISITOR_EXPAND(std::vector<float>)
    VISITOR_EXPAND(std::vector<std::string>)
    VISITOR_EXPAND(int64_t)
    VISITOR_EXPAND(double)
    VISITOR_EXPAND(std::vector<int64_t>)
    VISITOR_EXPAND(std::vector<double>)
#undef VISITOR_EXPAND

   private:
    paddle::cpp::OpDesc* op_desc_;
    const std::string& attr_name_;
  };
  absl::visit(Visitor{op_desc, attr_name}, attr_value);
}

void PaddleModelConvertor::RunOp(
    const std::string& op_type,
    const std::map<std::string, std::vector<std::string>>& inputs,
    const std::map<std::string, std::vector<std::string>>& outputs,
    const std::map<std::string, Attribute>& attrs,
    const OpMapperContext& ctx) {
  paddle::cpp::OpDesc op_desc;
  op_desc.SetType(op_type);
  for (const auto& in_pair : inputs) {
    op_desc.SetInput(in_pair.first, in_pair.second);
  }
  for (const auto& out_pair : outputs) {
    op_desc.SetOutput(out_pair.first, out_pair.second);
  }
  for (const auto& attr_pair : attrs) {
    SetOpDescAttr(attr_pair.first, attr_pair.second, &op_desc);
  }

  RunOp(op_desc, ctx);
}

void PaddleModelConvertor::RunOp(
    const std::string& op_type,
    const std::map<std::string, std::vector<std::string>>& inputs,
    const std::map<std::string, std::vector<std::string>>& outputs,
    const std::map<std::string, Attribute>& attrs) {
  RunOp(op_type, inputs, outputs, attrs, *ctx_);
}

Program PaddleModelConvertor::operator()() { return builder_->Build(); }

void PaddleModelConvertor::CreateInput(const std::string& dtype,
                                       const cinn::utils::ShapeType& shape,
                                       const std::string& name) {
  OpMapperContext::FeedInfo feed_info = {shape, common::Str2Type(dtype)};

  ctx_->AddFeedInfo(name, feed_info);
  RunOp("feed", {}, {{"Out", {name}}}, {});
}

}  // namespace frontend
}  // namespace cinn
