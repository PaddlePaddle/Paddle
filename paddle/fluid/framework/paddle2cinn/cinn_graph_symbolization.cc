/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"

#include <algorithm>
#include <iterator>
#include <queue>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/paddle2cinn/transform_desc.h"
#include "paddle/fluid/framework/variable.h"

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/var_type_utils.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ir::Node;
using CinnTensor = ::cinn::hlir::framework::Tensor;

namespace utils {

OpMapperContext::FeedInfo GetCinnFeedInfoFromTensor(const Tensor& tensor) {
  OpMapperContext::FeedInfo info;
  const auto& dim = tensor.dims();
  for (int i = 0; i < dim.size(); i++) {
    info.shape.emplace_back(sttic_cast<int>(dim[i]));
  }

  auto cinn_var_type = TransformVarTypeToCinn(tensor.type());
  info.type = ::cinn::frontend::utils::CppVarType2CommonType(cinn_var_type);
  return info;
}

void TransformPaddleVariableToCinn(
    const Variable& pd_var, ::cinn::hlir::framework::Variable* cinn_var) {
  const auto& pd_tensor = pd_var.Get<Tensor>();
  auto& cinn_tensor = absl::get<CinnTensor>(*cinn_var);

  auto feed_info = GetCinnFeedInfoFromTensor(pd_tensor);
  // here we only need preserve dtype and shape, do not need preserve data
  cinn_tensor.set_type(feed_info.type);
  cinn_tensor.Resize(::cinn::hlir::framework::Shape(feed_info.shape));
}
}  // namespace utils

void CinnGraphSymbolization::AddFeedInfoIntoContext(
    OpMapperContext* ctx) const {
  for (auto& feed_pair : feed_targets_) {
    const auto& feed_name = feed_pair.first;
    const auto* tensor = feed_pair.second;

    ctx.AddFeedInfo(feed_name, utils::GetCinnFeedInfoFromTensor(*tensor));
  }
}

// get the graph's op input Parameter var name set
std::unordered_set<std::string>
CinnGraphSymbolization::GetGraphInputParameterNames() const {
  std::unordered_set<std::string> names;

  for (auto* node : graph_.Nodes()) {
    if (node->IsOp()) {
      for (auto* var : node->inputs) {
        if (var->Var()->IsParameter()) {
          // Only need preserve the input parameter var of graph,
          // others do not.
          names.insert(var->Name());
        }
      }
    }
  }

  return names;
}

// Transform paddle scope to cinn, note that we only preserve the graph’s
// input parameter variable and ignore others.
std::shared_ptr<::cinn::hlir::framework::Scope>
CinnGraphSymbolization::TransformPaddleScopeToCinn() const {
  auto cinn_scope = ::cinn::hlir::framework::Scope::Create();

  // get the graph's input parameter variable name list
  auto parameter_names = GetGraphInputParameterNames();

  for (const auto& var_name : scope_.LocalVarNames()) {
    // if cannot find var in graph input, skip
    if (parameter_names.count(var_name) == 0) continue;

    auto* pd_var = scope_.FindLocalVar(var_name);

    // scope accepte the CINN format name, so here we need transform
    // paddle format name to CINN format.
    auto* cinn_var = cinn_scope->Var<CinnTensor>(
        ::cinn::utils::TransValidVarName(var.name()));

    utils::TransformPaddleVariableToCinn(*pd_var, cinn_var);
  }

  return cinn_scope;
}

std::vector<std::unique_ptr<CinnOpDesc>>
CinnGraphSymbolization::TransformAllGraphOpToCinn() const {
  std::vector<std::unique_ptr<CinnOpDesc>> cinn_op_descs_;

  const auto& sorted_ops = ir::TopologySortOperations(graph_);
  for (auto* node : sorted_ops) {
    cinn_op_descs_.emplace_back(std::make_unique<CinnOpDesc>());
    auto& cinn_desc = cinn_op_descs_.back();

    TransformOpDescToCinn(node->Op(), cinn_desc.get());
  }
  return cinn_op_descs_;
}

void CinnGraphSymbolization::RunOp(const CinnOpDesc& op_desc,
                                   const OpMapperContext& ctx) const {
  const auto& op_type = op_desc.Type();
  auto kernel = ::cinn::frontend::OpMapperRegistry::Global()->Find(op_type);
  PADDLE_ENFORCE_NE(
      kernel, nullptr,
      platform::errors::NotFound("Op %s Not Support by CINN", op_type.c_str()));
  VLOG(4) << "Running Op " << op_type;
  kernel->Run(op_desc, ctx);
}

void CinnGraphSymbolization::RunGraph(const OpMapperContext& ctx) const {
  auto cinn_op_descs_ = TransformAllGraphOpToCinn();
  // run the CINN op one by one, note that all ops
  // have been sorted at constructor.
  for (auto* op_desc : cinn_op_descs_) {
    RunOp(*op_desc, ctx);
  }
}

::cinn::frontend::Program CinnGraphSymbolization::operator()() const {
  std::string builder_name = "NetBuilder_of_graph_" + std::to_string(graph_id_);
  VLOG(4) << "NetBuilder Name " << builder_name;

  ::cinn::frontend::NetBuilder builder(builder_name);

  auto cinn_scope = TransformPaddleScopeToCinn();

  OpMapperContext ctx(*cinn_scope, target_, &builder, &var_map_,
                      &var_model_to_program_map_);

  AddFeedInfoIntoContext(&ctx);
  RunGraph(ctx);

  return builder.Build();
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
