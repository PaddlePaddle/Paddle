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
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/transform_desc.h"
#include "paddle/fluid/framework/variable.h"

#include "cinn/frontend/op_mappers/use_op_mappers.h"
#include "cinn/frontend/var_type_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ir::Node;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using OpMapperContext = CinnGraphSymbolization::OpMapperContext;
using CinnOpDesc = CinnGraphSymbolization::CinnOpDesc;
using FeedInfoMap = CinnGraphSymbolization::FeedInfoMap;

namespace utils {

OpMapperContext::FeedInfo GetCinnFeedInfoFromTensor(
    const Tensor& tensor, bool skip_trans_type = false) {
  OpMapperContext::FeedInfo info;
  const auto& dim = tensor.dims();
  for (int i = 0; i < dim.size(); i++) {
    info.shape.emplace_back(static_cast<int>(dim[i]));
  }

  // use FP32 as default type if skip_trans_type=true to pass CINN
  // enforce check that is shape and type of each input should be filled,
  // and we will ensure these feeds doesn't be used in execution on cinn_launch
  // op
  auto tensor_type = ::paddle::framework::proto::VarType::FP32;
  if (!skip_trans_type) {
    tensor_type = tensor.type();
  }
  auto cinn_var_type = TransformVarDataTypeToCinn(tensor_type);
  info.type = ::cinn::frontend::utils::CppVarType2CommonType(cinn_var_type);
  return info;
}
}  // namespace utils

FeedInfoMap CinnGraphSymbolization::GetFeedInfoMapFromInput() const {
  const std::unordered_set<std::string>* no_need_buffer_feeds = nullptr;
  if (graph_.Has(kNoNeedBufferFeeds)) {
    no_need_buffer_feeds =
        &graph_.Get<std::unordered_set<std::string>>(kNoNeedBufferFeeds);
  }

  FeedInfoMap feed_map;
  for (auto& feed_pair : input_tensors_) {
    const auto& feed_name = feed_pair.first;
    const auto* tensor = feed_pair.second;
    PADDLE_ENFORCE_NE(tensor, nullptr,
                      platform::errors::PreconditionNotMet(
                          "The input variable %s's tensor cannot be NULL,"
                          "we need the variable's dtype and shape from tensor.",
                          feed_name.c_str()));

    VLOG(4) << "Get feed info from input: " << feed_name;
    // if this feed declared as no need buffer then we can not access
    // its type so passing skip_trans_type=true
    if (no_need_buffer_feeds) {
      feed_map[feed_name] = utils::GetCinnFeedInfoFromTensor(
          *tensor, no_need_buffer_feeds->count(feed_name) > 0);
    } else {
      feed_map[feed_name] = utils::GetCinnFeedInfoFromTensor(*tensor);
    }

    PADDLE_ENFORCE_NE(
        feed_map[feed_name].shape.size(), 0UL,
        platform::errors::PreconditionNotMet(
            "The input variable %s's tensor shape cannot be empty,"
            "we need the variable's dtype and shape from tensor.",
            feed_name.c_str()));
  }
  return feed_map;
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
CinnGraphSymbolization::CreateCinnScope(const FeedInfoMap& feed_map) {
  auto cinn_scope = ::cinn::hlir::framework::Scope::Create();

  // get the graph's input parameter variable name list
  auto parameter_names = GetGraphInputParameterNames();

  for (const auto& param_name : parameter_names) {
    PADDLE_ENFORCE_GT(
        feed_map.count(param_name), 0UL,
        platform::errors::NotFound("Cannot find parameter %s from input list,"
                                   "please add the tensor into input.",
                                   param_name.c_str()));

    // if cannot find var in graph input, skip.
    // scope accepte the CINN format name, so here we need transform
    // paddle format name to CINN format.
    auto valid_name = ::cinn::utils::TransValidVarName(param_name);
    auto* cinn_var = cinn_scope->Var<CinnTensor>(valid_name);

    auto& cinn_tensor = absl::get<CinnTensor>(*cinn_var);
    // here we only need preserve dtype and shape, do not need preserve data
    auto feed_info = feed_map.at(param_name);
    cinn_tensor->set_type(feed_info.type);
    cinn_tensor->Resize(::cinn::hlir::framework::Shape(feed_info.shape));
    VLOG(4) << "add paddle param var [" << param_name
            << "] info cinn scope var[" << valid_name << "]";
    var_model_to_program_map_[param_name] = valid_name;
  }

  return cinn_scope;
}

std::vector<Node*> CinnGraphSymbolization::TopologicalSort() const {
  std::unordered_set<Node*> op_nodes;
  std::for_each(graph_.Nodes().begin(), graph_.Nodes().end(),
                [&op_nodes](Node* n) {
                  if (n->IsOp()) {
                    op_nodes.emplace(n);
                  }
                });

  std::unordered_map<Node*, std::unordered_map<Node*, size_t>> adj_list;
  std::unordered_map<Node*, size_t> in_degrees;
  for (auto* n : op_nodes) {
    // the op's input is var
    for (auto* in_var : n->inputs) {
      // the var's input is op
      for (auto* in_op : in_var->inputs) {
        if (op_nodes.count(in_op)) {
          ++adj_list[in_op][n];
          ++in_degrees[n];
        }
      }
    }
  }

  // find topology entries
  std::queue<Node*> queue;
  for (auto* n : op_nodes) {
    if (!in_degrees[n]) {
      queue.push(n);
    }
  }

  // topological sorting
  std::vector<Node*> sorted_ops;
  while (!queue.empty()) {
    auto* cur_op = queue.front();
    queue.pop();

    VLOG(4) << "topological sort insert: " << cur_op->Name() << " "
            << reinterpret_cast<void*>(cur_op) << " input "
            << cur_op->inputs.size();
    sorted_ops.emplace_back(cur_op);
    for (const auto& adj_pair : adj_list[cur_op]) {
      in_degrees.at(adj_pair.first) -= adj_pair.second;
      if (!in_degrees[adj_pair.first]) {
        queue.push(adj_pair.first);
      }
    }
  }

  PADDLE_ENFORCE_EQ(sorted_ops.size(), op_nodes.size(),
                    platform::errors::PreconditionNotMet(
                        "The sorting graph contains cycles."));
  return sorted_ops;
}

std::vector<std::unique_ptr<CinnOpDesc>>
CinnGraphSymbolization::TransformAllGraphOpToCinn() const {
  std::vector<std::unique_ptr<CinnOpDesc>> cinn_op_descs;

  auto sorted_ops = TopologicalSort();
  for (auto* node : sorted_ops) {
    cinn_op_descs.emplace_back(std::make_unique<CinnOpDesc>());
    auto& cinn_desc = cinn_op_descs.back();

    TransformOpDescToCinn(node->Op(), cinn_desc.get());
  }
  return cinn_op_descs;
}

void CinnGraphSymbolization::RunOp(const CinnOpDesc& op_desc,
                                   const OpMapperContext& ctx) const {
  const auto& op_type = op_desc.Type();
  auto* kernel = ::cinn::frontend::OpMapperRegistry::Global()->Find(op_type);
  PADDLE_ENFORCE_NE(kernel, nullptr,
                    platform::errors::NotFound(
                        "Op %s is Not Supported by CINN, please register"
                        " this op in the CINN repo.",
                        op_type.c_str()));
  VLOG(4) << "Running Op " << op_type;
  kernel->Run(op_desc, ctx);
}

void CinnGraphSymbolization::RunGraph(const OpMapperContext& ctx) const {
  auto cinn_op_descs = TransformAllGraphOpToCinn();
  // run the CINN op one by one, note that all ops
  // have been sorted at constructor.
  for (auto& op_desc : cinn_op_descs) {
    RunOp(*op_desc, ctx);
  }
}

std::unordered_set<std::string> CinnGraphSymbolization::GetFetchIds() const {
  std::unordered_set<std::string> fetch_names;
  fetch_names.reserve(fetch_var_names_.size());
  std::for_each(
      fetch_var_names_.begin(), fetch_var_names_.end(),
      [this, &fetch_names](const std::string& name) {
        PADDLE_ENFORCE_EQ(
            var_model_to_program_map_.count(name), 1,
            platform::errors::PreconditionNotMet(
                "Cannot find %s in var_model_to_program_map_", name.c_str()));
        fetch_names.insert(var_model_to_program_map_.at(name));
      });
  return fetch_names;
}

::cinn::frontend::Program CinnGraphSymbolization::operator()() {
  std::string builder_name = "NetBuilder_of_graph_" + std::to_string(graph_id_);
  VLOG(4) << "NetBuilder Name " << builder_name;

  ::cinn::frontend::NetBuilder builder(builder_name);

  auto feed_map = GetFeedInfoMapFromInput();
  auto cinn_scope = CreateCinnScope(feed_map);

  OpMapperContext ctx(*cinn_scope, target_, &builder, &var_map_,
                      &var_model_to_program_map_, &fetch_var_names_);
  // add all tensor's feed info into context
  for (auto& feed_pair : feed_map) {
    ctx.AddFeedInfo(feed_pair.first, feed_pair.second);
    VLOG(4) << "add feed var [" << feed_pair.first << "] info context";
  }
  RunGraph(ctx);

  return builder.Build();
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
