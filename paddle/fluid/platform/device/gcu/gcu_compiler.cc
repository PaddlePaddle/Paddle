/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device/gcu/gcu_compiler.h"

#include <stdlib.h>

#include <algorithm>
#include <mutex>  // NOLINT
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dtu/hlir/dispatch.h"
#include "dtu/hlir/types.h"
#include "paddle/fluid/platform/device/gcu/common/op_common.h"
#include "paddle/fluid/platform/device/gcu/compiler/tops_compiler.h"
#include "paddle/fluid/platform/device/gcu/equivalence_trans/all_ops.h"
#include "paddle/fluid/platform/device/gcu/gcu_backend.h"
#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_utils.h"
#include "paddle/phi/core/tensor_utils.h"

namespace paddle {
namespace platform {
namespace gcu {

namespace {
const size_t kGlobalGraphID = 0;
const size_t kFpBpGraphID = 1;
const size_t kUpdateGraphID = 2;

static std::set<std::string> kUnusedArchetype = {"ReserveSpace"};

// for dyn such as interpolate op
static std::map<std::string, std::vector<int64_t>> g_map_var_fixed_shape;

GraphType IdentityGraphType(const Graph*& graph) {  // NOLINT
  if (graph->Has(kGraphType)) {
    auto type = graph->Get<int>(kGraphType);
    return type == 0 ? FP : BP;
  }
  return OTHER;
}

bool IsDyn(const std::vector<int64_t>& shape) {
  return std::any_of(
      shape.begin(), shape.end(), [](const int64_t& dim) { return dim < 0; });
}

bool IsLogicalOp(const std::string& op_type) {
  const std::set<std::string> logical_ops = {
      "logical_or", "logical_xor", "logical_not", "logical_and"};
  return logical_ops.count(op_type) != 0;
}

// funs for debug
std::string StringVectorDebugStr(const std::vector<std::string>& strs,
                                 const std::string& debug_info = "") {
  std::ostringstream os;
  os << "debug info:" << debug_info << "\n";
  for (const auto& str : strs) {
    os << "    value:" << str << "\n";
  }
  return os.str();
}

std::string StringMapDebugStr(
    const std::unordered_map<std::string, std::string>& str_map,
    const std::string& debug_info = "") {
  std::ostringstream os;
  os << "debug info:" << debug_info << "\n";
  for (const auto& str_pair : str_map) {
    os << "    key:" << str_pair.first << ", value:" << str_pair.second << "\n";
  }
  return os.str();
}

const char* const kPlaceHolder = " ";
std::vector<std::string> ParseAttr(std::string attr_value) {
  std::vector<std::string> out_var_names;
  if (attr_value == "") return out_var_names;
  const char* divided_symbol = ";";
  size_t pos = attr_value.find(divided_symbol);
  if (pos == attr_value.npos) {
    out_var_names.emplace_back(attr_value);
  }
  while (pos != attr_value.npos) {
    std::string sub_str = attr_value.substr(0, pos);
    out_var_names.emplace_back(sub_str);
    attr_value = attr_value.substr(pos + 1, attr_value.size());
    pos = attr_value.find(divided_symbol);
  }
  if (attr_value.length() != 0) {
    out_var_names.emplace_back(attr_value);
  }
  return out_var_names;
}

std::string GetShapeStr(std::vector<int64_t> shape) {
  std::stringstream ss;
  ss << "[";
  for (const auto& dim : shape) {
    ss << dim << ",";
  }
  ss << "]";
  return ss.str();
}
}  // namespace

std::map<std::string, std::vector<int64_t>> GcuCompiler::GetVarFixedMap() {
  return g_map_var_fixed_shape;
}

void GcuCompiler::Reset() {
  feed_list_.clear();
  var_node_cache_.clear();
  gcu_node_cache_.clear();
  reflection_.Clear();
  weights_name_to_gcuop_.clear();
  var_name_to_input_.clear();
  leaf_var_nodes_.clear();
  // is_training_graph_ = false;
  counter_ = 0;
}

bool GcuCompiler::IsRefNode(const Node* node) {
  bool is_ref_node = false;
  auto op_desc = node->Op();
  std::set<std::string> names_in;
  for (const auto& e : op_desc->Inputs()) {
    if (e.second.empty()) {
      continue;
    }
    for (std::string n : e.second) {
      names_in.insert(n);
    }
  }
  for (const auto& e : op_desc->Outputs()) {
    if (e.second.empty()) {
      continue;
    }
    for (std::string n : e.second) {
      if (names_in.count(n) > 0) {
        is_ref_node = true;
        break;
      }
    }
    if (is_ref_node) {
      break;
    }
  }
  return is_ref_node;
}

namespace {
std::string GetVarKey(const std::string& name,
                      size_t graph_id,
                      const std::string& role) {
  return name + "_graph_" + std::to_string(graph_id) + "_" + role;
}

void UpdateSymbolForVar(const std::string& var_key,
                        const std::string& new_symbol,
                        GlobalMemoryRef& gm_ref) {  // NOLINT
  PADDLE_ENFORCE_NE(
      gm_ref.var_to_symbol.count(var_key),
      0,
      platform::errors::NotFound("var_key %s is not found in var_to_symbol",
                                 var_key.c_str()));
  std::string old_symbol = gm_ref.var_to_symbol[var_key];
  gm_ref.var_to_symbol[var_key] = new_symbol;
  gm_ref.symbol_to_vars[old_symbol].erase(var_key);
  gm_ref.symbol_to_vars[new_symbol].emplace(var_key);
}

void GenerateInOutKeys(const std::vector<std::vector<std::string>>& all_feeds,
                       const std::vector<std::vector<std::string>>& all_fetches,
                       const std::vector<ResourceReflection>& resource_refs,
                       GlobalMemoryRef& gm_ref) {  // NOLINT
  for (size_t graph_id = 0; graph_id < all_feeds.size(); ++graph_id) {
    for (const auto& var : all_feeds[graph_id]) {
      gm_ref.input_keys[graph_id].emplace_back(
          GetVarKey(var, graph_id, "feed"));
    }
    for (const auto& var : all_fetches[graph_id]) {
      gm_ref.output_keys[graph_id].emplace_back(
          GetVarKey(var, graph_id, "fetch"));
    }
  }
  gm_ref.global_in_out_keys = {gm_ref.input_keys[kGlobalGraphID],
                               gm_ref.output_keys[kGlobalGraphID]};
  if (!(gm_ref.leaf_outputs.empty())) {
    size_t last_graph_id = all_feeds.size() - 1;
    for (const auto& var : gm_ref.leaf_outputs) {
      auto var_key = GetVarKey(var, last_graph_id, "leaf");
      gm_ref.output_keys[last_graph_id].emplace_back(var_key);
      gm_ref.var_to_symbol[var_key] = var_key;
      gm_ref.symbol_to_vars[var_key].emplace(var_key);
      gm_ref.leaf_output_keys.emplace_back(var_key);
    }
  }

  auto gen_weight_keys =
      [&](const ResourceReflection& res_ref, size_t start, size_t graph_id) {
        auto map_inputs_to_pd_var = res_ref.map_inputs_to_pd_var;
        size_t end = map_inputs_to_pd_var.size();
        for (size_t i = start; i < end; ++i) {
          auto var_name = map_inputs_to_pd_var[i].var_name;
          auto weight_key = "weight_" + var_name;
          gm_ref.input_keys[graph_id].emplace_back(weight_key);
          gm_ref.weights.emplace_back(var_name);
          gm_ref.weight_to_symbol[var_name] = weight_key;
          gm_ref.var_to_symbol[weight_key] = weight_key;
          gm_ref.symbol_to_vars[weight_key].emplace(weight_key);
        }

        auto map_ref_out_to_weight = res_ref.map_ref_out_to_weight;
        for (const auto map_weight : map_ref_out_to_weight) {
          auto var_name =
              map_inputs_to_pd_var[std::get<0>(map_weight.second)].var_name;
          auto key = GetVarKey(var_name, graph_id, "fetch_weight");
          gm_ref.output_keys[graph_id].emplace_back(key);
          gm_ref.var_to_symbol[key] = key;
          gm_ref.symbol_to_vars[key].emplace(key);
        }
      };

  if (resource_refs.size() == 1) {
    // standalone mode
    auto resource_ref = resource_refs[kGlobalGraphID];
    gen_weight_keys(
        resource_ref, all_feeds[kGlobalGraphID].size(), kGlobalGraphID);
  } else {
    auto fp_bp_ref = resource_refs[kFpBpGraphID];
    gen_weight_keys(fp_bp_ref, all_feeds[kFpBpGraphID].size(), kFpBpGraphID);
    auto up_g_ref = resource_refs[kUpdateGraphID];
    gen_weight_keys(up_g_ref, all_feeds[kUpdateGraphID].size(), kUpdateGraphID);
  }

  for (size_t graph_id = 0; graph_id < all_feeds.size(); ++graph_id) {
    std::string info = "input keys for graph id:" + std::to_string(graph_id);
    VLOG(6) << StringVectorDebugStr(gm_ref.input_keys[graph_id], info);
    info = "output keys for graph id:" + std::to_string(graph_id);
    VLOG(6) << StringVectorDebugStr(gm_ref.output_keys[graph_id], info);
  }
  VLOG(6) << StringMapDebugStr(gm_ref.weight_to_symbol, "weight_to_symbol");
  VLOG(6) << StringMapDebugStr(gm_ref.var_to_symbol,
                               "var_to_symbol in GenerateInOutKeys");
}

void GenerateSymbolMap(
    const std::vector<std::vector<std::string>>& all_feeds,
    const std::vector<std::vector<std::string>>& all_fetches,
    std::unordered_map<std::string, std::string>& var_to_symbol,  // NOLINT
    std::unordered_map<std::string,
                       std::unordered_set<std::string>>&
        symbol_to_vars) {  // NOLINT
  std::unordered_map<std::string, std::string> var_name_to_recent_key;

  auto can_not_reuse_any = [&](const std::vector<std::string>& var_names,
                               size_t graph_id,
                               const std::string& role) {
    for (size_t i = 0; i < var_names.size(); ++i) {
      auto key = GetVarKey(var_names[i], graph_id, role);
      var_to_symbol[key] = key;
      symbol_to_vars[key].emplace(key);
      var_name_to_recent_key[var_names[i]] = key;
    }
  };

  auto reuse_recent = [&](const std::vector<std::string>& var_names,
                          size_t graph_id,
                          const std::string& role) {
    for (size_t i = 0; i < var_names.size(); ++i) {
      auto key = GetVarKey(var_names[i], graph_id, role);
      auto reuse_key = var_name_to_recent_key[var_names[i]];  // check recently
      if (var_to_symbol.count(reuse_key) != 0) {              // can reuse
        var_to_symbol[key] = var_to_symbol[reuse_key];
        symbol_to_vars[var_to_symbol[reuse_key]].emplace(key);
      } else {
        var_to_symbol[key] = key;
        symbol_to_vars[key].emplace(key);
      }
      var_name_to_recent_key[var_names[i]] = key;
    }
  };

  if (all_feeds.size() > 1) {
    // global inputs: can not reuse any other
    can_not_reuse_any(all_feeds[kGlobalGraphID], kGlobalGraphID, "feed");

    // fp_bp inputs: can reuse global inputs
    reuse_recent(all_feeds[kFpBpGraphID], kFpBpGraphID, "feed");

    // fp_bp outputs: can not reuse any other
    can_not_reuse_any(all_fetches[kFpBpGraphID], kFpBpGraphID, "fetch");

    // update graph inputs: can reuse recent same name var
    reuse_recent(all_feeds[kUpdateGraphID], kUpdateGraphID, "feed");

    // update graph outputs: can not reuse any other
    can_not_reuse_any(all_fetches[kUpdateGraphID], kUpdateGraphID, "fetch");

    // global output: can reuse fp_bp outputs and update graph outputs
    reuse_recent(all_fetches[kGlobalGraphID], kGlobalGraphID, "fetch");

  } else {
    // standalone mode
    can_not_reuse_any(all_feeds[kGlobalGraphID], kGlobalGraphID, "feed");
    can_not_reuse_any(all_fetches[kGlobalGraphID], kGlobalGraphID, "fetch");
  }

  // print reuse result
  for (const auto& var_to_sym : var_to_symbol) {
    VLOG(6) << "var_to_symbol  var key:" << var_to_sym.first
            << ", symbol:" << var_to_sym.second;
    for (const auto& var : symbol_to_vars[var_to_sym.second]) {
      VLOG(6) << "      this symbol var key:" << var;
    }
  }
}

void GenerateAllreduceMap(
    const std::vector<std::vector<std::string>>& all_feeds,
    const std::map<std::string, PaddleVarDesc>& var_to_pd_desc,
    GlobalMemoryRef& ref) {  // NOLINT
  auto up_g_feeds = all_feeds[kUpdateGraphID];
  for (const auto& feed : up_g_feeds) {
    if (feed.find("_gcu_all_reduce") != std::string::npos) {
      std::string peer_var_name = feed.substr(0, feed.find("_gcu_all_reduce"));
      auto feed_key = GetVarKey(feed, kUpdateGraphID, "feed");
      auto peer_key = GetVarKey(peer_var_name, kFpBpGraphID, "fetch");
      // NOTE: reuse input should UpdateSymbolForVar
      UpdateSymbolForVar(feed_key, ref.var_to_symbol.at(peer_key), ref);
      ref.allreduce_params.emplace_back(
          CollectiveParams(peer_var_name,
                           ref.var_to_symbol.at(peer_key),
                           feed,
                           ref.var_to_symbol.at(feed_key),
                           var_to_pd_desc.at(peer_var_name),
                           var_to_pd_desc.at(feed),
                           true));
    }
  }
  // print allreduce map
  for (const auto& item : ref.allreduce_params) {
    VLOG(6) << "allreduce input:" << item.in_out_desc[0].var_name
            << ", symbol:" << item.in_out_desc[0].symbol
            << ", allreduce output:" << item.in_out_desc[1].var_name
            << ", symbol:" << item.in_out_desc[1].symbol;
  }
}

void GenerateWeightUpdateMap(
    const std::unordered_map<std::string, std::string>& var_to_symbol,
    const std::vector<ResourceReflection>& resource_refs,
    std::map<std::string, WeightUpdateParams>&
        weight_update_params) {  // NOLINT
  auto generate_map = [&](const ResourceReflection& ref, size_t graph_id) {
    auto map_ref_out_to_weight = ref.map_ref_out_to_weight;
    auto map_inputs_to_pd_var = ref.map_inputs_to_pd_var;
    for (const auto& item : map_ref_out_to_weight) {
      auto var_desc = map_inputs_to_pd_var[std::get<0>(item.second)];
      auto var_name = var_desc.var_name;
      auto weight_out_key = GetVarKey(var_name, graph_id, "fetch_weight");
      //   auto weight_in_key = GetVarKey(var_name, graph_id, "feed");
      weight_update_params[var_name] = WeightUpdateParams(
          var_name, var_to_symbol.at(weight_out_key), var_desc);
    }
  };

  if (resource_refs.size() == 1) {
    // standalone mode
    auto resource_ref = resource_refs[kGlobalGraphID];
    generate_map(resource_ref, kGlobalGraphID);
  } else {
    // Should be in order.
    // The weight output of UpdateGraph may replace the weight output of
    // FpBpGraph
    auto fp_bp_ref = resource_refs[kFpBpGraphID];
    generate_map(fp_bp_ref, kFpBpGraphID);
    auto up_g_ref = resource_refs[kUpdateGraphID];
    generate_map(up_g_ref, kUpdateGraphID);
  }

  // print update result
  for (const auto& item : weight_update_params) {
    VLOG(6) << "GenerateWeightUpdateMap weight_update_params:" << item.first
            << " may be updated by " << item.second.symbol;
  }
}

void GenerateGlobalMemoryRef(
    const std::vector<std::vector<std::string>>& all_feeds,
    const std::vector<std::vector<std::string>>& all_fetches,
    const std::vector<ResourceReflection>& resource_refs,
    const std::map<std::string, PaddleVarDesc>& var_to_pd_desc,
    const std::map<std::string, GcuTransInfo>& trans_infos,
    GlobalMemoryRef& ref) {  // NOLINT
  GenerateInOutKeys(all_feeds, all_fetches, resource_refs, ref);
  GenerateSymbolMap(
      all_feeds, all_fetches, ref.var_to_symbol, ref.symbol_to_vars);
  if (all_feeds.size() > 1) {
    GenerateAllreduceMap(all_feeds, var_to_pd_desc, ref);
  }
  GenerateWeightUpdateMap(
      ref.var_to_symbol, resource_refs, ref.weight_update_params);
  ref.weights_trans_info = trans_infos;
}

bool IsOptimizer(const NodePtr node) {
  if (node->IsOp() != true) return false;
  auto op_desc = node->Op();
  if (op_desc->HasAttr(kGcuGraphOpCategory)) {
    auto gcu_op_category =
        PADDLE_GET_CONST(std::string, op_desc->GetAttr(kGcuGraphOpCategory));
    return gcu_op_category == GcuGraphOpCategory::OPTIMIZER;
  }
  if (!(op_desc->HasAttr(
          paddle::framework::OpProtoAndCheckerMaker::OpRoleAttrName()))) {
    return false;
  }
  int op_role = PADDLE_GET_CONST(
      int,
      op_desc->GetAttr(
          paddle::framework::OpProtoAndCheckerMaker::OpRoleAttrName()));
  return static_cast<paddle::framework::OpRole>(op_role) ==
         paddle::framework::OpRole::kOptimize;
}
}  // namespace

std::vector<std::string> GcuCompiler::GetInOrOutVarByArcheType(
    const NodePtr node, const std::string& archetype) {
  PADDLE_ENFORCE_EQ(
      node->IsOp(),
      true,
      platform::errors::NotFound("Expect input node is op but not!"));
  bool target_in = node->Op()->Inputs().count(archetype) != 0;
  bool target_out = node->Op()->Outputs().count(archetype) != 0;
  if (!target_in && !target_out) {
    return {};
  }
  return target_in ? node->Op()->Inputs().find(archetype)->second
                   : node->Op()->Outputs().find(archetype)->second;
}

std::vector<std::set<std::string>> GcuCompiler::GetOptimizerLinkageVar(
    const NodePtr node) {
  std::vector<std::set<std::string>> res;

  if (node->IsOp() != true) {
    return res;
  }
  if (!IsOptimizer(node)) {
    return res;
  }
  auto op_desc = node->Op();
  auto optimizer_type = op_desc->Type();
  if (optimizer_type == "scale" || optimizer_type == "sum") {
    return res;
  }
  auto linkage_param = TransformUtil::GetOptimizerLinkageParam();
  if (linkage_param.count(optimizer_type) == 0) {
    PADDLE_THROW(platform::errors::NotFound(
        "optimzer %s lack of trans linkage "
        "info.Please supplement in gcu/utils/utils.cc",
        optimizer_type.c_str()));
  }
  std::vector<std::vector<std::string>> v;
  for (const auto& param : linkage_param[optimizer_type]) {
    auto var_names = GetInOrOutVarByArcheType(node, param);
    v.push_back(var_names);
  }
  size_t num = v[0].size();
  bool is_same_num = std::all_of(
      v.begin(), v.end(), [num](const std::vector<std::string>& ele) {
        return ele.size() == num;
      });
  PADDLE_ENFORCE_EQ(
      is_same_num,
      true,
      platform::errors::Fatal("optimzer %s linkage param num should same.",
                              optimizer_type.c_str()));
  for (size_t i = 0; i < num; i++) {
    std::set<std::string> linkage_param_vars;
    for (size_t j = 0; j < v.size(); j++) {
      linkage_param_vars.insert(v[j][i]);
    }
    res.push_back(linkage_param_vars);
  }
  return res;
}

void GcuCompiler::Preprocess(const Graph*& graph,
                             const std::vector<std::string>& feed_list,
                             const std::vector<std::string>& fetch_list) {
  // save var nodes
  for (Node* node : graph->Nodes()) {
    if (node->IsOp()) continue;
    if (node->inputs.size() == 1 &&
        node->inputs.at(0)->Name() == "share_buffer" && node->outputs.empty()) {
      continue;
    }
    auto op_name = node->Name();
    var_node_cache_[op_name] = node;
    if (fetch_list.empty() &&
        (!(node->inputs.empty()) && (node->outputs.empty()))) {
      leaf_var_nodes_.insert(op_name);
    }
  }
  auto sorted_ops = framework::ir::TopologySortOperations(*graph);
  // fix paddle bug that logical op output data type is default same with input
  // but running result data type is bool only
  for (const NodePtr node : sorted_ops) {
    if (!node->IsOp() || !node->Op()) {
      continue;
    }
    auto op_type = node->Op()->Type();
    if (!IsLogicalOp(op_type)) continue;
    for (NodePtr& node : node->outputs) {
      PADDLE_ENFORCE_EQ(
          node->IsVar(),
          true,
          platform::errors::Fatal("op type:%s outputs should be var type!"));
      if (!node->Var()) continue;
      auto var_desc = node->Var();
      var_desc->SetDataType(paddle::framework::proto::VarType::BOOL);
    }
  }

  if (running_mode_ == RunningMode::SERIAL ||
      running_mode_ == RunningMode::FORCE_SERIAL) {
    return;
  }
  // prepaired for no transpose
  for (const NodePtr node : sorted_ops) {
    if (!node->IsOp() || !node->Op()) {
      continue;
    }
    if (!IsOptimizer(node)) {
      if (g_channel_last_kernels.count(node->Op()->Type()) == 0) {
        continue;
      }
      // 1.process normal op
      auto op_type = node->Op()->Type();
      for (const auto& archetype_name : g_channel_last_kernels[op_type]) {
        auto vars = GetInOrOutVarByArcheType(node, archetype_name);
        std::for_each(vars.begin(),
                      vars.end(),
                      [&](const std::string& to_be_transed_var) {
                        if (trans_infos_.count(to_be_transed_var) != 0) {
                          return;
                        }
                        auto var_op = var_node_cache_[to_be_transed_var];
                        auto orgin_shape = var_op->Var()->GetShape();
                        auto dtype = paddle::framework::TransToPhiDataType(
                            var_op->Var()->GetDataType());
                        auto size = phi::SizeOf(dtype);
                        GcuTransInfo args{GcuLayout::NCHW,
                                          GcuLayout::HWCN,
                                          orgin_shape,
                                          orgin_shape,
                                          size,
                                          nullptr,
                                          nullptr};
                        GcuTransfer transfer;
                        auto ret = transfer.Trans(args, true);
                        trans_infos_[to_be_transed_var] = ret;
                        trans_infos_[to_be_transed_var + "_gcu_all_reduce"] =
                            ret;
                      });
      }
    } else {
      // 2.process optimizer op
      auto linkage_param_vars = GetOptimizerLinkageVar(node);
      if (linkage_param_vars.empty()) {
        continue;
      }
      for (const auto& s : linkage_param_vars) {
        auto is_hit =
            std::any_of(s.begin(), s.end(), [&](const std::string& opt_var) {
              return trans_infos_.count(opt_var) > 0;
            });
        if (is_hit) {
          std::for_each(s.begin(), s.end(), [&](const std::string& opt_var) {
            if (trans_infos_.count(opt_var) != 0) {
              return;
            }
            auto var_op = var_node_cache_[opt_var];
            auto orgin_shape = var_op->Var()->GetShape();
            VLOG(0) << opt_var << " " << orgin_shape.size();
            if (orgin_shape.size() < 4) {
              return;
            }
            auto dtype = paddle::framework::TransToPhiDataType(
                var_op->Var()->GetDataType());
            auto size = phi::SizeOf(dtype);
            GcuTransInfo args{GcuLayout::NCHW,
                              GcuLayout::HWCN,
                              orgin_shape,
                              orgin_shape,
                              size,
                              nullptr,
                              nullptr};
            GcuTransfer transfer;
            auto ret = transfer.Trans(args, true);
            trans_infos_[opt_var] = ret;
          });
        } else {
          continue;
        }
      }
    }
  }
}

bool GcuCompiler::IsTrainingGraph(const std::vector<const Graph*>& graphs) {
  for (size_t graph_id = 0; graph_id < graphs.size(); ++graph_id) {
    auto graph = graphs[graph_id];
    for (Node* node : graph->Nodes()) {
      if (node->IsVar()) continue;
      if (IsOptimizer(node)) {
        return true;
      }
    }
  }
  return false;
}

void GcuCompiler::Compile(
    const std::vector<const Graph*>& graph_list,
    const std::vector<std::vector<std::string>>& all_feeds,
    const std::vector<std::vector<std::string>>& all_fetches,
    const std::string& program_key) {
  std::vector<runtime::ExecutablePtr> exectuables;
  std::vector<ResourceReflection> resource_refs;
  size_t start = kGlobalGraphID;
  if (graph_list.size() > 1) {
    start = kFpBpGraphID;
    exectuables.emplace_back(nullptr);  // a placehold for GlobalGraph
    resource_refs.emplace_back(
        ResourceReflection());  // a placehold for GlobalGraph
  }
  // [global running mode]
  //  adaptive mode: do weight nchw->hwcn trans in current step and update
  //                 host weight in next step for pf
  //  series mode: no weight trans, keep weight nchw format training
  is_training_graph_ = IsTrainingGraph(graph_list);
  if (std::getenv(kRunningMode) == nullptr ||
      std::string(std::getenv(kRunningMode)) != RunningMode::FORCE_SERIAL) {
    if (is_training_graph_) {
      running_mode_ = RunningMode::ADAPTIVE;
    }
  }
  VLOG(0) << " == running mode:" << running_mode_;
  for (size_t graph_id = start; graph_id < graph_list.size(); ++graph_id) {
    builder_ = std::make_shared<GcuBuilder>();
    PADDLE_ENFORCE_NE(
        builder_,
        nullptr,
        platform::errors::Fatal("builfer is nullptr, graph id:%zu", graph_id));
    builder_->SetShapeInference(true);
    std::vector<uint64_t> input_sizes;
    std::vector<uint64_t> output_sizes;
    ConvertGraph(graph_list[graph_id],
                 all_feeds[graph_id],
                 all_fetches[graph_id],
                 input_sizes,
                 output_sizes);
    auto hlir_module = builder_->GetModule();
    VLOG(6) << "Compile begin to CompileHLIR for graph " << graph_id;
    auto exec = CompileExecutable(hlir_module);
    VLOG(6) << "Compile CompileHLIR end for graph " << graph_id;
    auto resource_ref = GetReflectionInfo();
    auto gcu_executable =
        std::shared_ptr<runtime::Executable>(new runtime::Executable(exec));
    exectuables.emplace_back(gcu_executable);
    resource_refs.emplace_back(resource_ref);
  }
  VLOG(6) << "Compile begin to GenerateGlobalMemoryRef";
  GlobalMemoryRef global_ref;
  if (!leaf_var_nodes_.empty()) {
    std::vector<std::string> leaf_outputs(leaf_var_nodes_.begin(),
                                          leaf_var_nodes_.end());
    global_ref.leaf_outputs = leaf_outputs;
  }
  if (!transed_var_nodes_.empty()) {
    global_ref.weights_to_trans.insert(transed_var_nodes_.begin(),
                                       transed_var_nodes_.end());
  }
  GenerateGlobalMemoryRef(all_feeds,
                          all_fetches,
                          resource_refs,
                          var_to_pd_var_desc_,
                          trans_infos_,
                          global_ref);
  global_ref.is_training_graph = is_training_graph_;
  global_ref.running_mode = running_mode_;
  VLOG(3) << "Compile begin to Save compile result, is_training_graph:"
          << is_training_graph_ << ", program key:" << program_key;
  TransformUtil::GraphToGcuExecutable(program_key, exectuables, resource_refs);
  TransformUtil::GraphToGlobalMemoryRef(program_key, global_ref);
  var_to_pd_var_desc_.clear();
}

void GcuCompiler::ConvertGraph(const Graph* graph,
                               const std::vector<std::string>& feed_list,
                               const std::vector<std::string>& fetch_list,
                               std::vector<uint64_t>& input_sizes,
                               std::vector<uint64_t>& output_sizes) {
  // clear history info
  Reset();
  GraphType graph_type = IdentityGraphType(graph);
  bool dump_traceback = false;
  if (std::getenv(kDumpTraceBack) != nullptr &&
      std::string(std::getenv(kDumpTraceBack)) == "true") {
    dump_traceback = true;
  }

  VLOG(6) << "ConvertGraph begin, graph_type:" << graph_type;
  Preprocess(graph, feed_list, fetch_list);

  for (std::string feed_name : feed_list) {
    auto node = var_node_cache_[feed_name];
    int64_t size = 1;
    VLOG(6) << " feed name: " << feed_name;
    GcuOpPtr feed = CreateInput(node, graph_type, size);
    auto shape = node->Var()->GetShape();
    PADDLE_ENFORCE_NE(
        IsDyn(shape),
        true,
        platform::errors::Fatal(
            "feed var %s should not be dynamic.Please check!Shape is:%s",
            feed_name.c_str(),
            TransformUtil::GetShapeStr(shape).c_str()));
    auto var_data_type = node->Var()->GetDataType();
    auto value = PaddleVarDesc(feed_name, shape, var_data_type, size);
    reflection_.map_inputs_to_pd_var[counter_] = value;
    var_to_pd_var_desc_[feed_name] = value;
    gcu_node_cache_[feed_name] = feed;
    input_sizes.push_back(size);
    counter_++;
  }

  // ordered the input by name
  std::map<std::string, Node*> input_nodes;
  for (Node* node : graph->Nodes()) {
    if (node->IsOp()) continue;
    auto op_name = node->Name();
    auto it = std::find(feed_list.begin(), feed_list.end(), op_name);
    if (it != feed_list.end()) continue;
    bool is_input_node =
        (node->inputs.empty() && (!node->outputs.empty())) ? true : false;
    VLOG(10) << "Ordered the input by name, op name:" << op_name
             << ", is_input_node:" << is_input_node
             << ", input is empty:" << node->inputs.empty()
             << ", out is empty:" << node->outputs.empty();
    if (is_input_node) {
      input_nodes.insert({op_name, node});
    }
  }
  for (const auto& input_node : input_nodes) {
    auto op_name = input_node.first;
    auto node = input_node.second;
    int64_t size = 1;
    GcuOpPtr input = CreateInput(node, graph_type, size);
    auto shape = node->Var()->GetShape();
    auto var_data_type = node->Var()->GetDataType();
    PADDLE_ENFORCE_NE(
        IsDyn(shape),
        true,
        platform::errors::Fatal(
            "input var %s should not be dynamic.Please check!Shape is:%s",
            op_name.c_str(),
            TransformUtil::GetShapeStr(shape).c_str()));
    auto value = PaddleVarDesc(op_name, shape, var_data_type, size);
    reflection_.map_inputs_to_pd_var[counter_] = value;
    var_to_pd_var_desc_[op_name] = value;
    gcu_node_cache_[op_name] = input;
    // for param update
    var_name_to_input_[op_name] = {counter_, size};
    counter_++;
  }

  // convert calc ops
  auto sorted_ops = framework::ir::TopologySortOperations(*graph);

  if (VLOG_IS_ON(10)) {
    std::set<std::string> op_types;
    for (NodePtr node : sorted_ops) {
      auto op_desc = node->Op();
      auto op_type = op_desc->Type();
      op_types.insert(op_type);
    }

    for (auto& op_type : op_types) {
      auto func =
          EquivalenceTransformer::GetInstance().Get(op_type, INSENSITIVE);
      VLOG(10) << "op type: " << op_type;
      if (func == nullptr) {
        VLOG(10) << "OpType " << op_type
                 << " is not register gcu op convert func, please check.";
      }
    }
  }

  for (NodePtr node : sorted_ops) {
    auto op_desc = node->Op();
    auto op_type = op_desc->Type();
    VLOG(10) << "op type: " << op_type;
    if (op_type == "share_buffer") {
      VLOG(10) << "Skip share_buffer when ConvertGraph";
      continue;
    }
    auto op_name = node->Name();
    auto func = EquivalenceTransformer::GetInstance().Get(op_type, INSENSITIVE);
    if (func != nullptr) {
      std::map<std::string, std::vector<GcuOpPtr>> input_ops;
      for (const auto& e : op_desc->Inputs()) {
        if (e.second.empty()) {
          continue;
        }
        if (e.first == "ReserveSpace") continue;
        std::vector<GcuOpPtr> v;
        for (std::string n : e.second) {
          auto gcu_op = gcu_node_cache_[n];
          if (gcu_op == nullptr) {
            VLOG(2) << "[WARN]Can not find transfered gcu op by"
                       "input name "
                    << n;
          }
          auto gcu_shape_str = GetShapeStr(gcu_op->GetType().GetShape());
          VLOG(3) << "Input Archetype name: " << e.first << " in name:" << n
                  << " shape:" << gcu_shape_str;
          v.push_back(gcu_op);
        }
        input_ops[e.first] = v;
      }
      VLOG(10) << "Transfered to gcu node start, op name:" << op_name
               << ", type:" << op_type;

      std::string gcu_traceback_info = "";
      if (dump_traceback && op_desc->HasAttr("gcu_traceback_info")) {
        gcu_traceback_info = PADDLE_GET_CONST(
            std::string, op_desc->GetAttr("gcu_traceback_info"));
        gcu_traceback_info = gcu_traceback_info + " op_name: " + op_name;
      } else {
        gcu_traceback_info = op_name;
      }

      SetBuilderScopeName(builder_, "paddle_trace_op_name", gcu_traceback_info);
      SetBuilderScopeName(builder_, "paddle_trace_op_type", op_type);
      GcuOpPtr op = func(builder_, node, input_ops, running_mode_);
      CleanBuilderScopeName(builder_, "paddle_trace_op_name");
      CleanBuilderScopeName(builder_, "paddle_trace_op_type");
      VLOG(10) << "Transfered to gcu node end, op name:" << op_name
               << ", type:" << op_type;
      PADDLE_ENFORCE_NE(
          op,
          nullptr,
          platform::errors::NotFound(
              "op type:%s transfered gcu node should not be nullptr!",
              op_name.c_str(),
              op_type.c_str()));
      gcu_node_cache_[op_name] = op;
      bool is_tuple_out = op->GetType().IsTuple();
      // check tuple condition same with pd
      if (is_tuple_out) {
        size_t gcu_output_num = op->GetType().GetTupleSize();
        size_t valid_output_counter = 0;
        for (const auto& e : op_desc->Outputs()) {
          if (!e.second.empty()) {
            VLOG(6) << "Out Archetype name:" << e.first;
            for (const auto& p : e.second) {
              VLOG(6) << "    correspond var name:" << p;
              valid_output_counter++;
            }
          }
        }
        if (op_type != "batch_norm") {
          PADDLE_ENFORCE_EQ(
              valid_output_counter,
              gcu_output_num,
              platform::errors::NotFound(
                  "op type:%s paddle valid output size is %u, but gcu is %u",
                  op_type.c_str(),
                  valid_output_counter,
                  gcu_output_num));
        }
      }
      if (!is_tuple_out) {
        for (const auto& e : op_desc->Outputs()) {
          if (e.second.empty()) {
            continue;
          }
          std::string weight_name = "";
          for (std::string n : e.second) {
            VLOG(3) << "Output Archetype name: " << e.first
                    << " out name:" << n;
            auto out_name = n;
            weight_name = out_name;
            gcu_node_cache_[out_name] = op;
            if (running_mode_ == RunningMode::ADAPTIVE) {
              continue;
            }
            // for shape infer check
            auto gcu_shape = op->GetType().GetShape();
            auto var_op = var_node_cache_[out_name];
            auto paddle_shape = var_op->Var()->GetShape();
            if (VLOG_IS_ON(10) && gcu_shape.size() != paddle_shape.size())
              builder_->Dump();
            // normalize scalar shape process, [] -> [1]
            if (gcu_shape.empty()) {
              gcu_shape = {1};
            }
            if (paddle_shape.empty()) {
              paddle_shape = {1};
            }
            PADDLE_ENFORCE_EQ(gcu_shape.size(),
                              paddle_shape.size(),
                              platform::errors::NotFound(
                                  "op_name:%s, op type:%s, out_name:%s"
                                  " transfered gcu node should have same rank!"
                                  "but paddle rank is %u, gcu rank is %u,"
                                  " paddle shape:%s, gcu shape:%s",
                                  op_name.c_str(),
                                  op_type.c_str(),
                                  out_name.c_str(),
                                  paddle_shape.size(),
                                  gcu_shape.size(),
                                  GetShapeStr(paddle_shape).c_str(),
                                  GetShapeStr(gcu_shape).c_str()));
            auto gcu_shape_str = GetShapeStr(gcu_shape);
            auto paddle_shape_str = GetShapeStr(paddle_shape);
            if (IsDyn(paddle_shape) && !IsDyn(gcu_shape)) {
              VLOG(6) << "out var_name:" << out_name.c_str() << " "
                      << "op_type:" << op_type.c_str() << " "
                      << "shape_pd:" << paddle_shape_str << " "
                      << "shape_gcu:" << gcu_shape_str << " "
                      << "[WARN]use gcu shape to flush paddle shape!";
              var_op->Var()->SetShape(gcu_shape);
              g_map_var_fixed_shape[out_name] = gcu_shape;
              continue;
            }
            if (VLOG_IS_ON(10) && gcu_shape_str != paddle_shape_str)
              builder_->Dump();
            PADDLE_ENFORCE_EQ(gcu_shape_str,
                              paddle_shape_str,
                              platform::errors::NotFound(
                                  "op_name:%s, op type:%s, out_name:%s"
                                  " transfered gcu node should have same shape!"
                                  "but paddle shape is %s, gcu shape is %s",
                                  op_name.c_str(),
                                  op_type.c_str(),
                                  out_name.c_str(),
                                  paddle_shape_str.c_str(),
                                  gcu_shape_str.c_str()));
          }
          if (IsRefNode(node)) {
            weights_name_to_gcuop_.insert({weight_name, op});
          }
        }
      } else {
        std::set<std::string> names_in;
        for (const auto& e : op_desc->Inputs()) {
          if (e.second.empty()) {
            continue;
          }
          for (std::string n : e.second) {
            names_in.insert(n);
          }
        }
        for (const auto& e : op_desc->Outputs()) {
          if (e.second.empty()) continue;
          if (e.first == "ReserveSpace") continue;
          for (const auto& out_name : e.second) {
            auto out_var_op = var_node_cache_[out_name];
            PADDLE_ENFORCE_NE(
                out_var_op,
                nullptr,
                platform::errors::NotFound(
                    "op name:%s op type:%s out name:%s not found var op!",
                    op_name.c_str(),
                    op_type.c_str(),
                    out_name.c_str()));
            VLOG(6) << "  out var name:" << out_name
                    << " var op name:" << out_var_op->Name();
            GcuOpPtr gte = AddGteOp(out_var_op, op);
            PADDLE_ENFORCE_NE(
                gte,
                nullptr,
                platform::errors::NotFound(
                    "op name:%s op type:%s transfer to gcu gte node failed!",
                    op_name.c_str(),
                    op_type.c_str()));
            gcu_node_cache_[out_name] = gte;
            // for param update
            if (names_in.count(out_name) != 0) {
              weights_name_to_gcuop_.insert({out_name, gte});
            }
            if (running_mode_ == RunningMode ::ADAPTIVE) {
              continue;
            }
            // for shape infer check
            if (e.first == "ReserveSpace") {
              continue;
            }
            auto gcu_shape = gte->GetType().GetShape();
            auto var_op = var_node_cache_[out_name];
            auto paddle_shape = var_op->Var()->GetShape();
            if (VLOG_IS_ON(10) && gcu_shape.size() != paddle_shape.size())
              builder_->Dump();
            // normalize scalar shape process, [] -> [1]
            if (gcu_shape.empty()) {
              gcu_shape = {1};
            }
            if (paddle_shape.empty()) {
              paddle_shape = {1};
            }
            PADDLE_ENFORCE_EQ(gcu_shape.size(),
                              paddle_shape.size(),
                              platform::errors::NotFound(
                                  "op_name:%s, op type:%s, out_name:%s"
                                  " transfered gcu node should have same rank!"
                                  "but paddle rank is %u, gcu rank is %u,"
                                  " paddle shape:%s, gcu shape:%s",
                                  op_name.c_str(),
                                  op_type.c_str(),
                                  out_name.c_str(),
                                  paddle_shape.size(),
                                  gcu_shape.size(),
                                  GetShapeStr(paddle_shape).c_str(),
                                  GetShapeStr(gcu_shape).c_str()));
            auto gcu_shape_str = GetShapeStr(gcu_shape);
            auto paddle_shape_str = GetShapeStr(paddle_shape);
            if (IsDyn(paddle_shape) && !IsDyn(gcu_shape)) {
              VLOG(6) << "out var_name:" << out_name.c_str() << " "
                      << "op_type:" << op_type.c_str() << " "
                      << "shape_pd:" << paddle_shape_str << " "
                      << "shape_gcu:" << gcu_shape_str << " "
                      << "[WARN] use gcu shape to flush paddle shape!";
              var_op->Var()->SetShape(gcu_shape);
              g_map_var_fixed_shape[out_name] = gcu_shape;
              continue;
            }
            if (VLOG_IS_ON(10) && gcu_shape_str != paddle_shape_str)
              builder_->Dump();
            PADDLE_ENFORCE_EQ(
                gcu_shape_str,
                paddle_shape_str,
                platform::errors::NotFound(
                    "op_name:%s, op type:%s, out_name:%s"
                    " transfered gcu node should have same shape! but paddle"
                    " shape is %s, gcu shape is %s, output name is %s",
                    op_name.c_str(),
                    op_type.c_str(),
                    out_name.c_str(),
                    paddle_shape_str.c_str(),
                    gcu_shape_str.c_str(),
                    out_name.c_str()));
          }
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::NotFound(
              "OpType %s is not register gcu op convert func, please check.",
              op_type.c_str()));
    }

    VLOG(10) << "op_type:" << op_type << " name:" << node->Name();
  }
  // convert output nodes
  SetOutput(fetch_list, graph_type, output_sizes);
  VLOG(6) << "Convert to gcu graph finished!";
  if (VLOG_IS_ON(6)) {
    VLOG(6) << "Hlir IrGraph After convert Paddle IR";
    builder_->Dump();
  }
}

GcuOpPtr GcuCompiler::CreateInput(const Node* node,
                                  const GraphType& graph_type,
                                  int64_t& size) {
  PADDLE_ENFORCE_NOT_NULL(node, "input node is null.");
  if (!node->IsVar()) {
    VLOG(3) << "ERROR! op name:" << node->Name()
            << " should be var type when convert to Gcu input node";
    return nullptr;
  }
  auto var_name = node->Name();
  PADDLE_ENFORCE_NOT_NULL(node->Var(), "var[%s] has no opdesc.", var_name);
  auto shape = node->Var()->GetShape();
  auto dtype =
      paddle::framework::TransToPhiDataType(node->Var()->GetDataType());
  auto ptype = TransformUtil::ConvertDataType(dtype);
  size = node->Var()->ElementSize();
  for (const auto& dim : shape) {
    size *= dim;
  }

  // for no transpose in graph
  bool trans_in_static =
      (!trans_infos_.empty()) && (trans_infos_.count(var_name) != 0);

  GcuShape dst_shape;
  if (trans_in_static) {
    dst_shape = trans_infos_[var_name].dst_shape;
  } else {
    dst_shape = shape;
  }
  builder::Type input_type(dst_shape, ptype);
  return std::make_shared<GcuOp>(builder_->CreateInput(input_type));
}

GcuOpPtr GcuCompiler::AddGteOp(const Node* node, const GcuOpPtr& input) {
  if (!node->IsVar()) {
    VLOG(3) << "ERROR! op name:" << node->Name()
            << " should be var type when convert to Gcu gte node";
    return nullptr;
  }
  auto attr_out_var_names = input->GetAttribute(kAttrOpOutVarName);
  PADDLE_ENFORCE_NE(attr_out_var_names == builder::Attribute(""),
                    true,
                    platform::errors::NotFound(
                        "lack of attr [%s] for gcu tuple op, please check.",
                        kAttrOpOutVarName));
  std::string out_var_names = attr_out_var_names.GetValueAsString();
  auto list_out_var_names = ParseAttr(out_var_names);
  auto var_name = node->Name();
  int32_t idx = 0;
  VLOG(3) << out_var_names;
  for (const auto& name : list_out_var_names) {
    if (name != var_name) {
      idx++;
    } else {
      break;
    }
  }
  auto shape = node->Var()->GetShape();
  auto dtype =
      paddle::framework::TransToPhiDataType(node->Var()->GetDataType());
  auto ptype = TransformUtil::ConvertDataType(dtype);
  builder::Type input_type(shape, ptype);
  return std::make_shared<GcuOp>(builder::GetTupleElement(*input, idx));
}

void GcuCompiler::SetOutput(const std::vector<std::string>& fetch_list,
                            const GraphType& graph_type,
                            std::vector<uint64_t>& output_sizes) {
  std::vector<::builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  std::vector<GcuOp> outputs;

  int32_t idx = SetOutputWithFetchList(
      fetch_list, graph_type, tuple_shape, tuple_dtype, outputs);
  if (idx == 1 && weights_name_to_gcuop_.empty()) {
    for (auto& gcu_op : outputs) {
      output_sizes.push_back(gcu_op.GetType().GetSize());
    }
    builder_->SetOutput(outputs);
    return;
  }

  // add extra output for param update
  for (const auto& p : weights_name_to_gcuop_) {
    std::string weight_name = p.first;
    auto gcu_op = p.second;
    if (!gcu_op) {
      PADDLE_ENFORCE_EQ(
          true,
          false,
          platform::errors::NotFound(
              "weight name [%s] correspond gcu op is nullptr, please check.",
              weight_name.c_str()));
    }
    // exist some op is ref node that it only pass input to output but not
    // change value such as sum op
    if (var_name_to_input_.count(weight_name) == 0) {
      VLOG(1) << "[warn]weight name [" << weight_name
              << "] not found in inputs list.";
      continue;
    }
    // save weight reflection info
    auto pair = var_name_to_input_[weight_name];
    auto value = std::tie(pair.first, pair.second);

    reflection_.map_ref_out_to_weight[idx] = value;
    tuple_shape.push_back(gcu_op->GetType().GetShape());
    tuple_dtype.push_back(gcu_op->GetType().GetPrimitiveType());
    outputs.push_back(*gcu_op);
    ++idx;
  }

  if (outputs.empty()) {
    if (SetOutputWithLeafNodes(graph_type, tuple_shape, tuple_dtype, outputs)) {
      return;
    }
  } else {
    leaf_var_nodes_.clear();
  }

  for (auto& gcu_op : outputs) {
    output_sizes.push_back(gcu_op.GetType().GetSize());
  }

  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto tuple = builder::Tuple(outputs, outputs_type);
  builder_->SetOutput({tuple});
}

void GcuCompiler::PostProcess(
    const std::vector<const Graph*>& before_graph_list,
    const Graph* post_graph) {
  auto sorted_ops = framework::ir::TopologySortOperations(*post_graph);
  for (NodePtr node : sorted_ops) {
    auto op_desc = node->Op();
    auto op_type = op_desc->Type();
    auto op_name = node->Name();
    if (op_type == "gcu_runtime") {
      TransformUtil::GcuRuntimeNodeToGraph(before_graph_list, node);
    }
  }
}

int32_t GcuCompiler::SetOutputWithFetchList(
    const std::vector<std::string>& fetch_list,
    const GraphType& graph_type,
    std::vector<std::vector<int64_t>>& tuple_shape,
    std::vector<::builder::PrimitiveType>& tuple_dtype,
    std::vector<GcuOp>& outputs) {
  int32_t idx = 0;
  for (auto tensor_name : fetch_list) {
    PADDLE_ENFORCE_NE(
        (gcu_node_cache_.count(tensor_name) == 0 ||
         var_node_cache_.count(tensor_name) == 0),
        true,
        platform::errors::PreconditionNotMet(
            "Output tensor %s is not found, gcu_node:%zu, var_node:%zu",
            tensor_name.c_str(),
            gcu_node_cache_.count(tensor_name),
            var_node_cache_.count(tensor_name)));

    auto fetch_node = gcu_node_cache_[tensor_name];
    auto fetch_var_node = var_node_cache_[tensor_name];
    auto shape = fetch_node->GetType().GetShape();
    auto paddle_data_type = fetch_var_node->Var()->GetDataType();
    int64_t size = fetch_var_node->Var()->ElementSize();
    for (const auto& dim : shape) {
      size *= dim;
    }

    auto value = PaddleVarDesc(tensor_name, shape, paddle_data_type, size);
    reflection_.map_outputs_to_pd_var[idx] = value;
    var_to_pd_var_desc_[tensor_name] = value;
    tuple_shape.push_back(fetch_node->GetType().GetShape());
    tuple_dtype.push_back(fetch_node->GetType().GetPrimitiveType());
    outputs.push_back(*fetch_node);
    VLOG(6) << "Add output[" << idx << "] with tensor:" << tensor_name;
    ++idx;
  }
  return idx;
}

bool GcuCompiler::SetOutputWithLeafNodes(
    const GraphType& graph_type,
    std::vector<std::vector<int64_t>>& tuple_shape,
    std::vector<::builder::PrimitiveType>& tuple_dtype,
    std::vector<GcuOp>& outputs) {
  PADDLE_ENFORCE_NE(
      leaf_var_nodes_.empty(),
      true,
      platform::errors::PreconditionNotMet(
          "Leaf var node is empty while outputs is not set, please check."));

  std::vector<std::string> fetch_list(leaf_var_nodes_.begin(),
                                      leaf_var_nodes_.end());
  int32_t idx = SetOutputWithFetchList(
      fetch_list, graph_type, tuple_shape, tuple_dtype, outputs);
  if (idx == 1) {
    builder_->SetOutput(outputs);
    VLOG(6) << "Add single output with leaf node:" << fetch_list[0];
    return true;
  }
  return false;
}

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
