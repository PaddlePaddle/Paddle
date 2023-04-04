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

#include "paddle/fluid/inference/analysis/passes/offload_params_pass.h"

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/offload_vars_pool.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace inference {
namespace analysis {

#ifdef PADDLE_WITH_CUDA
constexpr int kRootBlockIndex = 0;

void CopyFixedVars2GPU(const std::unordered_set<std::string> &fixed_vars,
                       framework::Scope *scope,
                       int device_id) {
  for (auto var_name : fixed_vars) {
    auto *var = scope->FindLocalVar(var_name);
    auto *t = var->GetMutable<phi::DenseTensor>();
    platform::Place place = platform::CUDAPlace(device_id);
    platform::CPUPlace cpu_place;
    phi::DenseTensor temp_tensor;
    temp_tensor.Resize(t->dims());
    framework::TensorCopySync(*t, cpu_place, &temp_tensor);
    t->clear();
    framework::TensorCopySync(temp_tensor, place, t);
  }
}

std::list<framework::LayerIdx2ParamsTensors> CopyOffloadLayersParams2Cpu(
    std::unordered_set<size_t> *offload_layers,
    const std::map<size_t, std::vector<std::string>> &offload_layer_attrs,
    std::unordered_set<std::string> *copy_completed_offload_vars,
    const std::unordered_set<std::string> &fixed_vars,
    framework::Scope *scope) {
  std::list<framework::LayerIdx2ParamsTensors> weight_queue;
  for (size_t layer : *offload_layers) {
    framework::LayerIdx2ParamsTensors info;
    info.first = layer;
    for (auto var_name : offload_layer_attrs.at(layer)) {
      if (fixed_vars.count(var_name)) continue;
      auto *src_var = scope->GetVar(var_name);
      CHECK(src_var->IsType<phi::DenseTensor>());
      auto *src_tensor = src_var->GetMutable<phi::DenseTensor>();
      framework::Variable *dst_var = nullptr;
      phi::DenseTensor *dst_tensor = nullptr;
      if (copy_completed_offload_vars->count(var_name)) {
        dst_var = scope->GetVar(var_name + "_cpu");
        dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
        info.second.first.push_back(dst_tensor);
        info.second.second.push_back(src_tensor);
      } else {
        copy_completed_offload_vars->insert(var_name);
        dst_var = scope->Var(var_name + "_cpu");
        dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
        dst_tensor->Resize(src_tensor->dims());
        dst_tensor->set_layout(src_tensor->layout());
        dst_tensor->set_type(src_tensor->dtype());
        platform::CUDAPinnedPlace cpu_pin_place;
        framework::TensorCopySync(*src_tensor, cpu_pin_place, dst_tensor);
        src_tensor->clear();
        info.second.first.push_back(dst_tensor);
        info.second.second.push_back(src_tensor);
      }
    }
    if (info.second.first.size()) weight_queue.push_back(info);
  }

  offload_layers->clear();
  for (auto &ele : weight_queue) {
    offload_layers->insert(ele.first);
  }

  return weight_queue;
}

bool FindFusedMultiTransformerOp(framework::OpDesc *while_op,
                                 const framework::ir::Graph &graph) {
  auto *sub_block =
      PADDLE_GET_CONST(framework::BlockDesc *, while_op->GetAttr("sub_block"));
  auto sub_graph = graph.GetSubGraph(sub_block->ID());
  CHECK(sub_graph != nullptr);
  auto topo_order = framework::ir::TopologySortOperations(*sub_graph);
  for (size_t i = 0; i < topo_order.size(); i++) {
    auto &node = topo_order[i];
    if (node->IsOp() && node->Op()->Type() == "fused_multi_transformer") {
      return true;
    }
  }
  return false;
}

std::unordered_set<size_t> CollectOffloadLayersAndParams(
    const std::vector<framework::ir::Node *> &block_topo_order,
    int block_id,
    std::map<size_t, std::vector<std::string>> *offload_layer_attrs,
    std::unordered_set<std::string> *fixed_vars,
    framework::Scope *scope,
    size_t *buffer_size,
    Argument *argument) {
  std::unordered_set<size_t> offload_layers;
  for (size_t i = 0; i < block_topo_order.size(); i++) {
    auto &node = block_topo_order[i];
    if (node->IsOp() && node->Op()->Type() == "fused_multi_transformer") {
      node->Op()->SetAttr("offload_params", true);
      node->Op()->SetAttr("offload_vars_pool_idx", block_id);

      std::vector<std::string> qkv_w_var_names = node->Op()->Input("QKVW");
      size_t layer_num = qkv_w_var_names.size();

      std::vector<std::string> expected_offload_params{"QKVW",
                                                       "FFN1Weight",
                                                       "FFN2Weight",
                                                       "OutLinearW",
                                                       "FFN1Bias",
                                                       "FFN2Bias",
                                                       "FFNLnBias",
                                                       "LnBias",
                                                       "OutLinearBias",
                                                       "QKVBias"};
      std::vector<int> custom_offload_layers =
          argument->custom_offload_layers();
      for (size_t j = 0; j < layer_num; j++) {
        if (custom_offload_layers.size() &&
            std::find(custom_offload_layers.begin(),
                      custom_offload_layers.end(),
                      j) == custom_offload_layers.end()) {
          continue;
        }
        offload_layers.insert(j);
        size_t offload_params_size = 0;
        for (auto offload_param : expected_offload_params) {
          PADDLE_ENFORCE_EQ(
              node->Op()->Input(offload_param).size(),
              layer_num,
              platform::errors::PreconditionNotMet(
                  "offload_param size should be equal to layer_num"));
          auto offload_var_name = node->Op()->Input(offload_param)[j];
          (*offload_layer_attrs)[j].push_back(offload_var_name);
          fixed_vars->erase(offload_var_name);
          auto *offload_var = scope->GetVar(offload_var_name);
          auto *offload_tensor = offload_var->GetMutable<phi::DenseTensor>();
          offload_params_size +=
              offload_tensor->numel() * SizeOf(offload_tensor->dtype());
        }
        if (offload_params_size > *buffer_size)
          *buffer_size = offload_params_size;
      }
      break;
    }
  }
  return offload_layers;
}
#endif

void OffLoadParamsPass::RunImpl(Argument *argument) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_EQ(
      argument->scope_valid(),
      true,
      platform::errors::PreconditionNotMet("The scope field should be valid"));

  if (!argument->use_gpu()) return;
  if (!argument->enable_offload_valid()) return;

  auto &graph = argument->main_graph();
  PADDLE_ENFORCE_EQ(argument->gpu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The gpu_device_id field should be valid"));
  auto *scope = argument->scope_ptr();
  auto main_block_topo_order = framework::ir::TopologySortOperations(graph);
  framework::OpDesc *while_op_with_decoder = nullptr;

  std::unordered_set<std::string> fixed_vars;
  for (auto iter = main_block_topo_order.begin();
       iter != main_block_topo_order.end();) {
    auto &node = *iter;
    if (!node->IsOp()) continue;
    for (auto *var_node : node->inputs) {
      if (!var_node->Var()->Persistable()) continue;
      auto var_name = var_node->Var()->Name();
      auto *var = scope->FindLocalVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var,
                              platform::errors::PreconditionNotMet(
                                  "The var should not be nullptr"));
      if (var->IsType<phi::DenseTensor>()) {
        fixed_vars.insert(var_name);
      }
      if (node->Op()->Type() == "while") {
        if (FindFusedMultiTransformerOp(node->Op(), graph))
          while_op_with_decoder = node->Op();
      }
    }
    iter++;
  }

  std::unordered_set<std::string> copy_completed_offload_vars;
  if (while_op_with_decoder) {
    auto *sub_block = PADDLE_GET_CONST(
        framework::BlockDesc *, while_op_with_decoder->GetAttr("sub_block"));
    auto while_op_sub_graph = graph.GetSubGraph(sub_block->ID());
    CHECK(while_op_sub_graph != nullptr);
    auto while_block_topo_order =
        framework::ir::TopologySortOperations(*while_op_sub_graph);
    std::map<size_t, std::vector<std::string>> offload_decoder_layers_attrs;

    size_t buffer_size = 0;
    auto offload_decoder_layers =
        CollectOffloadLayersAndParams(while_block_topo_order,
                                      sub_block->ID(),
                                      &offload_decoder_layers_attrs,
                                      &fixed_vars,
                                      scope,
                                      &buffer_size,
                                      argument);
    if (offload_decoder_layers.size() > 0) {
      auto weight_queue =
          CopyOffloadLayersParams2Cpu(&offload_decoder_layers,
                                      offload_decoder_layers_attrs,
                                      &copy_completed_offload_vars,
                                      fixed_vars,
                                      scope);
      framework::OffloadVarsPoolVector::Instance().Init(
          sub_block->ID(), buffer_size, weight_queue, offload_decoder_layers);
    }
  }

  std::map<size_t, std::vector<std::string>> offload_encoder_layers_attrs;
  size_t buffer_size = 0;
  auto offload_encoder_layers =
      CollectOffloadLayersAndParams(main_block_topo_order,
                                    kRootBlockIndex,
                                    &offload_encoder_layers_attrs,
                                    &fixed_vars,
                                    scope,
                                    &buffer_size,
                                    argument);
  if (offload_encoder_layers.size() > 0) {
    auto weight_queue =
        CopyOffloadLayersParams2Cpu(&offload_encoder_layers,
                                    offload_encoder_layers_attrs,
                                    &copy_completed_offload_vars,
                                    fixed_vars,
                                    scope);
    framework::OffloadVarsPoolVector::Instance().Init(
        kRootBlockIndex, buffer_size, weight_queue, offload_encoder_layers);
  }

  CopyFixedVars2GPU(fixed_vars, scope, argument->gpu_device_id());
#endif
}

std::string OffLoadParamsPass::repr() const { return "offload_params_pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
