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
#include "paddle/fluid/framework/sched_layers_pool.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace inference {
namespace analysis {

#if defined(PADDLE_WITH_CUDA)
std::pair<std::vector<size_t>, std::vector<size_t>> DivideLayer(
    std::map<size_t, std::vector<phi::DenseTensor *>>
        &op_idx_2_params_tensors,  // NOLINT
    std::map<size_t, std::vector<std::string>>
        &op_idx_2_params_var_names,                                // NOLINT
    std::map<size_t, std::string> &op_idx_2_op_name,               // NOLINT
    std::map<std::string, phi::DenseTensor *> &varname_2_tensor,   // NOLINT
    std::vector<std::pair<size_t, size_t>> &op_idx_2_params_size,  // NOLINT
    std::map<std::string, size_t> varname_2_varsize,               // NOLINT
    std::unordered_set<std::string> &fixed_vars,                   // NOLINT
    long long rest_memory,                                         // NOLINT
    int fixedlayer_algorithm,
    size_t &buffer_size) {  // NOLINT

  std::vector<size_t> fixed_layers;
  std::vector<size_t> sched_layers;

  // 下面根据不同的方式来划分 fixed_layers
  // fixedlayer_algorithm == 0 优先把权重较大的 op 划入 fixed layer
  // fixedlayer_algorithm == 1 优先把权重较小的 op 划入 fixed layer
  if (fixedlayer_algorithm == 0 || fixedlayer_algorithm == 1) {
    if (fixedlayer_algorithm == 0) {
      std::sort(op_idx_2_params_size.begin(),
                op_idx_2_params_size.end(),
                [](std::pair<size_t, size_t> &a, std::pair<size_t, size_t> &b) {
                  return a.second > b.second;
                });
    }
    if (fixedlayer_algorithm == 1) {
      std::sort(op_idx_2_params_size.begin(),
                op_idx_2_params_size.end(),
                [](std::pair<size_t, size_t> &a, std::pair<size_t, size_t> &b) {
                  return a.second < b.second;
                });
    }
    size_t i = 0;
    while (i < op_idx_2_params_size.size()) {
      auto &pair = op_idx_2_params_size[i];
      for (auto var_name : op_idx_2_params_var_names.at(pair.first)) {
        if (!fixed_vars.count(var_name)) {
          rest_memory -= varname_2_varsize.at(var_name);
        }
      }
      if (rest_memory >= 0) {
        fixed_layers.push_back(pair.first);
        fixed_vars.insert(op_idx_2_params_var_names.at(pair.first).begin(),
                          op_idx_2_params_var_names.at(pair.first).end());
      } else {
        break;
      }
      i++;
    }

    for (; i < op_idx_2_params_size.size(); i++) {
      if (op_idx_2_params_size[i].second > buffer_size)
        buffer_size = op_idx_2_params_size[i].second;
      sched_layers.push_back(op_idx_2_params_size[i].first);
    }
  } else if (fixedlayer_algorithm == 2) {
    // 专门针对 transformer 模型
    std::vector<size_t> fused_multi_transformer_op_idx;
    std::map<size_t, size_t>fused_multi_transformer_op_idx_2_params_size;
    for(auto op_idx_and_params_size : op_idx_2_params_size) {
      auto idx = op_idx_and_params_size.first;
      auto params_size = op_idx_and_params_size.second;
      if(op_idx_2_op_name.at(idx) == "fused_multi_transformer") {
        fused_multi_transformer_op_idx.push_back(idx);
        LOG(INFO) << "idx: " << idx;
        fused_multi_transformer_op_idx_2_params_size[idx] = params_size;
      } else {
        fixed_layers.push_back(idx);
        fixed_vars.insert(op_idx_2_params_var_names.at(idx).begin(),
                          op_idx_2_params_var_names.at(idx).end());
      }
      buffer_size = 402792448;
      
    }
    auto max_fixed = std::min(size_t(rest_memory / buffer_size), fused_multi_transformer_op_idx.size());
    LOG(INFO) << "max_fixed: " << max_fixed << ", total fused transformer layers: " << fused_multi_transformer_op_idx.size();
    auto scheduled_layers = fused_multi_transformer_op_idx.size() - max_fixed;
    /*
    max_fixed = min(max_fixed, total_layers)
    scheduled_layers = total_layers - max_fixed
    vals = [(i + 1) * scheduled_layers // total_layers for i in range(total_layers)]
    ret : List[int] = []
    last_v = 0
    for i, v in enumerate(vals):
        if v == last_v:
            ret.append(i)
        else:
            last_v = v
    return ret
    */
    std::vector<size_t> vals;
    for (size_t i = 0; i < fused_multi_transformer_op_idx.size(); i++) {
      vals.push_back((i + 1) * scheduled_layers / fused_multi_transformer_op_idx.size());
    }

    std::cout << "vals: ";
    for(auto i : vals)
      std::cout << i << " ";
    std::cout << std::endl;

    std::vector<size_t> tmp;
    size_t last_v = 0;
    for(size_t i = 0; i < vals.size(); i++) {
      auto v = vals[i];
      if(v == last_v)
        tmp.push_back(i);
      else
        last_v = v;
    }

    std::cout << "tmp: ";
    for(auto i : tmp)
      std::cout << i << " ";
    std::cout << std::endl;

    for(size_t i = 0; i < tmp.size(); i++) {
      auto fixed_layer = fused_multi_transformer_op_idx.at(tmp.at(i));
      fixed_layers.push_back(fixed_layer);
      fixed_vars.insert(op_idx_2_params_var_names.at(fixed_layer).begin(),
                          op_idx_2_params_var_names.at(fixed_layer).end());
    }
    for (auto i : fused_multi_transformer_op_idx) {
      if(std::find(fixed_layers.begin(),
                  fixed_layers.end(),
                  i) == fixed_layers.end()) {
        sched_layers.push_back(i);    
      }
    }
  } else {
    CHECK(false) << "fixedlayer_algorithm invalid";
  }

  LOG(INFO) << "fixed_vars: " << fixed_vars.size();
  LOG(INFO) << "fixed_layers: " << fixed_layers.size();
  for (size_t i : fixed_layers) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::sort(sched_layers.begin(), sched_layers.end());
  LOG(INFO) << "sched_layers: " << sched_layers.size();
  for (auto i : sched_layers) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  return std::pair<std::vector<size_t>, std::vector<size_t>>(fixed_layers,
                                                             sched_layers);
}

void CopyFixedlyer2Gpu(
    std::vector<size_t> &fixed_layers,  // NOLINT
    std::map<size_t, std::vector<std::string>>
        &op_idx_2_params_var_names,                               // NOLINT
    std::map<std::string, phi::DenseTensor *> &varname_2_tensor,  // NOLINT
    std::unordered_set<std::string> &copy_completed_fixed_vars,   // NOLINT
    platform::Place &place) {                                     // NOLINT
  size_t avail = 0;
  size_t total = 0;
  cudaSetDevice(platform::GetCurrentDeviceId());
  cudaMemGetInfo(&avail, &total);
  LOG(INFO) << "Before fixed_layers copy, used mem: "
            << (total - avail) / 1024 / 1024 << "MiB";

  for (size_t layer : fixed_layers) {
    for (auto var_name : op_idx_2_params_var_names[layer]) {
      if (copy_completed_fixed_vars.count(var_name)) continue;
      // LOG(INFO) << "copy var_name: " << var_name;
      copy_completed_fixed_vars.insert(var_name);
      auto *t = varname_2_tensor.at(var_name);
      platform::CPUPlace cpu_place;
      phi::DenseTensor temp_tensor;
      temp_tensor.Resize(t->dims());
      paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
      t->clear();
      paddle::framework::TensorCopySync(temp_tensor, place, t);
    }
    // cudaSetDevice(platform::GetCurrentDeviceId());
    // cudaMemGetInfo(&avail, &total);
    // LOG(INFO) << "After layer idx_" << layer
    //           << "_copy, used mem: " << (total - avail) / 1024 / 1024 << "MiB";
  }
  cudaSetDevice(platform::GetCurrentDeviceId());
  cudaMemGetInfo(&avail, &total);
  LOG(INFO) << "After fixed_layers copy, used mem: "
            << (total - avail) / 1024 / 1024 << "MiB";
}

std::list<framework::OpIdx2ParamsTensors> CopySchedlyer2Cpu(
    std::vector<size_t> &sched_layers,  // NOLINT
    std::map<size_t, std::vector<std::string>>
        &op_idx_2_params_var_names,                              // NOLINT
    std::unordered_set<std::string> &copy_completed_sched_vars,  // NOLINT
    std::unordered_set<std::string> &fixed_vars,                 // NOLINT
    framework::Scope *scope,
    bool pin_memory) {
  std::list<framework::OpIdx2ParamsTensors> weight_queue;
  for (size_t layer : sched_layers) {
    framework::OpIdx2ParamsTensors info;
    info.first = layer;
    for (std::string &var_name : op_idx_2_params_var_names[layer]) {
      if (fixed_vars.count(var_name)) continue;
      auto *src_var = scope->GetVar(var_name);
      CHECK(src_var->IsType<phi::DenseTensor>());
      auto *src_tensor = src_var->GetMutable<phi::DenseTensor>();
      framework::Variable *dst_var = nullptr;
      phi::DenseTensor *dst_tensor = nullptr;
      if (copy_completed_sched_vars.count(var_name)) {
        dst_var = scope->GetVar(var_name + "_cpu");
        dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
        info.second.first.push_back(dst_tensor);
        info.second.second.push_back(src_tensor);
        continue;
      } else {
        copy_completed_sched_vars.insert(var_name);
        dst_var = scope->Var(var_name + "_cpu");
        dst_tensor = dst_var->GetMutable<phi::DenseTensor>();
        dst_tensor->Resize(src_tensor->dims());
        dst_tensor->set_layout(src_tensor->layout());
        dst_tensor->set_type(src_tensor->dtype());
        if (pin_memory) {
          platform::CUDAPinnedPlace cpu_pin_place;
          paddle::framework::TensorCopySync(
              *src_tensor, cpu_pin_place, dst_tensor);
        } else {
          platform::CPUPlace cpu_place;
          paddle::framework::TensorCopySync(*src_tensor, cpu_place, dst_tensor);
        }
        src_tensor->clear();
        info.second.first.push_back(dst_tensor);
        info.second.second.push_back(src_tensor);
      }
    }
    if (info.second.first.size()) weight_queue.push_back(info);
  }

  sched_layers.clear();
  for (auto &ele : weight_queue) {
    sched_layers.push_back(ele.first);
  }

  return weight_queue;
}
#endif

void OffLoadParamsPass::RunImpl(Argument *argument) {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_EQ(
      argument->scope_valid(),
      true,
      platform::errors::PreconditionNotMet("The scope field should be valid"));

  if (!argument->use_gpu()) return;
  if (!argument->memory_limit_valid()) return;

  auto &graph = argument->main_graph();
  PADDLE_ENFORCE_EQ(argument->gpu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The gpu_device_id field should be valid"));
  auto *scope = argument->scope_ptr();

  platform::Place place = platform::CUDAPlace(argument->gpu_device_id());

  auto main_block_topo_order =
      paddle::framework::ir::TopologySortOperations(graph);

  // remove feed, fetch and while op。在执行的时候，并不会执行feed、 fetch
  // op；while op对应的子 block 我们后续需要特殊处理，所以这里也一并删除
  paddle::framework::OpDesc *while_op = nullptr;
  for (auto iter = main_block_topo_order.begin();
       iter != main_block_topo_order.end();) {
    auto &node = *iter;
    if (node->Op()->Type() == "feed" || node->Op()->Type() == "fetch" ||
        node->Op()->Type() == "while") {
      if (node->Op()->Type() == "while") while_op = node->Op();
      iter = main_block_topo_order.erase(iter);
    } else {
      iter++;
    }
  }
  std::unordered_set<std::string> fixed_vars;
  std::unordered_set<std::string> copy_completed_fixed_vars;
  std::unordered_set<std::string> copy_completed_sched_vars;

  // 这些都是 block 中关于 op 的 params
  // var统计信息，后面的任务我们需要根据限制的显存值来以及这些统计信息来划分
  // fixed layer， sched layer
  unsigned long long total_params_size = 0;  // NOLINT
  std::map<size_t, std::string> op_idx_2_op_name;
  std::map<size_t, std::vector<std::string>> op_idx_2_params_var_names;
  std::map<size_t, std::vector<phi::DenseTensor *>> op_idx_2_params_tensors;
  std::vector<std::pair<size_t, size_t>> op_idx_2_params_size;
  std::map<std::string, phi::DenseTensor *> varname_2_tensor;
  std::map<std::string, size_t> varname_2_varsize;
  std::set<std::string> all_params_vars;

  // 在这个for循环中，我们完成上述信息的初始化
  for (size_t i = 0; i < main_block_topo_order.size(); i++) {
    auto &node = main_block_topo_order[i];
    if (!node->IsOp()) continue;
    op_idx_2_op_name[i] = node->Op()->Type();
    size_t params_size = 0;
    for (auto *var_node : node->inputs) {
      if (!var_node->Var()->Persistable()) continue;
      auto var_name = var_node->Var()->Name();

      auto *var = scope->FindLocalVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var,
                              platform::errors::PreconditionNotMet(
                                  "The var should not be nullptr"));
      if (var->IsType<phi::DenseTensor>()) {
        auto *t = var->GetMutable<phi::DenseTensor>();
        op_idx_2_params_tensors[i].push_back(t);
        op_idx_2_params_var_names[i].push_back(var_name);
        varname_2_tensor[var_name] = t;
        auto var_size = t->numel() * SizeOf(t->dtype());
        varname_2_varsize[var_name] = var_size;
        params_size += var_size;
        if (!all_params_vars.count(var_name)) total_params_size += var_size;
      }
      all_params_vars.insert(var_name);
    }
    if (params_size > 0) {
      op_idx_2_params_size.emplace_back(i, params_size);
    }
  }
  LOG(INFO) << "total_params_size: " << total_params_size;

  // 这部分的逻辑就是根据上述统计得到的信息，划分 fixed layer， sched
  // layer，以及得到显存池 buffer 的大小、需要加载的权重队列、fixed_vars
  size_t buffer_size = 0;
  auto res = DivideLayer(op_idx_2_params_tensors,
                         op_idx_2_params_var_names,
                         op_idx_2_op_name,
                         varname_2_tensor,
                         op_idx_2_params_size,
                         varname_2_varsize,
                         fixed_vars,
                         // 614400000,
                         argument->memory_limit(),
                         argument->fixedlayer_algorithm(),
                         buffer_size);
  LOG(INFO) << "buffer_size: " << buffer_size;
  auto fixed_layers = res.first;
  auto sched_layers = res.second;

  // 将fixed layer对应的 op 的权重数据拷贝到 GPU
  if (fixed_layers.size())
    CopyFixedlyer2Gpu(fixed_layers,
                      op_idx_2_params_var_names,
                      varname_2_tensor,
                      copy_completed_fixed_vars,
                      place);

  // 接下来处理 while op， while op 对应的子block的处理逻辑与主block 完全一致
  if (while_op) {
    auto *sub_block = PADDLE_GET_CONST(framework::BlockDesc *,
                                       while_op->GetAttr("sub_block"));
    LOG(INFO) << "sub_block->ID(): " << sub_block->ID();
    auto while_op_sub_graph = graph.GetSubGraph(sub_block->ID());
    auto while_block_topo_order =
        paddle::framework::ir::TopologySortOperations(*while_op_sub_graph);

    unsigned long long total_params_size = 0;  // NOLINT
    std::map<size_t, std::vector<phi::DenseTensor *>> op_idx_2_params_tensors;
    std::map<size_t, std::vector<std::string>> op_idx_2_params_var_names;
    std::map<size_t, std::string> op_idx_2_op_name;
    std::map<std::string, phi::DenseTensor *> varname_2_tensor;
    std::vector<std::pair<size_t, size_t>> op_idx_2_params_size;
    std::map<std::string, size_t> varname_2_varsize;

    // 在这个 for 循环中，我们完成所有统计信息的初始化
    for (size_t i = 0; i < while_block_topo_order.size(); i++) {
      auto &node = while_block_topo_order[i];
      if (!node->IsOp()) continue;
      op_idx_2_op_name[i] = node->Op()->Type();
      size_t params_size = 0;
      for (auto *var_node : node->inputs) {
        if (!var_node->Var()->Persistable()) continue;
        auto var_name = var_node->Var()->Name();
        auto *var = scope->FindLocalVar(var_name);
        PADDLE_ENFORCE_NOT_NULL(var,
                                platform::errors::PreconditionNotMet(
                                    "The var should not be nullptr"));
        if (var->IsType<phi::DenseTensor>()) {
          auto *t = var->GetMutable<phi::DenseTensor>();
          op_idx_2_params_tensors[i].push_back(t);
          op_idx_2_params_var_names[i].push_back(var_name);
          varname_2_tensor[var_name] = t;
          auto var_size = t->numel() * SizeOf(t->dtype());
          varname_2_varsize[var_name] = var_size;
          params_size += var_size;
          if (!all_params_vars.count(var_name)) total_params_size += var_size;
        }
        all_params_vars.insert(var_name);
      }
      if (params_size > 0) {
        op_idx_2_params_size.emplace_back(i, params_size);
      }
    }
    LOG(INFO) << "total_params_size: " << total_params_size;

    size_t buffer_size = 0;
    auto res = DivideLayer(op_idx_2_params_tensors,
                           op_idx_2_params_var_names,
                           op_idx_2_op_name,
                           varname_2_tensor,
                           op_idx_2_params_size,
                           varname_2_varsize,
                           fixed_vars,
                           argument->memory_limit(),
                           argument->fixedlayer_algorithm(),
                           buffer_size);

    LOG(INFO) << "buffer_size: " << buffer_size;
    auto fixed_layers = res.first;
    auto sched_layers = res.second;

    // 将fixed layer对应的 op 的权重数据拷贝到 GPU
    if (fixed_layers.size())
      CopyFixedlyer2Gpu(fixed_layers,
                        op_idx_2_params_var_names,
                        varname_2_tensor,
                        copy_completed_fixed_vars,
                        place);

    // 将sched layer对应的 op 的权重数据拷贝到 新的 CPU 变量中
    if (sched_layers.size()) {
      auto weight_queue = CopySchedlyer2Cpu(sched_layers,
                                            op_idx_2_params_var_names,
                                            copy_completed_sched_vars,
                                            fixed_vars,
                                            scope,
                                            argument->pin_memory());

      // 初始化主block 对应的显存池：init sched_layers_pool
      paddle::framework::VectorSchedLayersPool::Instance()
          .Get(sub_block->ID())
          .Init(buffer_size, weight_queue, sched_layers);
    }

    LOG(INFO) << "sub block sched_layers: " << sched_layers.size();
    for (auto i : sched_layers) {
      std::cout << i << " ";
    }
    std::cout << std::endl;
  }

  // 最后 将主 block 中 sched layer对应的 op 的权重数据拷贝到 新的 CPU 变量中
  if (sched_layers.size()) {
    auto weight_queue = CopySchedlyer2Cpu(sched_layers,
                                          op_idx_2_params_var_names,
                                          copy_completed_sched_vars,
                                          fixed_vars,
                                          scope,
                                          argument->pin_memory());

    // 初始化主block 对应的显存池：init sched_layers_pool
    paddle::framework::VectorSchedLayersPool::Instance().Get(0).Init(
        buffer_size, weight_queue, sched_layers);
  }

  LOG(INFO) << "main block sched_layers: " << sched_layers.size();
  for (auto i : sched_layers) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
#endif
}

std::string OffLoadParamsPass::repr() const { return "offload-params-pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
