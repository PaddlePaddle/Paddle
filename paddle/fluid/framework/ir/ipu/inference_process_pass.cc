// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/inference_process_pass.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceProcessPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceProcessPass::ApplyImpl";

  // Get a new instance of ipu_backend
  auto ipu_backend = platform::ipu::IpuBackend::GetInstance();

  // Set scope
  auto& scope = graph->Get<Scope>(kParamScopeAttr);
  ipu_backend->SetScope(scope);

  // Set ipu_strategy
  static std::shared_ptr<platform::ipu::IpuStrategy> ipu_strategy_instance_(
      new platform::ipu::IpuStrategy());
  ipu_strategy_instance_->is_training = false;
  // Set graph replication
  auto replica_num = graph->Get<int>("replica_num");
  if (replica_num > 1) {
    ipu_strategy_instance_->popart_options.enableReplicatedGraphs = true;
    ipu_strategy_instance_->popart_options.replicatedGraphCount = replica_num;
  }
  // Set the num of IPUs
  auto num_ipus = graph->Get<int>("num_ipus");
  // Set sharding
  if (num_ipus > 1) {
    ipu_strategy_instance_->need_avg_shard = true;
    ipu_strategy_instance_->popart_options.virtualGraphMode =
        popart::VirtualGraphMode::Manual;
  } else {
    ipu_strategy_instance_->need_avg_shard = false;
    ipu_strategy_instance_->popart_options.virtualGraphMode =
        popart::VirtualGraphMode::Off;
  }
  // total num IPUs = num_ipus * replica_num
  ipu_strategy_instance_->num_ipus = num_ipus * replica_num;

  // Set micro_batch_size for shape inference
  ipu_strategy_instance_->micro_batch_size =
      graph->Get<int>("micro_batch_size");

  // Set pipelining
  auto enable_pipelining = graph->Get<bool>("enable_pipelining");
  ipu_strategy_instance_->popart_options.enablePipelining = enable_pipelining;
  if (enable_pipelining) {
    auto batches_per_step = graph->Get<int>("batches_per_step");
    PADDLE_ENFORCE_GE(
        batches_per_step,
        num_ipus,
        platform::errors::InvalidArgument("Batched per step should be equal or "
                                          "greater than the number of IPUs"));
    ipu_strategy_instance_->batches_per_step = batches_per_step;
  }

  // Set FP16
  auto enable_fp16 = graph->Get<bool>("enable_fp16");
  ipu_strategy_instance_->enable_fp16 = enable_fp16;
  if (enable_fp16) {
    auto enable_half_partial = graph->Get<bool>("enable_half_partial");
    if (enable_half_partial) {
      ipu_strategy_instance_->popart_options.partialsTypeMatMuls = "half";
    }
  }

  // Set executor
  ipu_strategy_instance_->enable_model_runtime_executor =
      graph->Get<bool>("enable_model_runtime_executor");

  // Set available memory proportion for matmul/conv
  ipu_strategy_instance_->available_memory_proportion =
      graph->Get<float>("available_memory_proportion");

  // Set tiles_per_ipu for IPUMODEL
  ipu_strategy_instance_->tiles_per_ipu = 128;

  // Set Cache path
  auto* ipu_cache_path = getenv("IPU_CACHE_PATH");
  if (ipu_cache_path) {
    ipu_strategy_instance_->popart_options.enableEngineCaching = true;
    ipu_strategy_instance_->popart_options.cachePath =
        std::string{ipu_cache_path};
  }

  // custom ops and patterns
  std::unordered_set<std::string> custom_op_names;
  auto custom_ops_info =
      graph->Get<std::vector<std::vector<std::string>>>("custom_ops_info");
  for (auto custom_op : custom_ops_info) {
    ipu_strategy_instance_->AddCustomOp(
        custom_op[0], custom_op[1], custom_op[2], atoi(custom_op[3].c_str()));
    custom_op_names.insert(custom_op[0]);
  }
  auto patterns =
      graph->Get<std::vector<std::vector<std::string>>>("custom_patterns");
  for (auto pattern : patterns) {
    if (pattern[1] == "True") {
      ipu_strategy_instance_->EnablePattern(pattern[0]);
    } else if (pattern[1] == "False") {
      ipu_strategy_instance_->DisablePattern(pattern[0]);
    }
  }

  ipu_backend->SetIpuStrategy(*(ipu_strategy_instance_.get()));

  // Get feed_list and fetch list
  std::vector<std::string> feed_list = {};
  std::vector<std::string> fetch_list = {};
  for (auto node : graph->Nodes()) {
    if (node->Name() == "feed") {
      if (node->IsOp()) {
        feed_list.push_back("");
      }
    } else if (node->Name() == "fetch") {
      if (node->IsOp()) {
        fetch_list.push_back("");
      }
    }
  }
  for (auto node : graph->Nodes()) {
    if (node->Name() == "feed") {
      if (node->IsOp()) {
        feed_list[PADDLE_GET_CONST(int, node->Op()->GetAttr("col"))] =
            node->outputs[0]->Name();
      }
    } else if (node->Name() == "fetch") {
      if (node->IsOp()) {
        fetch_list[PADDLE_GET_CONST(int, node->Op()->GetAttr("col"))] =
            node->inputs[0]->Name();
      }
    }
  }

  // Run passes
  std::vector<std::string> graph_pass = {"forward_graph_extract_pass",
                                         "infer_shape_pass",
                                         "avg_shard_pass",
                                         "popart_canonicalization_pass",
                                         "inference_dtype_transfer_pass"};
  std::vector<std::string> compile_pass = {"ipu_inplace_pass",
                                           "ipu_graph_builder_pass",
                                           "ipu_runtime_replacer_pass",
                                           "inference_postprocess_pass"};
  for (auto pass_name : graph_pass) {
    auto pass = PassRegistry::Instance().Get(pass_name);
    if (pass_name == "infer_shape_pass") {
      pass->Set(
          "feed_list",
          new std::vector<std::string>(feed_list.begin(), feed_list.end()));
    }
    if (pass_name == "popart_canonicalization_pass") {
      pass->Set("custom_ops",
                new std::unordered_set<std::string>(custom_op_names.begin(),
                                                    custom_op_names.end()));
    }
    pass->Apply(graph);
  }

  for (auto pass_name : compile_pass) {
    auto pass = PassRegistry::Instance().Get(pass_name);
    pass->Set("feed_list",
              new std::vector<std::string>(feed_list.begin(), feed_list.end()));
    pass->Set(
        "fetch_list",
        new std::vector<std::string>(fetch_list.begin(), fetch_list.end()));
    pass->Apply(graph);
  }

  VLOG(10) << "leave InferenceProcessPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_process_pass,
              paddle::framework::ir::InferenceProcessPass);
