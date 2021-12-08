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

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void InferenceProcessPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter InferenceProcessPass::ApplyImpl";

  // Get a new instance of ipu_backend
  std::shared_ptr<platform::ipu::IpuBackend> ipu_backend =
      platform::ipu::IpuBackend::GetNewInstance();

  // Set scope
  auto& scope = graph->Get<Scope>(kParamScopeAttr);
  ipu_backend->SetScope(scope);

  // Set ipu_strategy
  static std::shared_ptr<platform::ipu::IpuStrategy> ipu_strategy_instance_(
      new platform::ipu::IpuStrategy());
  ipu_strategy_instance_->is_training = false;
  auto num_ipus = graph->Get<int>("num_ipus");
  ipu_strategy_instance_->num_ipus = num_ipus;
  if (num_ipus > 1) {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        platform::ipu::VirtualGraphMode::Manual;
  } else {
    ipu_strategy_instance_->popart_options_.virtualGraphMode =
        platform::ipu::VirtualGraphMode::Off;
  }

  auto enable_pipelining = graph->Get<bool>("enable_pipelining");
  ipu_strategy_instance_->popart_options_.enablePipelining = enable_pipelining;
  if (enable_pipelining) {
    auto batches_per_step = graph->Get<int>("batches_per_step");
    PADDLE_ENFORCE_GE(
        batches_per_step, num_ipus,
        platform::errors::InvalidArgument("Batched per step should be equal or "
                                          "greater than the number of IPUs"));
    ipu_strategy_instance_->batches_per_step = batches_per_step;
  }
  ipu_strategy_instance_->batch_size = graph->Get<int>("batch_size");
  ipu_strategy_instance_->need_avg_shard = graph->Get<bool>("need_avg_shard");

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
        feed_list[BOOST_GET_CONST(int, node->Op()->GetAttr("col"))] =
            node->outputs[0]->Name();
      }
    } else if (node->Name() == "fetch") {
      if (node->IsOp()) {
        fetch_list[BOOST_GET_CONST(int, node->Op()->GetAttr("col"))] =
            node->inputs[0]->Name();
      }
    }
  }

  // Run passes
  std::vector<std::string> graph_pass = {"forward_graph_extract_pass",
                                         "infer_shape_pass", "avg_shard_pass",
                                         "popart_canonicalization_pass"};
  std::vector<std::string> compile_pass = {
      "ipu_inplace_pass", "ipu_graph_builder_pass", "ipu_runtime_replacer_pass",
      "inference_postprocess_pass"};
  for (auto pass_name : graph_pass) {
    auto pass = PassRegistry::Instance().Get(pass_name);
    if (pass_name == "infer_shape_pass") {
      pass->Set("feed_list", new std::vector<std::string>(feed_list.begin(),
                                                          feed_list.end()));
    }
    pass->Apply(graph);
  }

  for (auto pass_name : compile_pass) {
    auto pass = PassRegistry::Instance().Get(pass_name);
    pass->Set("feed_list",
              new std::vector<std::string>(feed_list.begin(), feed_list.end()));
    pass->Set("fetch_list", new std::vector<std::string>(fetch_list.begin(),
                                                         fetch_list.end()));
    pass->Apply(graph);
  }

  VLOG(10) << "leave InferenceProcessPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inference_process_pass,
              paddle::framework::ir::InferenceProcessPass);
