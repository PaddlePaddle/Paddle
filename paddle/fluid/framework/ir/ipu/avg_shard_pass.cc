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

#include "paddle/fluid/framework/ir/ipu/avg_shard_pass.h"

#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void AvgShardPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter AvgShardPass::ApplyImpl";

  std::shared_ptr<platform::ipu::IpuBackend> ipu_backend =
      platform::ipu::IpuBackend::GetInstance();

  if (ipu_backend->GetIpuStrategy()->need_avg_shard) {
    VLOG(10) << "start AvgShardPass";
    auto nodes = ir::TopologySortOperations(*graph);
    auto num_ipus = ipu_backend->GetIpuStrategy()->num_ipus;

    int shard_position = nodes.size() / num_ipus;
    int index_and_stage = -1;
    for (int i = 0; i < nodes.size(); i++) {
      if ((i % shard_position) == 0 && index_and_stage < num_ipus - 1) {
        index_and_stage++;
      }
      nodes[i]->Op()->SetAttr("ipu_index", index_and_stage);
      nodes[i]->Op()->SetAttr("ipu_stage", index_and_stage);
    }
    VLOG(10) << "end AvgShardPass";
  }

  VLOG(10) << "leave AvgShardPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(avg_shard_pass, paddle::framework::ir::AvgShardPass);
