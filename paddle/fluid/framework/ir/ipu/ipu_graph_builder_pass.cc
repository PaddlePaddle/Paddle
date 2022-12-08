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

#include "paddle/fluid/framework/ir/ipu/ipu_graph_builder_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

namespace paddle {
namespace framework {
namespace ir {

void IpuGraphBuilderPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter IpuGraphBuilderPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  std::vector<std::string> feed_list;
  feed_list = Get<std::vector<std::string>>("feed_list");

  std::vector<std::string> fetch_list;
  fetch_list = Get<std::vector<std::string>>("fetch_list");

  auto ipu_backend = platform::ipu::IpuBackend::GetInstance();

  ipu_backend->Compile(graph, feed_list, fetch_list);

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuGraphBuilderPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(ipu_graph_builder_pass,
              paddle::framework::ir::IpuGraphBuilderPass)
    .RequirePassAttr("feed_list")
    .RequirePassAttr("fetch_list");
