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

#include "paddle/fluid/framework/process_mesh_desc.h"

namespace paddle {
namespace framework {

int32_t ProcessMeshDesc::next_id = -1;

ProcessMeshDesc::ProcessMeshDesc(const std::vector<int32_t> &topo,
                                 const std::vector<int32_t> &process_group,
                                 int32_t parent_id) {
  int32_t cur_id = ++next_id;
  desc_.set_id(cur_id);
  desc_.set_parent_id(parent_id);
  for (size_t i = 0; i != topo.size(); ++i) {
    desc_.add_topology(topo[i]);
  }
  for (size_t i = 0; i != process_group.size(); ++i) {
    desc_.add_process_group(process_group[i]);
  }
  ProcessMeshDescMap::GetInstance().Insert(cur_id, this);
}

std::vector<int32_t> ProcessMeshDesc::Topology() const {
  size_t size = desc_.topology_size();
  std::vector<int32_t> ret(size);
  for (auto i = 0; i != desc_.topology_size(); ++i) {
    ret[i] = desc_.topology(i);
  }
  return ret;
}

std::vector<int32_t> ProcessMeshDesc::ProcessGroup() const {
  size_t size = desc_.process_group_size();
  std::vector<int32_t> ret(size);
  for (auto i = 0; i != desc_.process_group_size(); ++i) {
    ret[i] = desc_.process_group(i);
  }
  return ret;
}

ProcessMeshDescMap &ProcessMeshDescMap::GetInstance() {
  static ProcessMeshDescMap g_process_mesh_desc_map;
  return g_process_mesh_desc_map;
}

}  // namespace framework
}  // namespace paddle
