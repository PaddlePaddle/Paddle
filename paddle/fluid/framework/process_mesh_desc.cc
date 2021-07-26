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

ProcessMeshDesc* ProcessMeshDesc::ParentProcessMesh() const {
  auto _map = ProcessMeshDescMap::Instance();
  return &_map.Get(desc_->parent_idx());
}

ProcessMeshDescMap& ProcessMeshDescMap::Instance() {
  static ProcessMeshDescMap g_process_mesh_map;
  return g_process_mesh_map;
}

}  // namespace framework
}  // namespace paddle
