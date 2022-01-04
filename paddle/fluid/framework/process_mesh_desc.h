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

#pragma once

#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/proto_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

class ProcessMeshDesc {
 public:
  ProcessMeshDesc(const std::vector<int32_t>& topo,
                  const std::vector<int32_t>& process_group, int32_t parent_id);

  int32_t ID() const { return desc_.id(); }
  int32_t Parent() const { return desc_.parent_id(); }

  std::vector<int32_t> Topology() const;
  std::vector<int32_t> ProcessGroup() const;

  static int32_t next_id;

 private:
  proto::ProcessMeshDesc desc_;  // not_own
};

class ProcessMeshDescMap {
 public:
  static ProcessMeshDescMap& GetInstance();

  bool Has(int32_t index) const { return map_.find(index) != map_.end(); }

  void Insert(int32_t index, ProcessMeshDesc* mesh) {
    PADDLE_ENFORCE_NE(
        Has(index), true,
        platform::errors::AlreadyExists("Index (%d) has been used.", index));
    map_.insert(std::make_pair(index, mesh));
  }

 private:
  ProcessMeshDescMap() = default;
  // Use raw pointer to avoid double free
  std::unordered_map<int32_t, ProcessMeshDesc*> map_;
  DISABLE_COPY_AND_ASSIGN(ProcessMeshDescMap);
};
}  // namespace framework
}  // namespace paddle
