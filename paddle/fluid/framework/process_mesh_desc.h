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

#include "paddle/fluid/framework/proto_desc.h"

namespace paddle {
namespace framework {

class ProcessMeshDesc {
 public:
  ProcessMeshDesc(const std::vector<int32_t> &topo,
                  const std::vector<int32_t> &process_group)
      : topology(topo), process_group(process_group) {}

  int32_t ID() const { return desc_->idx(); }

  int32_t Parent() const { return desc_->parent_idx(); }

  ProcessMeshDesc *ParentProcessMeshDesc() const;
  proto::ProcessMeshDesc *ProcessMeshDesc::Proto();

  std::vector<int32_t> Topology() const { return topology; }
  std::vector<int32_t> ProcessGroup() const { return process_group; }

 private:
  proto::ProcessMeshDesc *desc_;  // not_own
  std::vector<int32_t> topology;
  std::vector<int32_t> process_group;

  DISABLE_COPY_AND_ASSIGN(ProcessMeshDesc);
};

class ProcessMeshDescMap {
 public:
  static ProcessMeshDescMap &Instance();

  bool Has(int32_t index) const { return map_.find(index) != map.end(); }

  void Insert(int32_t index, const ProcessMeshDesc &mesh) {
    PADDLE_ENFORCE_NE(Has(indx), true, platform::errors::AlreadyExists(
                                           "Index (%d) has been used.", index));
    map_.insert({index, mesh});
  }

  const ProcessMeshDesc &Get(int32_t index) const {
    auto it = map_.find(index);
    PADDLE_ENFORCE_NE(it, map_.end(), platform::errors::InvalidArgument(
                                          "Index (%d) does not exist.", index));
    return it->second;
  }

  const std::unordered_map<int32_t, ProcessMeshDesc> &map() const {
    return map_;
  }

 private:
  ProcessMeshDescMap() = default;
  std::unordered_map<int32_t, ProcessMeshDesc> map_;

  DISABLE_COPY_AND_ASSIGN(ProcessMeshDescMap);
};
}  // namespace framework
}  // namespace paddle
