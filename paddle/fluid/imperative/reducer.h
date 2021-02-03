// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform

namespace imperative {
class ParallelContext;
class VarBase;
class VariableWrapper;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace imperative {

#if (defined PADDLE_WITH_NCCL) || (defined PADDLE_WITH_XPU_BKCL)
class Group {
 public:
  // Here, we use dense_contents_ & sparse_contents_ to
  // achieve the tensor fuse. When is_sparse_ is true, sparse_contents_ work,
  // conversely, dense_contents_ works. It is mutex relationship.
  framework::Variable dense_contents_;
  framework::Variable* sparse_contents_ = nullptr;
  bool is_sparse_ = false;

  // for concat kernel
  std::vector<framework::Tensor> dense_tensors_;

  std::vector<size_t> length_;

  int64_t all_length_{0};
  // Global indices of participating variables in the group
  std::vector<size_t> variable_indices_;

  // Number of params that haven't been ready. When it is 0, it means
  // the group is ready.
  size_t pending_ = -1;

  // external message of group
  framework::proto::VarType::Type dtype_;

  // context is used to select the stream for concat
  void ConcatTensors(const platform::DeviceContext& context);

  // context is used to select the stream for split
  void SplitTensors(const platform::DeviceContext& context);

  friend std::ostream& operator<<(std::ostream&, const Group&);
};

struct VariableLocator {
  // record the index in groups_
  size_t group_index;
  size_t inside_group_index;
};

class Reducer {
 public:
  explicit Reducer(
      const std::vector<std::shared_ptr<imperative::VarBase>>& vars,
      const std::vector<std::vector<size_t>>& group_indices,
      const std::vector<bool>& is_sparse_gradient,
      std::shared_ptr<imperative::ParallelContext> parallel_ctx,
      const std::vector<size_t>& group_size_limits, bool find_unused_vars);

  virtual ~Reducer() {}

  void InitializeGroups(const std::vector<std::vector<size_t>>& group_indices);

  void InitializeDenseGroups(const std::vector<size_t>& variable_indices_,
                             Group* p_group);

  void PrepareDeps(const std::unordered_set<GradOpNode*>& init_nodes);

  void PrepareForBackward(
      const std::vector<std::shared_ptr<imperative::VarBase>>& outputs);

  void AddDistHook(size_t var_index);

  void MarkVarReady(const size_t var_index, const bool is_used_var);

  void MarkGroupReady(size_t group_index);

  void FinalizeBackward();

  std::vector<std::vector<size_t>> RebuildGruops();

  inline bool NeedRebuildGroup() { return !has_rebuilt_group_; }

 private:
  std::vector<std::shared_ptr<imperative::VarBase>> vars_;
  std::vector<std::vector<size_t>> group_indices_;
  static std::shared_ptr<Reducer> s_instance_;
  std::vector<Group> groups_;
  size_t next_group_ = 0;
  platform::Place place_;
  std::once_flag once_flag_;
  std::vector<bool> is_sparse_gradient_;
  std::shared_ptr<imperative::ParallelContext> parallel_ctx_;
  std::vector<VariableLocator> variable_locators_;

  int nrings_ = 1;

  // Following variables are to help rebuild group
  bool has_rebuilt_group_{false};
  std::vector<std::shared_ptr<imperative::VarBase>> rebuild_vars_;
  std::vector<int64_t> rebuild_var_indices_;
  const std::vector<size_t> group_size_limits_;

  // Following variables are to help unused vars
  std::unordered_map<GradOpNode*, size_t> node_deps_;
  std::unordered_map<VariableWrapper*, size_t> var_index_map_;
  std::vector<size_t> unused_vars_;
  bool has_marked_unused_vars_{false};
  bool find_unused_vars_{false};
  bool all_group_ready_{false};
};

std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>>& tensors,
    const std::vector<bool>& is_sparse_gradient,
    const std::vector<size_t>& group_size_limits,
    const std::vector<int64_t>& tensor_indices = {});
#endif

}  // namespace imperative
}  // namespace paddle
