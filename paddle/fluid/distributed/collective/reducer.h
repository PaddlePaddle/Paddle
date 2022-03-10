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

#pragma once

#include <map>
#include <vector>
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/distributed/collective/ProcessGroupNCCL.h"
#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/utils/hook_utils.h"
#include "paddle/fluid/eager/api/utils/tensor_utils.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace distributed {
using Tensor = paddle::experimental::Tensor;
using Scalar = paddle::experimental::ScalarBase<paddle::experimental::Tensor>;
using ScalarArray =
    paddle::experimental::ScalarArrayBase<paddle::experimental::Tensor>;

std::vector<std::vector<size_t>> Eager_AssignGroupBySize(
    const std::vector<Tensor>, const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices = {});

class EagerGroup {
 public:
  Tensor tensor_;

  // for concat kernel
  std::vector<Tensor> dense_tensors_;
  std::vector<int64_t> length_;
  int64_t all_length_{0};
  std::vector<ScalarArray> origin_shapes_;

  // Global indices of participating tensors in the group
  std::vector<size_t> tensor_indices_;

  // Number of params that haven't been ready. When it is 0, it means
  // the group is ready.
  size_t pending_ = -1;

  // external message of group
  phi::DataType dtype_;

  friend std::ostream &operator<<(std::ostream &, const EagerGroup &);
};

struct TensorLocator {
  // record the index in groups_
  size_t group_index;
  size_t inside_group_index;
};

class EagerReducer {
 public:
  explicit EagerReducer(
      const std::vector<Tensor> tensors,
      const std::vector<std::vector<size_t>> &group_indices,
      const std::vector<bool> &is_sparse_gradient,
      std::shared_ptr<distributed::ProcessGroupNCCL> process_group,
      const std::vector<size_t> &group_size_limits,
      bool find_unused_parameters);

  virtual ~EagerReducer() {}

  std::shared_ptr<egr::GradNodeBase> GetGradNodeFromTensor(Tensor *tensor);

  void InitializeGroups(const std::vector<std::vector<size_t>> &group_indices);
  void InitializeDenseGroups(const std::vector<size_t> &tensor_indices_,
                             EagerGroup *p_group);
  void PrepareForBackward(const std::vector<Tensor> &outputs);
  void AddDistHook(size_t var_index);
  void MarkVarReady(const size_t var_index, const bool is_used_var);
  void MarkGroupReady(const size_t group_index);
  void FusedAllReduceSchedule(EagerGroup *group, const int curr_group_index);

 private:
  std::vector<Tensor> tensors_;
  std::vector<std::vector<size_t>> group_indices_;
  std::vector<bool> is_sparse_gradient_;
  std::shared_ptr<distributed::ProcessGroupNCCL> process_group_;
  std::vector<size_t> group_size_limits_;
  bool find_unused_vars_each_step_;

  std::vector<EagerGroup> groups_;
  std::vector<TensorLocator> variable_locators_;
  PlaceType place_;
  size_t next_group_ = 0;
  int64_t nranks_ = -1;

  bool grad_need_hooks_{false};

  std::vector<bool> vars_marked_ready_;
  std::vector<int> local_used_vars_;
};

}  //  namespace distributed
}  //  namespace paddle
