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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/imperative/all_reduce.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/variable_wrapper.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/math/concat_and_split.h"

#include "paddle/fluid/platform/cuda_resource_pool.h"

namespace paddle {
namespace imperative {

struct Group {
  // Here, we use dense_contents & sparse_contents to
  // achieve the tensor fuse. When is_sparse_ is true, sparse_contents work,
  // conversely, dense_contents works. It is mutex relationship.
  framework::Variable dense_contents;
  framework::Variable* sparse_contents = nullptr;
  bool is_sparse_ = false;

  // for concat kernel
  std::vector<framework::Tensor> dense_tensors;

  std::vector<size_t> offset_;
  std::vector<size_t> length_;
  // Global indices of participating variables in the group
  std::vector<size_t> variable_indices_;

  // Number of params that haven't been ready. When it is 0, it means
  // the group is ready.
  size_t pending = -1;

  // external message of group
  framework::proto::VarType::Type dtype;

  // void ConcatTensorsForAllReduce(){

  // }
};

struct VariableIndex {
  // record the index in groups_
  size_t group_index;
  size_t variable_index;
};

class Reducer {
 public:
  explicit Reducer(
      const std::vector<std::shared_ptr<imperative::VarBase>>& vars,
      const std::vector<std::vector<size_t>>& group_indices,
      const std::vector<bool>& is_sparse_gradient,
      std::shared_ptr<imperative::ParallelContext> parallel_ctx);

  virtual ~Reducer() {}

  void InitializeGroups(const std::vector<std::vector<size_t>>& group_indices);

  void SetGradSpace(Group* p_group);

  void PrepareForBackward();

  void AddDistHook(VariableWrapper* var_warpper,
                   const VariableIndex& var_index);

  void MarkVariableReady(const VariableIndex& var_index,
                         VariableWrapper* var_warpper);

  void MarkGroupReady(size_t group_index);

  void FinalizeBackward();

  // Reducer Singleton
  static std::shared_ptr<Reducer> SetInstance(
      const std::vector<std::shared_ptr<imperative::VarBase>>& vars,
      const std::vector<std::vector<size_t>>& group_indices,
      const std::vector<bool>& is_sparse_gradient,
      std::shared_ptr<imperative::ParallelContext> parallel_ctx) {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::imperative::Reducer(
          vars, group_indices, is_sparse_gradient, parallel_ctx));
    }
    return s_instance_;
  }

  static std::shared_ptr<Reducer> GetInstance() {
    PADDLE_ENFORCE_EQ(
        s_instance_ != NULL, true,
        platform::errors::InvalidArgument("Reducer is not initialized."));
    return s_instance_;
  }

  void ReleaseReducer();

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

  std::vector<std::shared_ptr<platform::CudaEventObject>> events_;
  std::shared_ptr<platform::CudaEventObject> copy_enent_;
  cudaStream_t compute_stream_;
  cudaStream_t comm_stream_;
  paddle::operators::math::ConcatFunctor<paddle::platform::CUDADeviceContext,
                                         float>
      concat_functor_;
  paddle::operators::math::SplitFunctor<paddle::platform::CUDADeviceContext,
                                        float>
      split_functor_;
};

std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>>& tensors,
    const std::vector<bool>& is_sparse_gradient,
    const std::vector<size_t>& group_size_limits);

}  // namespace imperative
}  // namespace paddle
