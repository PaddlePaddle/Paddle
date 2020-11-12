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

#include "paddle/fluid/imperative/reducer.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/framework/data_type.h"

namespace paddle {
namespace imperative {

std::shared_ptr<Reducer> Reducer::s_instance_ = NULL;

Reducer::Reducer(const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
                 const std::vector<std::vector<size_t>> &group_indices,
                 std::shared_ptr<imperative::ParallelContext> parallel_ctx)
    : vars_(vars), group_indices_(group_indices), parallel_ctx_(parallel_ctx) {
  // parallel_ctx->Print_ParallelStrategy();
  VLOG(3) << "Start construct the Reducer ...";
  // initialize groups
  initialize_groups(group_indices);

  // initialize varname2index_
  {
    for (size_t group_index = 0; group_index < group_indices.size();
         ++group_index) {
      for (size_t var_index = 0; var_index < group_indices[group_index].size();
           ++var_index) {
        size_t index = group_indices[group_index][var_index];
        const std::string &var_name = vars_[index]->GradVarName();
        varname2index_[var_name] = VariableIndex{
            .group_index = group_index, .variable_index = var_index,
        };
      }
    }
  }
}

void Reducer::initialize_groups(
    const std::vector<std::vector<size_t>> &group_indices) {
  VLOG(3) << "Start initialize groups ..";
  // clear the group
  groups_.clear();
  groups_.reserve(group_indices.size());

  auto group_nums = group_indices.size();
  for (size_t group_index = 0; group_index < group_nums; ++group_index) {
    Group group;
    int64_t all_length = 0;
    group.variable_indices_ = group_indices[group_index];
    size_t offset = 0;

    for (size_t index = 0; index < group_indices[group_index].size(); ++index) {
      const auto variable_index = group_indices[group_index][index];
      const auto &var = vars_[variable_index];
      const auto var_name = var->Name();

      // TODO(shenliang03): to process the selectrows
      auto lod_tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();

      PADDLE_ENFORCE_EQ(lod_tensor->IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "Tensor `%s` is not initialized.", var_name));
      auto size = lod_tensor->numel();
      PADDLE_ENFORCE_GT(
          size, 0, platform::errors::InvalidArgument(
                       "The number of tensor `%s`'s elements is 0.", var_name));
      all_length += size;

      group.offset_.push_back(offset);
      group.length_.push_back(size);
      offset += size;

      // check the dtype and place, it must be same.
      auto dtype = var->DataType();
      auto place = var->Place();
      if (index > 0) {
        PADDLE_ENFORCE_EQ(dtype, group.dtype,
                          platform::errors::InvalidArgument(
                              "Tensor `%s` has different dtype.", var_name));
        PADDLE_ENFORCE_EQ(place, place_,
                          platform::errors::InvalidArgument(
                              "Tensor `%s` has different place.", var_name));
      } else {
        group.dtype = dtype;
        place_ = place;
      }
    }

    // Alloc the continuous space
    auto tensor = group.contents.GetMutable<framework::LoDTensor>();
    tensor->Resize(framework::make_ddim({all_length}))
        .mutable_data(place_, group.dtype);

    // Debug Message For Reducer
    VLOG(3) << "the groups_[" << group_index << "] basic message:";
    VLOG(3) << "all_length " << all_length;
    VLOG(3) << "offset:";
    for (auto ele : group.offset_) VLOG(3) << ele;
    VLOG(3) << "length:";
    for (auto ele : group.length_) VLOG(3) << ele;

    groups_.emplace_back(std::move(group));
  }
}

void Reducer::set_grad_space(Group *p_group) {
  const std::vector<size_t> &global_indices = p_group->variable_indices_;
  const auto &offset = p_group->offset_;
  const auto &length = p_group->length_;
  for (size_t index = 0; index < global_indices.size(); ++index) {
    const auto &var = vars_[global_indices[index]];  // varbase of var
    auto &grad_var = var->GradVarBase();             // varbase of var grad
    auto grad_tensor =
        grad_var->MutableVar()->GetMutable<framework::LoDTensor>();
    auto &contents = p_group->contents;

    PADDLE_ENFORCE_EQ(
        contents.IsInitialized(), true,
        platform::errors::PreconditionNotMet("Bucket must be initialized."));
    auto dim = grad_tensor->dims();
    grad_tensor
        ->ShareDataWith(contents.GetMutable<framework::LoDTensor>()->Slice(
            static_cast<int64_t>(offset[index]),
            static_cast<int64_t>(offset[index] + length[index])))
        .Resize(dim);
  }
}

void Reducer::prepare_for_backward() {
  VLOG(3) << "start reseting count..";
  next_group_ = 0;
  for (size_t group_index = 0; group_index < groups_.size(); ++group_index) {
    auto &group = groups_[group_index];
    group.pending = group.variable_indices_.size();
  }
}

void Reducer::add_dist_hook(VariableWrapper *var_warpper) {
  const std::string &var_name = var_warpper->Name();
  if (varname2index_.find(var_name) == varname2index_.end()) {
    VLOG(3) << "This " << var_name << " is not trainable";
    return;
  }

  VariableIndex var_index = varname2index_[var_name];

  auto group_index = var_index.group_index;
  auto &group = groups_[group_index];

  mark_variable_ready(var_index, var_warpper);
  if (--group.pending == 0) {
    // can start allreduce
    mark_group_ready(group_index);
  }

  if (next_group_ == groups_.size()) {
    finalize_backward();
  }
}

void Reducer::mark_variable_ready(const VariableIndex &var_index,
                                  VariableWrapper *var_warpper) {
  auto group_index = var_index.group_index;
  auto variable_index = var_index.variable_index;
  auto &group = groups_[group_index];
  auto offset = group.offset_[variable_index];
  auto length = group.length_[variable_index];
  // auto &contents = group.contents;
  auto contents_tensor = group.contents.GetMutable<framework::LoDTensor>();

  auto tensor = var_warpper->MutableVar()->GetMutable<framework::LoDTensor>();
  const auto &var_dtype = var_warpper->DataType();
  void *src_data = tensor->mutable_data(var_warpper->Place(), var_dtype);

  void *dst_data = contents_tensor
                       ->Slice(static_cast<int64_t>(offset),
                               static_cast<int64_t>(offset + length))
                       .mutable_data(place_, group.dtype);
  // use cal_stream
  auto *cal_stream = static_cast<platform::CUDADeviceContext *>(
                         platform::DeviceContextPool::Instance().Get(place_))
                         ->stream();
  memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place_), dst_data,
               BOOST_GET_CONST(platform::CUDAPlace, var_warpper->Place()),
               src_data, framework::SizeOfType(var_dtype) * length, cal_stream);
}

void Reducer::mark_group_ready(size_t group_index) {
  if (group_index > next_group_) return;
  for (; next_group_ < groups_.size() && groups_[next_group_].pending == 0;
       ++next_group_) {
    parallel_ctx_->SyncCalcStream(place_);
    parallel_ctx_->AllReduce(groups_[next_group_].contents,
                             &(groups_[next_group_].contents));
  }
}

void Reducer::finalize_backward() {
  parallel_ctx_->SyncCommStream(place_);
  VLOG(3) << "set gradient space by group";
  for (auto &group : groups_) {
    set_grad_space(&group);
  }
  VLOG(3) << "finalize_backward is finished...";
}

std::vector<std::vector<size_t>> assign_group_by_size(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<size_t> &group_size_limits) {
  // the return vector
  std::vector<std::vector<size_t>> res;

  // Key: the var type
  // Value: should use which index in group_size_limits for group size limit
  std::unordered_map<std::string, size_t> group_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this group>
  std::unordered_map<std::string, std::pair<std::vector<size_t>, size_t>>
      next_group;

  for (size_t i = 0; i < vars.size(); ++i) {
    const auto &var = vars[i];
    if (var->Var().IsType<framework::SelectedRows>()) {
      // we keep sparse var a single group
      res.push_back({i});
      continue;
    }

    const auto &var_dtype = var->DataType();
    const auto var_dtype_str = framework::DataTypeToString(var_dtype);
    VLOG(3) << "var[" << var->GradVarName() << "] 's type is "
            << var->DataType();
    auto &group_info = next_group[var_dtype_str];
    int64_t var_size = -1;
    if (var->Var().IsType<framework::LoDTensor>()) {
      var_size = var->Var().Get<framework::LoDTensor>().numel();
      VLOG(3) << "dims: " << var->Var().Get<framework::LoDTensor>().dims();
    } else {
      VLOG(3) << "var " << var->Name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }
    VLOG(3) << "var[" << var->GradVarName() << "] 's size is " << var_size;
    group_info.first.push_back(i);
    group_info.second += framework::SizeOfType(var_dtype) * var_size;

    if (group_limit_index.find(var_dtype_str) == group_limit_index.end()) {
      // means it is the first var of var_dtype
      group_limit_index[var_dtype_str] = 0;
    }
    auto &cur_limit_index = group_limit_index[var_dtype_str];
    if (group_info.second >= group_size_limits[cur_limit_index]) {
      // exceed group capacity and create a new group
      res.emplace_back(std::move(group_info.first));
      group_info = std::pair<std::vector<size_t>, size_t>();
      cur_limit_index =
          std::min(cur_limit_index + 1, group_size_limits.size() - 1);
    }
  }

  // add the final groups
  for (auto &e : next_group) {
    auto &group_info = e.second;
    if (!group_info.first.empty()) {
      res.emplace_back(std::move(group_info.first));
    }
  }

  for (const auto &group_index : res) {
    PADDLE_ENFORCE_NE(
        group_index.empty(), true,
        platform::errors::PreconditionNotMet(
            "assign_group_by_size construct empty group, please check"));
  }
  std::sort(res.begin(), res.end(),
            [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
              return x.front() < y.front();
            });
  return res;
}

}  // namespace imperative
}  // namespace paddle
