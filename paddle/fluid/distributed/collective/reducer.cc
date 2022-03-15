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

#include "paddle/fluid/distributed/collective/reducer.h"

namespace paddle {
namespace distributed {

std::vector<std::vector<size_t>> Eager_AssignGroupBySize(
    const std::vector<Tensor> tensors,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices) {
  PADDLE_ENFORCE_EQ(
      tensors.size(), is_sparse_gradient.size(),
      platform::errors::PreconditionNotMet(
          "tensors len must be equal to is_sparse_gradient len, but "
          "[%lu] != [%lu]",
          tensors.size(), is_sparse_gradient.size()));
  auto check_perm = [](const std::vector<int64_t> &x) -> bool {
    size_t len = x.size();
    std::vector<size_t> cnt(len, 0);
    for (size_t i = 0; i < len; ++i) {
      if (x[i] >= static_cast<int64_t>(len) || x[i] < 0 || cnt[x[i]]) {
        return false;
      }
      cnt[x[i]]++;
    }
    return true;
  };

  PADDLE_ENFORCE_EQ(true, check_perm(tensor_indices),
                    platform::errors::PreconditionNotMet(
                        "tensor_indices must be a permutation from 0 to %lu",
                        tensor_indices.size()));
  // the return vector
  std::vector<std::vector<size_t>> res;

  // Key: the var type
  // Value: should use which index in group_size_limits for group size limit
  std::map<experimental::DataType, size_t> group_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this group>
  std::map<experimental::DataType, std::pair<std::vector<size_t>, size_t>>
      next_group;

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto &var = tensors[i];

    size_t tensor_real_index = i;
    if (!tensor_indices.empty()) {
      tensor_real_index = tensor_indices[i];
    }

    if (is_sparse_gradient[tensor_real_index]) {
      // we keep sparse var a single group
      res.push_back({tensor_real_index});
      continue;
    }

    const auto &var_dtype = var.dtype();
    VLOG(3) << "var[" << var.name() << "] 's type is " << var_dtype;
    auto &group_info = next_group[var_dtype];

    int64_t var_size = -1;

    if (var.is_dense_tensor()) {
      var_size =
          std::dynamic_pointer_cast<phi::DenseTensor>(var.impl())->numel();
    } else {
      VLOG(3) << "var " << var.name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }

    group_info.first.push_back(tensor_real_index);
    group_info.second += experimental::SizeOf(var_dtype) * var_size;
    // group_info.second += framework::SizeOfType(var_dtype) * var_size;

    if (group_limit_index.find(var_dtype) == group_limit_index.end()) {
      // means it is the first var of var_dtype
      group_limit_index[var_dtype] = 0;
    }
    auto &cur_limit_index = group_limit_index[var_dtype];
    if (group_info.second >= group_size_limits[cur_limit_index]) {
      // exceed group capacity and create a new group
      res.emplace_back(std::move(group_info.first));
      group_info = std::pair<std::vector<size_t>, size_t>();
      cur_limit_index =
          (std::min)(cur_limit_index + 1, group_size_limits.size() - 1);
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
            "AssignGroupBySize construct empty group, please check."));
  }
  if (tensor_indices.empty()) {
    std::sort(res.begin(), res.end(),
              [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
                return x.front() < y.front();
              });
  }
  return res;
}

template <typename DeviceContext, typename T>
static void ConcatTensorsForAllReduce(
    const DeviceContext &context,
    const std::vector<phi::DenseTensor> &dense_tensors_,
    Tensor *p_dense_contents) {
  operators::math::ConcatFunctor<DeviceContext, T> concat_functor_;
  concat_functor_(
      context, dense_tensors_, 0,
      std::dynamic_pointer_cast<phi::DenseTensor>(p_dense_contents->impl())
          .get());
}

template <typename DeviceContext, typename T>
static void SplitTensorsForAllReduce(
    const DeviceContext &context, Tensor *p_dense_contents,
    std::vector<phi::DenseTensor> *p_dense_tensors) {
  auto *in =
      std::dynamic_pointer_cast<phi::DenseTensor>(p_dense_contents->impl())
          .get();
  std::vector<phi::DenseTensor *> outs;
  std::vector<const phi::DenseTensor *> shape_refer;

  outs.reserve(p_dense_tensors->size());
  shape_refer.reserve(p_dense_tensors->size());

  for (auto &tensor : *p_dense_tensors) {
    outs.emplace_back(&tensor);
    shape_refer.emplace_back(&tensor);
  }

  operators::math::SplitFunctor<DeviceContext, T> split_functor_;
  split_functor_(context, *in, shape_refer, 0, &outs);
}

// context is used to select the stream for concat
template <typename DeviceContext>
static void ConcatTensorsWithType(
    const DeviceContext &context,
    const std::vector<phi::DenseTensor> &dense_tensors_,
    Tensor *p_dense_contents, phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatTensorsForAllReduce<DeviceContext, platform::float16>(
          context, dense_tensors_, p_dense_contents);
      break;
    case phi::DataType::FLOAT32:
      ConcatTensorsForAllReduce<DeviceContext, float>(context, dense_tensors_,
                                                      p_dense_contents);
      break;
    case phi::DataType::FLOAT64:
      ConcatTensorsForAllReduce<DeviceContext, double>(context, dense_tensors_,
                                                       p_dense_contents);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors for "
          "allreduce.",
          type));
  }
}

// context is used to select the stream for split
template <typename DeviceContext>
static void SplitTensorsWithType(const DeviceContext &context,
                                 Tensor *p_dense_contents,
                                 std::vector<phi::DenseTensor> *p_dense_tensors,
                                 phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      SplitTensorsForAllReduce<DeviceContext, platform::float16>(
          context, p_dense_contents, p_dense_tensors);
      break;
    case phi::DataType::FLOAT32:
      SplitTensorsForAllReduce<DeviceContext, float>(context, p_dense_contents,
                                                     p_dense_tensors);
      break;
    case phi::DataType::FLOAT64:
      SplitTensorsForAllReduce<DeviceContext, double>(context, p_dense_contents,
                                                      p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          type));
  }
}

void EagerGroup::ConcatTensors(const platform::Place &place) {
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto *default_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    ConcatTensorsWithType(*default_ctx, dense_tensors_, &dense_contents_,
                          dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    auto *default_ctx = static_cast<platform::CPUDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    ConcatTensorsWithType(*default_ctx, dense_tensors_, &dense_contents_,
                          dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Concat grad tensor not supported on place (%s)", place));
  }
}

void EagerGroup::SplitTensors(const platform::Place &place) {
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto *default_ctx = static_cast<platform::CUDADeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    SplitTensorsWithType(*default_ctx, &dense_contents_, &dense_tensors_,
                         dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split grad tensor since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    auto *default_ctx = static_cast<platform::CPUDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    SplitTensorsWithType(*default_ctx, &dense_contents_, &dense_tensors_,
                         dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split grad tensor not supported on place (%s)", place));
  }
}

EagerReducer::EagerReducer(
    const std::vector<Tensor> tensors,
    const std::vector<std::vector<size_t>> &group_indices,
    const std::vector<bool> &is_sparse_gradient,
    std::shared_ptr<distributed::ProcessGroup> process_group,
    const std::vector<size_t> &group_size_limits, bool find_unused_parameters)
    : tensors_(tensors),
      group_indices_(group_indices),
      is_sparse_gradient_(is_sparse_gradient),
      process_group_(process_group),
      group_size_limits_(group_size_limits),
      find_unused_vars_each_step_(find_unused_parameters) {
  VLOG(3) << "Start construct the Reducer ...";

  nranks_ = process_group_->GetSize();

  // initialize groups
  InitializeGroups(group_indices);

  for (size_t global_var_index = 0; global_var_index < tensors_.size();
       ++global_var_index) {
    auto tensor = tensors_[global_var_index];
    auto reduce_hook = [=](void) -> void {
      this->AddDistHook(global_var_index);
    };

    const auto &grad_node = GetGradNodeFromTensor(&tensor);

    PADDLE_ENFORCE(
        grad_node.get() != nullptr,
        paddle::platform::errors::Fatal("Detected NULL grad_node,"
                                        "Leaf tensor should have had grad_node "
                                        "with type: GradNodeAccumulation"));
    const auto &accumulation_grad_node =
        std::dynamic_pointer_cast<egr::GradNodeAccumulation>(grad_node);
    accumulation_grad_node->RegisterReduceHook(
        std::make_shared<egr::CppTensorVoidHook>(reduce_hook));
  }

  vars_marked_ready_.resize(tensors_.size(), false);
  local_used_vars_.resize(tensors_.size(), 0);
}

std::shared_ptr<egr::GradNodeBase> EagerReducer::GetGradNodeFromTensor(
    Tensor *tensor) {
  auto *autograd_meta = tensor->get_autograd_meta();
  const auto &grad_node =
      static_cast<egr::AutogradMeta *>(autograd_meta)->GetMutableGradNode();
  return grad_node;
}

void EagerReducer::InitializeGroups(
    const std::vector<std::vector<size_t>> &group_indices) {
  VLOG(3) << "Start initialize groups ..";

  // clear the group
  groups_.clear();
  groups_.reserve(group_indices.size());

  variable_locators_.clear();
  variable_locators_.resize(tensors_.size());

  auto group_nums = group_indices.size();
  for (size_t group_index = 0; group_index < group_nums; ++group_index) {
    const auto &tensor_indices_ = group_indices[group_index];
    PADDLE_ENFORCE_GT(
        tensor_indices_.size(), 0,
        platform::errors::PreconditionNotMet(
            "The number of group[%d]'s elements is 0.", group_index));

    EagerGroup group;

    // It's just for check the sparse or dense
    auto first_var = tensors_[tensor_indices_.front()];
    if (tensor_indices_.size() == 1 &&
        is_sparse_gradient_[tensor_indices_.front()]) {
      // process the sparse gradient. one sparse, one group
      group.dtype_ = first_var.dtype();
    } else {
      // process the dense gradient.
      InitializeDenseGroups(tensor_indices_, &group);
      experimental::Backend backend;
      switch (inner_place_.GetType()) {
        case phi::AllocationType::GPU:
          backend = experimental::Backend::GPU;
          break;
        case phi::AllocationType::CPU:
          backend = experimental::Backend::CPU;
          break;
        default:
          PADDLE_THROW(platform::errors::Unimplemented(
              "Place type (%s) is not supported. ", inner_place_));
          break;
      }
      group.dense_contents_ = paddle::experimental::empty(
          ScalarArray({group.all_length_}), group.dtype_, backend);
    }

    // map tensors to this group by VariableLocator
    size_t inside_group_index = 0;
    for (const auto var_index : tensor_indices_) {
      TensorLocator tensor_locator;
      tensor_locator.group_index = group_index;
      tensor_locator.inside_group_index = inside_group_index++;
      variable_locators_[var_index] = tensor_locator;
    }
    group.tensor_indices_ = std::move(tensor_indices_);
    groups_.emplace_back(std::move(group));

    VLOG(3) << "The Group[" << group_index << "]:" << groups_.back();
  }
}

void EagerReducer::InitializeDenseGroups(
    const std::vector<size_t> &tensor_indices_, EagerGroup *p_group) {
  VLOG(3) << "InitializeDenseGroups.";
  int64_t all_length = 0;
  for (size_t index = 0; index < tensor_indices_.size(); ++index) {
    auto tensor_index = tensor_indices_[index];
    auto &tensor = tensors_[tensor_index];
    auto &tensor_name = tensor.name();

    PADDLE_ENFORCE_EQ(tensor.is_initialized(), true,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s is not initialized.", tensor_name));
    const auto size = tensor.numel();
    PADDLE_ENFORCE_GT(
        size, 0, platform::errors::PreconditionNotMet(
                     "The number of tensor %s's elements is 0.", tensor_name));
    all_length += size;

    p_group->length_.push_back(size);

    // for concat operator
    p_group->origin_shapes_.push_back(ScalarArray(tensor.shape()));
    p_group->dense_tensors_.push_back(phi::DenseTensor());

    const auto &dtype = tensor.dtype();
    const auto &place = tensor.place();
    const auto &inner_place = tensor.impl()->place();
    if (index > 0) {
      PADDLE_ENFORCE_EQ(dtype, p_group->dtype_,
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has unexpected dtype.", tensor_name));
      PADDLE_ENFORCE_EQ(place, place_,
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different place. Expected place is "
                            "%s, but actual place is %s",
                            tensor_name, inner_place_, inner_place));
    } else {
      p_group->dtype_ = dtype;
      place_ = place;
      inner_place_ = inner_place;
    }
  }
  p_group->all_length_ = all_length;
}

void EagerReducer::PrepareForBackward(const std::vector<Tensor> &outputs) {
  VLOG(3) << "after forward, then reset count for backward.";
  grad_need_hooks_ = true;
  next_group_ = 0;
  std::for_each(groups_.begin(), groups_.end(), [](EagerGroup &group) {
    group.pending_ = group.tensor_indices_.size();
  });

  // reinitialize vars_marked_ready_ for next iteration
  vars_marked_ready_.clear();
  vars_marked_ready_.resize(tensors_.size(), false);
}

void EagerReducer::AddDistHook(size_t var_index) {
  PADDLE_ENFORCE_LT(var_index, variable_locators_.size(),
                    platform::errors::OutOfRange(
                        "Out of bounds variable index. it must be less"
                        "than %d, but it is %d",
                        variable_locators_.size(), var_index));

  // gradient synchronization is not required when grad_need_hooks_ is false.
  if (!grad_need_hooks_) {
    return;
  }

  auto &tensor = tensors_[var_index];
  const auto &grad_node = GetGradNodeFromTensor(&tensor);

  VLOG(3) << "Var[" << var_index << "] [" << (*grad_node).name()
          << "] arrived and triggered disthook";

  local_used_vars_[var_index] = 1;

  MarkVarReady(var_index, true);
}

void EagerReducer::MarkVarReady(const size_t var_index,
                                const bool is_used_var) {
  const auto &var_locator = variable_locators_[var_index];
  const auto group_index = var_locator.group_index;
  const auto inside_group_index = var_locator.inside_group_index;

  auto &group = groups_[group_index];
  auto &group_tensor = group.dense_tensors_[inside_group_index];
  auto *autograd_meta = tensors_[var_index].get_autograd_meta();
  auto &grad_tensor = static_cast<egr::AutogradMeta *>(autograd_meta)->Grad();

  group_tensor
      .ShareDataWith(
          *(std::dynamic_pointer_cast<phi::DenseTensor>(grad_tensor.impl())))
      .Resize({grad_tensor.numel()});

  vars_marked_ready_[var_index] = true;

  if (--group.pending_ == 0) {
    // can start allreduce
    MarkGroupReady(group_index);
  }
}

void EagerReducer::MarkGroupReady(size_t group_index) {
  VLOG(3) << "Group[" << group_index << "] is ready";

  PADDLE_ENFORCE_GE(
      group_index, next_group_,
      platform::errors::PreconditionNotMet(
          "The index of the incoming group must be greater "
          "than or equal to the previously synchronized group index, "
          "expect it to greater than or equal to %d, but got %d.",
          next_group_, group_index));

  if (group_index > next_group_) {
    VLOG(3) << "It will adjust the order of group in next batch automatically";
    return;
  }

  for (; next_group_ < groups_.size() && groups_[next_group_].pending_ == 0;
       ++next_group_) {
    UNUSED auto &group = groups_[next_group_];
    FusedAllReduceSchedule(&group, next_group_);
  }
}

void EagerReducer::FusedAllReduceSchedule(EagerGroup *group,
                                          const int curr_group_index) {
  // The overall timeline: concat > div_nranks > allreduce > split
  distributed::AllreduceOptions opts;
  opts.reduce_op = ReduceOp::SUM;

  VLOG(3) << "group [" << curr_group_index << "] start fused_allreduce.";

  // concat tensors
  group->ConcatTensors(inner_place_);

  // div nranks
  double scaling = 1.0 / nranks_;
  paddle::experimental::scale_(group->dense_contents_, scaling, 0.0, false);

  // all_reduce
  std::vector<Tensor> reduce_tensors = {group->dense_contents_};
  tasks_.push_back(process_group_->AllReduce(reduce_tensors, opts));

  if (tasks_.size() == groups_.size()) {
    for (size_t index = 0; index < tasks_.size(); index++) {
      auto &task = tasks_.back();
      task->Synchronize();
      tasks_.pop_back();
    }
    for (size_t index = 0; index < groups_.size(); index++) {
      auto &group = groups_[index];
      group.SplitTensors(inner_place_);
    }
  }
}

std::ostream &operator<<(std::ostream &out, const EagerGroup &group) {
  const auto &tensors_ = group.tensor_indices_;
  out << "numel: " << group.all_length_ << " ;var number: " << tensors_.size()
      << "\n";
  auto begin = tensors_.begin();
  auto end = tensors_.end();
  out << "[";
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) out << ' ';
    out << *begin;
  }
  if (begin != end) {
    out << " ...";
  }
  out << "]\n";
  return out;
}

}  //  namespace distributed
}  //  namespace paddle
