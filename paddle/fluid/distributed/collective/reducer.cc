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
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace distributed {

static Backend TransToBackend(platform::Place place) {
  static const std::map<phi::AllocationType, Backend> type_backend = {
      {phi::AllocationType::GPU, Backend::GPU},
      {phi::AllocationType::CPU, Backend::CPU},
  };

  phi::AllocationType type = place.GetType();
  auto it = type_backend.find(type);
  PADDLE_ENFORCE_EQ(it != type_backend.end(),
                    true,
                    platform::errors::InvalidArgument(
                        "Place type (%s) is not supported. ", place));
  return it->second;
}

std::vector<std::vector<size_t>> Eager_AssignGroupBySize(
    const std::vector<Tensor> tensors,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices) {
  PADDLE_ENFORCE_EQ(
      tensors.size(),
      is_sparse_gradient.size(),
      platform::errors::PreconditionNotMet(
          "tensors len must be equal to is_sparse_gradient len, but "
          "[%lu] != [%lu]",
          tensors.size(),
          is_sparse_gradient.size()));
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

  PADDLE_ENFORCE_EQ(true,
                    check_perm(tensor_indices),
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
        group_index.empty(),
        true,
        platform::errors::PreconditionNotMet(
            "AssignGroupBySize construct empty group, please check."));
  }
  if (tensor_indices.empty()) {
    std::sort(res.begin(),
              res.end(),
              [](const std::vector<size_t> &x, const std::vector<size_t> &y) {
                return x.front() < y.front();
              });
  }
  return res;
}

template <typename DeviceContext, typename T>
struct ConcatTensorsForAllReduce {
  void operator()(const DeviceContext &context,
                  const std::vector<phi::DenseTensor> &dense_tensors_,
                  Tensor *p_dense_contents) {
    operators::math::ConcatFunctor<DeviceContext, T> concat_functor_;
    concat_functor_(
        context,
        dense_tensors_,
        0,
        std::dynamic_pointer_cast<phi::DenseTensor>(p_dense_contents->impl())
            .get());
  }
};

template <typename DeviceContext, typename T>
struct SplitTensorsForAllReduce {
  void operator()(const DeviceContext &context,
                  Tensor *p_dense_contents,
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
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE
// note(wangran16): A temporary solution for all backends.
template <typename T>
struct ConcatTensorsForAllReduce<platform::CustomDeviceContext, T> {
  void operator()(const platform::CustomDeviceContext &context,
                  const std::vector<phi::DenseTensor> &dense_tensors_,
                  Tensor *p_dense_contents) {
    phi::DeviceGuard guard(context.GetPlace());
    auto *out =
        std::dynamic_pointer_cast<phi::DenseTensor>(p_dense_contents->impl())
            .get();
    uint8_t *out_data = reinterpret_cast<uint8_t *>(out->data<T>());
    auto *device = phi::DeviceManager::GetDeviceWithPlace(context.GetPlace());

    size_t offset = 0;
    for (const auto &tensor : dense_tensors_) {
      const uint8_t *in_data =
          reinterpret_cast<const uint8_t *>(tensor.data<T>());
      auto sz = tensor.numel() * sizeof(T);
      device->MemoryCopyD2D(out_data + offset, in_data, sz, nullptr);
      offset += sz;
    }
  }
};

template <typename T>
struct SplitTensorsForAllReduce<platform::CustomDeviceContext, T> {
  void operator()(const platform::CustomDeviceContext &context,
                  Tensor *p_dense_contents,
                  std::vector<phi::DenseTensor> *p_dense_tensors) {
    auto *in =
        std::dynamic_pointer_cast<phi::DenseTensor>(p_dense_contents->impl())
            .get();
    uint8_t *in_data = reinterpret_cast<uint8_t *>(in->data<T>());
    auto *device = phi::DeviceManager::GetDeviceWithPlace(context.GetPlace());

    size_t offset = 0;
    for (auto &tensor : *p_dense_tensors) {
      uint8_t *out_data = reinterpret_cast<uint8_t *>(tensor.data<T>());
      auto sz = tensor.numel() * sizeof(T);
      device->MemoryCopyD2D(out_data, in_data + offset, sz, nullptr);
      offset += sz;
    }
  }
};
#endif

// context is used to select the stream for concat
template <typename DeviceContext>
static void ConcatTensorsWithType(
    const DeviceContext &context,
    const std::vector<phi::DenseTensor> &dense_tensors_,
    Tensor *p_dense_contents,
    phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatTensorsForAllReduce<DeviceContext, platform::float16>()(
          context, dense_tensors_, p_dense_contents);
      break;
    case phi::DataType::FLOAT32:
      ConcatTensorsForAllReduce<DeviceContext, float>()(
          context, dense_tensors_, p_dense_contents);
      break;
    case phi::DataType::FLOAT64:
      ConcatTensorsForAllReduce<DeviceContext, double>()(
          context, dense_tensors_, p_dense_contents);
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
      SplitTensorsForAllReduce<DeviceContext, platform::float16>()(
          context, p_dense_contents, p_dense_tensors);
      break;
    case phi::DataType::FLOAT32:
      SplitTensorsForAllReduce<DeviceContext, float>()(
          context, p_dense_contents, p_dense_tensors);
      break;
    case phi::DataType::FLOAT64:
      SplitTensorsForAllReduce<DeviceContext, double>()(
          context, p_dense_contents, p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          type));
  }
}

void EagerGroup::ConcatTensors(const platform::Place &place) {
  dense_contents_ =
      paddle::experimental::empty(IntArray({all_length_}), dtype_, place);

  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto *default_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    ConcatTensorsWithType(
        *default_ctx, dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto *default_ctx = static_cast<platform::CustomDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    ConcatTensorsWithType(
        *default_ctx, dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with "
        "CUSTOM_DEVICE,"
        "Please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    auto *default_ctx = static_cast<phi::CPUContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    ConcatTensorsWithType(
        *default_ctx, dense_tensors_, &dense_contents_, dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Concat grad tensor not supported on place (%s)", place));
  }
}

void EagerGroup::SplitTensors(const platform::Place &place) {
  if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto *default_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    SplitTensorsWithType(
        *default_ctx, &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split grad tensor since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto *default_ctx = static_cast<platform::CustomDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    SplitTensorsWithType(
        *default_ctx, &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split grad tensor since it's not compiled with "
        "CUSTOM_DEVICE,"
        "Please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    auto *default_ctx = static_cast<phi::CPUContext *>(
        platform::DeviceContextPool::Instance().Get(place));
    SplitTensorsWithType(
        *default_ctx, &dense_contents_, &dense_tensors_, dtype_);
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
    const std::vector<size_t> &group_size_limits,
    bool find_unused_parameters)
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
        std::make_shared<egr::CppVoidHook>(reduce_hook));

    gradnode_index_map_[grad_node.get()] = global_var_index;
  }

  vars_marked_ready_.resize(tensors_.size(), false);
  local_used_vars_.resize(tensors_.size(), 0);

  if (find_unused_vars_each_step_) {
    global_used_vars_ = paddle::experimental::empty(
        IntArray({static_cast<int32_t>(tensors_.size())}),
        DataType::INT32,
        inner_place_);
  }
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
        tensor_indices_.size(),
        0,
        platform::errors::PreconditionNotMet(
            "The number of group[%d]'s elements is 0.", group_index));

    EagerGroup group;

    // It's just for check the sparse or dense
    auto first_var = tensors_[tensor_indices_.front()];
    if (tensor_indices_.size() == 1 &&
        is_sparse_gradient_[tensor_indices_.front()]) {
      // process the sparse gradient. one sparse, one group
      group.dtype_ = first_var.dtype();
      group.is_sparse_ = true;
    } else {
      // process the dense gradient.
      InitializeDenseGroups(tensor_indices_, &group);
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

    PADDLE_ENFORCE_EQ(is_sparse_gradient_[tensor_index],
                      false,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s's GRAD must be Tensor, but received "
                          "GRAD is SelectedRows",
                          tensor_name));

    PADDLE_ENFORCE_EQ(tensor.initialized(),
                      true,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s is not initialized.", tensor_name));
    const auto size = tensor.numel();
    PADDLE_ENFORCE_GT(
        size,
        0,
        platform::errors::PreconditionNotMet(
            "The number of tensor %s's elements is 0.", tensor_name));
    all_length += size;

    p_group->length_.push_back(size);

    // for concat operator
    p_group->origin_shapes_.push_back(IntArray(tensor.shape()));
    p_group->dense_tensors_.push_back(phi::DenseTensor());

    const auto &dtype = tensor.dtype();
    const auto &inner_place = tensor.impl()->place();
    if (index > 0) {
      PADDLE_ENFORCE_EQ(dtype,
                        p_group->dtype_,
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has unexpected dtype.", tensor_name));
    } else {
      p_group->dtype_ = dtype;
      inner_place_ = inner_place;
    }
  }
  p_group->all_length_ = all_length;
}

void EagerReducer::TraverseBackwardGraph(const std::vector<Tensor> &outputs) {
  std::queue<egr::GradNodeBase *> queue;
  std::set<egr::GradNodeBase *> visited;

  for (const auto &output : outputs) {
    auto *auto_grad_meta =
        static_cast<egr::AutogradMeta *>(output.get_autograd_meta());
    if (!auto_grad_meta) continue;
    auto shared_grad_node = auto_grad_meta->GetMutableGradNode();
    if (shared_grad_node == nullptr || shared_grad_node.get() == nullptr ||
        auto_grad_meta->StopGradient()) {
      continue;
    }
    egr::GradNodeBase *grad_node = shared_grad_node.get();
    queue.emplace(grad_node);
  }

  while (!queue.empty()) {
    egr::GradNodeBase *node = queue.front();
    queue.pop();
    const paddle::small_vector<std::vector<egr::GradSlotMeta>,
                               egr::kSlotSmallVectorSize> &metas =
        node->OutputMeta();
    for (size_t i = 0; i < metas.size(); i++) {
      for (size_t j = 0; j < metas[i].size(); j++) {
        const egr::Edge &edge = metas[i][j].GetEdge();
        auto next_node_shared = edge.GetMutableGradNode();
        if (!next_node_shared || !next_node_shared.get()) {
          continue;
        }
        auto *next_node = next_node_shared.get();
        const bool was_inserted = visited.insert(next_node).second;
        if (was_inserted) {
          queue.emplace(next_node);
        }
      }
    }
  }

  for (const auto &it : gradnode_index_map_) {
    if (visited.count(it.first) == 0) {
      unused_vars_.push_back(it.second);
      VLOG(3) << "[Rank " << process_group_->GetRank() << "]: "
              << "Tensor " << tensors_[it.second].name() << " at index "
              << it.second << " is marked as unused.";
    }
  }
}

void EagerReducer::PrepareForBackward(const std::vector<Tensor> &outputs) {
  VLOG(3) << "after forward, then reset count for backward.";
  grad_need_hooks_ = true;
  next_group_ = 0;
  std::for_each(groups_.begin(), groups_.end(), [](EagerGroup &group) {
    group.pending_ = group.tensor_indices_.size();
    group.sparse_contents_ = Tensor();
  });

  // reinitialize vars_marked_ready_ for next iteration
  vars_marked_ready_.clear();
  vars_marked_ready_.resize(tensors_.size(), false);

  PADDLE_ENFORCE_EQ(
      groups_need_finalize_,
      false,
      platform::errors::PreconditionNotMet(
          "A serious error has occurred here. Please "
          "set find_unused_parameters=True to traverse backward graph "
          "in each step to prepare reduce in advance. If you have "
          "set, There may be several reasons for this error: "
          "1) Please note that all forward outputs derived from the module "
          "parameters must participate in the calculation of losses and "
          "subsequent gradient calculations. If not, the wrapper will hang, "
          "waiting for autograd to generate gradients for these parameters. "
          "you can use detach or stop_gradient to make the unused parameters "
          "detached from the autograd graph. "
          "2) Used multiple forwards and one backward. You may be able to wrap "
          "multiple forwards in a model."));

  // The first var to trigger the unused parameter
  has_marked_unused_vars_ = false;

  if (find_unused_vars_once_ || find_unused_vars_each_step_) {
    unused_vars_.clear();
    TraverseBackwardGraph(outputs);
    // only check once in first step
    find_unused_vars_once_ = false;
  }

  if (find_unused_vars_each_step_ && unused_vars_.empty()) {
    LOG_FIRST_N(WARNING, 1)
        << "All parameters are involved in the backward pass. "
           "It is recommended to set find_unused_parameters to False "
           "to improve performance. However, if unused parameters "
           "appear in subsequent iterative training, then an error "
           "will occur. Please make it clear that in the subsequent "
           "training, there will be no parameters that are not used "
           "in the backward pass, and then set find_unused_parameters";
  }

  if (unused_vars_.size() == tensors_.size()) {
    LOG_FIRST_N(WARNING, 1)
        << "There is no parameter in the device involved "
           "in the backward calculation. If there are "
           "parameters on other devices involved in the "
           "backward, then a serious error will occur here.";
  }
}

void EagerReducer::AddDistHook(size_t var_index) {
  PADDLE_ENFORCE_LT(var_index,
                    variable_locators_.size(),
                    platform::errors::OutOfRange(
                        "Out of bounds variable index. it must be less"
                        "than %d, but it is %d",
                        variable_locators_.size(),
                        var_index));

  // gradient synchronization is not required when grad_need_hooks_ is false.
  if (!grad_need_hooks_) {
    return;
  }

  VLOG(3) << "Tensor[" << var_index << "] [" << tensors_[var_index].name()
          << "@Grad] arrived and triggered disthook";

  local_used_vars_[var_index] = 1;

  if (!has_marked_unused_vars_) {
    has_marked_unused_vars_ = true;
    for (const auto unused_index : unused_vars_) {
      MarkVarReady(unused_index, false);
    }
  }
  MarkVarReady(var_index, true);
}

void EagerReducer::MarkVarReady(const size_t var_index,
                                const bool is_used_var) {
  VLOG(3) << "Tensor[" << var_index << "][" << tensors_[var_index].name()
          << "] is marked ready.";
  // error happened, if the var is ready before.
  if (vars_marked_ready_[var_index]) {
    auto error_info = string::Sprintf(
        "Error happened, when parameter[%d][%s] has been ready before. "
        "Please set find_unused_parameters=True to traverse backward graph "
        "in each step to prepare reduce in advance. If you have set, "
        "there may be several reasons for this error: "
        "1) In multiple reentrant backward phase, some parameters are reused."
        "2) Using model parameters outside of forward function. Please "
        "make sure that model parameters are not shared in concurrent "
        "forward-backward passes.",
        var_index,
        tensors_[var_index].name());

    PADDLE_ENFORCE_EQ(has_marked_unused_vars_,
                      false,
                      platform::errors::PreconditionNotMet(error_info));

    error_info +=
        "3) Unused parameters retrieval is incorrect. "
        "The return value of forward will be used to retrieve"
        " the unused parameters of the entire model. These "
        "gradients of unused parameters will not be synchronized "
        "between multiple cards. However, if the unused "
        "parameters participate in the backward calculation "
        "again at a later time (e.g. after the forward function, "
        "the loss calculation uses the unused "
        "paramters of the forward and trigger backward), "
        "its gradient will be wrong.";

    PADDLE_ENFORCE_EQ(has_marked_unused_vars_,
                      true,
                      platform::errors::PreconditionNotMet(error_info));
  } else {
    vars_marked_ready_[var_index] = true;
  }
  groups_need_finalize_ = true;

  const auto &var_locator = variable_locators_[var_index];
  const auto group_index = var_locator.group_index;
  const auto inside_group_index = var_locator.inside_group_index;

  auto &group = groups_[group_index];
  auto &group_tensor = group.dense_tensors_[inside_group_index];
  const auto length = group.length_[inside_group_index];

  if (!group.is_sparse_) {
    if (is_used_var) {
      auto *autograd_meta = tensors_[var_index].get_autograd_meta();
      auto &grad_tensor =
          static_cast<egr::AutogradMeta *>(autograd_meta)->Grad();
      group_tensor
          .ShareDataWith(*(
              std::dynamic_pointer_cast<phi::DenseTensor>(grad_tensor.impl())))
          .Resize({grad_tensor.numel()});
    } else {
      // TODO(shenliang03): maybe save the memory by avoiding tensor
      // construction
      if (!group_tensor.initialized()) {
        group_tensor.Resize({static_cast<int64_t>(length)});
        group_tensor.mutable_data(inner_place_, group.dtype_);
      }
      if (HasGrad(var_index)) {
        VLOG(3) << "Tensor[" << tensors_[var_index].name() << "] has grad";
        auto grad_tensor = egr::EagerUtils::mutable_grad(tensors_[var_index]);
        group_tensor
            .ShareDataWith(*(std::dynamic_pointer_cast<phi::DenseTensor>(
                grad_tensor->impl())))
            .Resize({length});
      } else {
        VLOG(3) << "Tensor[" << tensors_[var_index].name()
                << "] doesn't have grad";
        auto *dev_ctx =
            platform::DeviceContextPool::Instance().Get(inner_place_);
        group_tensor.Resize({static_cast<int64_t>(length)});
        phi::funcs::set_constant(*dev_ctx, &group_tensor, 0.0);
      }
    }
  } else {
    auto *autograd_meta = tensors_[var_index].get_autograd_meta();
    auto &grad_tensor = static_cast<egr::AutogradMeta *>(autograd_meta)->Grad();

    // process sparse group
    PADDLE_ENFORCE_EQ(
        HasGrad(var_index),
        true,
        platform::errors::PreconditionNotMet(
            "The sparse parameter[%d][%s] should have gradient. "
            "Currently, DataParallel does not support sparse "
            "parameters without generating gradients during training. "
            "For example, if is_sparese=True is used in Embedding, "
            "the current step of this parameter cannot generate gradient "
            "because of stop_gradient/detatch, where error will occur.",
            var_index,
            tensors_[var_index].name()));

    // need to check tensor type
    PADDLE_ENFORCE_EQ(
        grad_tensor.is_selected_rows(),
        true,
        platform::errors::PreconditionNotMet(
            "The sparse parameter[%d][%s] must have a selectedrows gradient. "
            "Before forward pass, the parameter type is inferred to be "
            "SelectedRows, but after backward pass, its actual type becomes "
            "LodTensor. It is currently not supported by DataParallel. "
            "For example, if sparse embedding is used, and the weight of "
            "embedding is shared with subsequent dense parameters, then "
            "the parameter gradient of the embedding will be converted "
            "to dense parameters.",
            var_index,
            tensors_[var_index].name()));

    group.sparse_contents_.set_impl(grad_tensor.impl());
  }

  if (--group.pending_ == 0) {
    // can start allreduce
    MarkGroupReady(group_index);
  }

  if (next_group_ == groups_.size()) {
    FinalizeBackward();
  }
}

void EagerReducer::MarkGroupReady(size_t group_index) {
  VLOG(3) << "Group[" << group_index << "] is ready";

  PADDLE_ENFORCE_GE(
      group_index,
      next_group_,
      platform::errors::PreconditionNotMet(
          "The index of the incoming group must be greater "
          "than or equal to the previously synchronized group index, "
          "expect it to greater than or equal to %d, but got %d.",
          next_group_,
          group_index));

  if (group_index > next_group_) {
    VLOG(3) << "It will adjust the order of group in next batch automatically";
    return;
  }

  for (; next_group_ < groups_.size() && groups_[next_group_].pending_ == 0;
       ++next_group_) {
    UNUSED auto &group = groups_[next_group_];
    if (group.is_sparse_) {
      AllReduceSparse(&group, next_group_);
    } else {
      FusedAllReduceSchedule(&group, next_group_);
    }
  }
}

bool EagerReducer::HasGrad(size_t var_index) {
  auto grad = egr::EagerUtils::mutable_grad(tensors_[var_index]);
  if (grad && grad->initialized()) {
    return true;
  } else {
    return false;
  }
}

void EagerReducer::ProcessUnusedDenseVars() {
  // The calculation stream must be used here to
  // avoid conflicts with communication.
  VLOG(3) << "Local used vars : "
          << string::join_strings(local_used_vars_, ',');

  const auto *dev_ctx =
      platform::DeviceContextPool::Instance().Get(inner_place_);
  auto *global_used_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(global_used_vars_.impl())
          .get();
  framework::TensorFromVector<int32_t>(
      local_used_vars_, *dev_ctx, global_used_tensor);

  distributed::AllreduceOptions opts;
  opts.reduce_op = ReduceOp::SUM;
  std::vector<Tensor> reduce_tensors = {global_used_vars_};
  std::vector<phi::DenseTensor> in_out;
  for (auto &t : reduce_tensors) {
    in_out.push_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
  }
  process_group_->AllReduce(in_out, in_out, opts)->Synchronize();

  framework::TensorToVector<int>(
      *global_used_tensor, *dev_ctx, &local_used_vars_);
  dev_ctx->Wait();

  // sync compute stream to get global used var message,
  // but maybe affect speed performance
  VLOG(3) << "Global used vars : "
          << string::join_strings(local_used_vars_, ',');

  for (const auto var_index : unused_vars_) {
    const bool global_unused = (local_used_vars_[var_index] == 0);

    // global used but local unused, set grad
    VLOG(3) << "[Rank " << process_group_->GetRank() << "]: "
            << "Var [" << var_index << "] [" << tensors_[var_index].name()
            << "] global_unused: " << global_unused
            << "  has grad: " << HasGrad(var_index);

    if (!global_unused) {
      VLOG(3) << "Set Tensor[" << var_index << "]'s Grad for [Rank "
              << process_group_->GetRank() << "]";
      const auto &var_locator = variable_locators_[var_index];
      const auto group_index = var_locator.group_index;
      const auto &group = groups_[group_index];
      const auto inside_group_index = var_locator.inside_group_index;
      auto &src_tensor = group.dense_tensors_[inside_group_index];

      // sparse no need to check and no support find_unused_parameters
      if (group.is_sparse_) {
        continue;
      }

      // NOTE(haohongxiang): Calling SetFakeEmpty here is to make sure that
      // gradient accumulation can continue normally after clear_gradients()
      // especiall in cases including complex control flow.
      std::static_pointer_cast<egr::GradNodeAccumulation>(
          GetGradNodeFromTensor(&tensors_[var_index]))
          ->SetFakeEmpty(false);

      Tensor grad_value(std::make_shared<phi::DenseTensor>(src_tensor));

      auto dest_var_base = tensors_[var_index];
      auto grad_tensor = egr::EagerUtils::mutable_grad(dest_var_base);
      grad_tensor->copy_(grad_value, inner_place_, true);
      grad_tensor->reshape(dest_var_base.shape());
    }
  }
}

void EagerReducer::FinalizeBackward() {
  groups_need_finalize_ = false;
  grad_need_hooks_ = false;
  for (auto &group : groups_) {
    if (!group.is_sparse_) {
      group.task->Synchronize();
    }
  }

  for (auto &group : groups_) {
    if (!group.is_sparse_) {
      group.SplitTensors(inner_place_);
      group.dense_contents_.reset();
    }
  }

  if (find_unused_vars_each_step_) {
    ProcessUnusedDenseVars();
    local_used_vars_.clear();
    local_used_vars_.resize(tensors_.size(), 0);
    VLOG(3) << "ProcessUnusedDenseVars is finished.";
  }

  VLOG(3) << "In the batch, Reducer is finished.";
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
  paddle::experimental::scale_(
      group->dense_contents_, 1.0 / nranks_, 0.0, false);

  // all_reduce
  std::vector<Tensor> reduce_tensors = {group->dense_contents_};
  std::vector<phi::DenseTensor> in_out;
  for (auto &t : reduce_tensors) {
    in_out.push_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
  }
  group->task = process_group_->AllReduce(in_out, in_out, opts);

  // split in FinalizeBackward()
}

void EagerReducer::AllReduceSparse(EagerGroup *group,
                                   const int curr_group_index) {
  // div nranks
  Tensor sparse_tensor(group->sparse_contents_);
  paddle::experimental::scale_(sparse_tensor, 1.0 / nranks_, 0.0, false);

  VLOG(3) << "sparse_group [" << curr_group_index << "] start allreduce.";

  auto *dev_ctx = platform::DeviceContextPool::Instance().Get(inner_place_);
  if (platform::is_gpu_place(inner_place_)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    dev_ctx = static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(inner_place_));
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_custom_place(inner_place_)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    dev_ctx = static_cast<platform::CustomDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(inner_place_));
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with "
        "CUSTOM_DEVICE,"
        "Please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (platform::is_cpu_place(inner_place_)) {
    dev_ctx = static_cast<phi::CPUContext *>(
        platform::DeviceContextPool::Instance().Get(inner_place_));
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split grad tensor not supported on place (%s)", inner_place_));
  }

  auto src = std::dynamic_pointer_cast<phi::SelectedRows>(
      group->sparse_contents_.impl());
  const auto &src_rows = src->rows();

  const auto &rank_ = process_group_->GetRank();
  const auto &size_ = process_group_->GetSize();

  framework::Vector<int64_t> rows_num_vector(size_);
  rows_num_vector[rank_] = static_cast<int64_t>(src_rows.size());

  Tensor rows_num_tensor = paddle::experimental::empty(
      IntArray({static_cast<int64_t>(size_)}), DataType::INT64, inner_place_);
  auto *rows_num_dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(rows_num_tensor.impl()).get();
  framework::TensorFromVector<int64_t>(
      rows_num_vector, *dev_ctx, rows_num_dense_tensor);

  distributed::AllreduceOptions opts;
  opts.reduce_op = ReduceOp::SUM;
  std::vector<Tensor> reduce_tensors = {rows_num_tensor};
  std::vector<phi::DenseTensor> in_out;
  for (auto &t : reduce_tensors) {
    in_out.push_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
  }
  process_group_->AllReduce(in_out, in_out, opts)->Synchronize();

  framework::TensorToVector<int64_t>(
      *rows_num_dense_tensor, *dev_ctx, &rows_num_vector);
  dev_ctx->Wait();

  const auto *cpu_rows_num_ptr = rows_num_vector.data();
  auto rows_num = std::accumulate(
      cpu_rows_num_ptr, cpu_rows_num_ptr + size_, static_cast<int64_t>(0));

  VLOG(3) << "Gather rows: " << string::join_strings(rows_num_vector, ',')
          << ", total rows number: " << rows_num
          << ", height: " << src->height();

  dev_ctx->Wait();

  Tensor src_value_tensor(std::make_shared<phi::DenseTensor>(src->value()));
  std::vector<int64_t> dst_shape = src_value_tensor.shape();

  if (std::all_of(cpu_rows_num_ptr, cpu_rows_num_ptr + size_, [&](int64_t row) {
        return row == cpu_rows_num_ptr[0];
      })) {
    // During sparse communication, the number of each card is same.
    // allgather is used to speed up the allreduce by replacing broadcast.

    VLOG(3) << "allgather replaces broadcast to speed up in sparse allreduce";

    Tensor dst_rows_tensor =
        paddle::experimental::empty(IntArray({static_cast<int64_t>(rows_num)}),
                                    DataType::INT64,
                                    inner_place_);
    Tensor src_rows_tensor = paddle::experimental::empty(
        IntArray({static_cast<int64_t>((*src).rows().size())}),
        DataType::INT64,
        inner_place_);
    auto *src_rows_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(src_rows_tensor.impl())
            .get();
    framework::TensorFromVector<int64_t>(
        (*src).rows(), *dev_ctx, src_rows_dense_tensor);

    std::vector<Tensor> src_rows_tensors = {src_rows_tensor};
    std::vector<Tensor> dst_rows_tensors = {dst_rows_tensor};
    std::vector<phi::DenseTensor> in;
    std::vector<phi::DenseTensor> out;
    for (auto &t : src_rows_tensors) {
      in.push_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
    }
    for (auto &t : dst_rows_tensors) {
      out.push_back(*std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
    }
    process_group_->AllGather(in, out)->Synchronize();

    framework::Vector<int64_t> dst_rows_vector(rows_num, 0);
    auto *dst_rows_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(dst_rows_tensor.impl())
            .get();
    framework::TensorToVector<int64_t>(
        *dst_rows_dense_tensor, *dev_ctx, &dst_rows_vector);
    dev_ctx->Wait();

    dst_shape[dst_shape.size() - 2] = rows_num;
    auto dst_dense_tensor = std::dynamic_pointer_cast<phi::DenseTensor>(
        paddle::experimental::full(
            IntArray(dst_shape), 0, src_value_tensor.dtype(), inner_place_)
            .impl());

    auto dst =
        std::make_shared<phi::SelectedRows>(dst_rows_vector, (*src).height());
    *(dst->mutable_value()) = *dst_dense_tensor;
    Tensor dst_value_tensor(std::make_shared<phi::DenseTensor>(dst->value()));

    std::vector<Tensor> src_value_tensors = {src_value_tensor};
    std::vector<Tensor> dst_value_tensors = {dst_value_tensor};
    std::vector<phi::DenseTensor> src_dense;
    std::vector<phi::DenseTensor> dst_dense;
    for (auto &t : src_value_tensors) {
      src_dense.push_back(
          *std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
    }
    for (auto &t : dst_value_tensors) {
      dst_dense.push_back(
          *std::dynamic_pointer_cast<phi::DenseTensor>(t.impl()));
    }
    process_group_->AllGather(src_dense, dst_dense)->Synchronize();

    src->set_rows(dst_rows_vector);
    *(src->mutable_value()) =
        *(std::dynamic_pointer_cast<phi::DenseTensor>(dst_value_tensor.impl()));
  } else {
    std::vector<Tensor> rows_tensors;
    std::vector<Tensor> values_tensors;

    for (int i = 0; i < size_; ++i) {
      std::vector<int64_t> value_tensor_shape = {
          cpu_rows_num_ptr[i], dst_shape[dst_shape.size() - 1]};
      Tensor rows_tensor = paddle::experimental::full(
          IntArray({static_cast<int64_t>(cpu_rows_num_ptr[i])}),
          0,
          DataType::INT64,
          inner_place_);
      Tensor values_tensor = paddle::experimental::full(
          IntArray(value_tensor_shape), 0, src->value().dtype(), inner_place_);
      std::vector<phi::DenseTensor> rows_dense_vector;
      std::vector<phi::DenseTensor> values_dense_vector;

      if (i == rank_) {
        auto *rows_dense_tensor =
            std::dynamic_pointer_cast<phi::DenseTensor>(rows_tensor.impl())
                .get();
        framework::TensorFromVector<int64_t>(
            src_rows, *dev_ctx, rows_dense_tensor);
        values_tensor.set_impl(
            std::make_shared<phi::DenseTensor>(src->value()));
      }
      rows_dense_vector.push_back(
          *std::dynamic_pointer_cast<phi::DenseTensor>(rows_tensor.impl()));
      values_dense_vector.push_back(
          *std::dynamic_pointer_cast<phi::DenseTensor>(values_tensor.impl()));

      auto b_opts = BroadcastOptions();
      b_opts.source_rank = i;
      process_group_->Broadcast(rows_dense_vector, rows_dense_vector, b_opts);
      process_group_
          ->Broadcast(values_dense_vector, values_dense_vector, b_opts)
          ->Wait();
      rows_tensors.push_back(rows_tensor);
      values_tensors.push_back(values_tensor);
    }

    Tensor dst_rows_tensor =
        paddle::experimental::concat(rows_tensors, phi::Scalar(0));
    framework::Vector<int64_t> dst_rows_vector(rows_num, 0);
    auto *dst_rows_dense_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(dst_rows_tensor.impl())
            .get();
    framework::TensorToVector<int64_t>(
        *dst_rows_dense_tensor, *dev_ctx, &dst_rows_vector);
    src->set_rows(dst_rows_vector);

    Tensor dst_values_tensor =
        paddle::experimental::concat(values_tensors, phi::Scalar(0));
    *(src->mutable_value()) = *(
        std::dynamic_pointer_cast<phi::DenseTensor>(dst_values_tensor.impl()));
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
