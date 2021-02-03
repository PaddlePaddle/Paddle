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
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/variable_wrapper.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"

#include "paddle/fluid/imperative/parallel_context.h"

namespace paddle {
namespace imperative {

#if (defined PADDLE_WITH_NCCL) || (defined PADDLE_WITH_XPU_BKCL)
template <typename DeviceContext, typename T>
static void ConcatTensorsForAllReduce(
    const DeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents) {
  operators::math::ConcatFunctor<DeviceContext, T> concat_functor_;
  concat_functor_(context, dense_tensors_, 0,
                  p_dense_contents->GetMutable<framework::LoDTensor>());
}

template <typename DeviceContext, typename T>
static void SplitTensorsForAllReduce(
    const DeviceContext &context, framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors) {
  auto *in = p_dense_contents->GetMutable<framework::LoDTensor>();
  std::vector<framework::Tensor *> outs;
  std::vector<const framework::Tensor *> shape_refer;

  outs.reserve(p_dense_tensors->size());
  shape_refer.reserve(p_dense_tensors->size());

  for (auto &tensor : *p_dense_tensors) {
    outs.emplace_back(&tensor);
    shape_refer.emplace_back(&tensor);
  }
  // Sometimes direct copies will be faster
  if (p_dense_tensors->size() < 10) {
    operators::StridedMemcpyWithAxis0<T>(context, *in, shape_refer, &outs);
  } else {
    operators::math::SplitFunctor<DeviceContext, T> split_functor_;
    split_functor_(context, *in, shape_refer, 0, &outs);
  }
}

// context is used to select the stream for concat
template <typename DeviceContext>
static void ConcatTensorsWithType(
    const DeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP16:
      ConcatTensorsForAllReduce<DeviceContext, platform::float16>(
          context, dense_tensors_, p_dense_contents);
      break;
    case framework::proto::VarType::FP32:
      ConcatTensorsForAllReduce<DeviceContext, float>(context, dense_tensors_,
                                                      p_dense_contents);
      break;
    case framework::proto::VarType::FP64:
      ConcatTensorsForAllReduce<DeviceContext, double>(context, dense_tensors_,
                                                       p_dense_contents);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

// context is used to select the stream for split
template <typename DeviceContext>
static void SplitTensorsWithType(
    const DeviceContext &context, framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP16:
      SplitTensorsForAllReduce<DeviceContext, platform::float16>(
          context, p_dense_contents, p_dense_tensors);
      break;
    case framework::proto::VarType::FP32:
      SplitTensorsForAllReduce<DeviceContext, float>(context, p_dense_contents,
                                                     p_dense_tensors);
      break;
    case framework::proto::VarType::FP64:
      SplitTensorsForAllReduce<DeviceContext, double>(context, p_dense_contents,
                                                      p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

#ifdef PADDLE_WITH_XPU_BKCL
template <>
void SplitTensorsForAllReduce<platform::XPUDeviceContext, float>(
    const platform::XPUDeviceContext &context,
    framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors) {
  auto *in = p_dense_contents->GetMutable<framework::LoDTensor>();
  std::vector<framework::Tensor *> outs;
  std::vector<const framework::Tensor *> shape_refer;

  outs.reserve(p_dense_tensors->size());
  shape_refer.reserve(p_dense_tensors->size());

  for (auto &tensor : *p_dense_tensors) {
    outs.emplace_back(&tensor);
    shape_refer.emplace_back(&tensor);
  }
  operators::math::SplitFunctor<platform::XPUDeviceContext, float>
      split_functor_;
  split_functor_(context, *in, shape_refer, 0, &outs);
}

// context is used to select the stream for concat
template <>
void ConcatTensorsWithType<platform::XPUDeviceContext>(
    const platform::XPUDeviceContext &context,
    const std::vector<framework::Tensor> &dense_tensors_,
    framework::Variable *p_dense_contents,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP32:
      ConcatTensorsForAllReduce<platform::XPUDeviceContext, float>(
          context, dense_tensors_, p_dense_contents);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}

// context is used to select the stream for split
template <>
void SplitTensorsWithType<platform::XPUDeviceContext>(
    const platform::XPUDeviceContext &context,
    framework::Variable *p_dense_contents,
    std::vector<framework::Tensor> *p_dense_tensors,
    framework::proto::VarType::Type type) {
  switch (type) {
    case framework::proto::VarType::FP32:
      SplitTensorsForAllReduce<platform::XPUDeviceContext, float>(
          context, p_dense_contents, p_dense_tensors);
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors for "
          "allreduce.",
          framework::DataTypeToString(type)));
  }
}
#endif

void Group::ConcatTensors(const platform::DeviceContext &context) {
  VLOG(3) << "Before concat, set output tensor size is " << all_length_;
  auto tensor = dense_contents_.GetMutable<framework::LoDTensor>();
  tensor->Resize(framework::make_ddim({all_length_}))
      .mutable_data(context.GetPlace(), dtype_);

  auto place = context.GetPlace();
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_NCCL
    ConcatTensorsWithType(
        static_cast<const platform::CUDADeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat grad tensors since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_BKCL
    ConcatTensorsWithType(
        static_cast<const platform::XPUDeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't concat xpu grads since it's not compiled with BKCL,"
        "Please recompile or reinstall Paddle with BKCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    ConcatTensorsWithType(
        static_cast<const platform::CPUDeviceContext &>(context),
        dense_tensors_, &dense_contents_, dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Concat grad tensor not supported on place (%s)", place));
  }
}

void Group::SplitTensors(const platform::DeviceContext &context) {
  auto place = context.GetPlace();
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_NCCL
    SplitTensorsWithType(
        static_cast<const platform::CUDADeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split grad tensor since it's not compiled with NCCL,"
        "Please recompile or reinstall Paddle with NCCL support."));
#endif
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU_BKCL
    SplitTensorsWithType(
        static_cast<const platform::XPUDeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
#else
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Paddle can't split xpu grad since it's not compiled with BKCL,"
        "Please recompile or reinstall Paddle with BKCL support."));
#endif
  } else if (platform::is_cpu_place(place)) {
    SplitTensorsWithType(
        static_cast<const platform::CPUDeviceContext &>(context),
        &dense_contents_, &dense_tensors_, dtype_);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Split grad tensor not supported on place (%s)", place));
  }
}

std::ostream &operator<<(std::ostream &out, const Group &group) {
  const auto &vars = group.variable_indices_;
  out << "numel: " << group.all_length_ << " ;is_sparse: " << group.is_sparse_
      << " ;var number: " << vars.size() << "\n";
  auto begin = vars.begin();
  auto end = vars.end();
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

Reducer::Reducer(const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
                 const std::vector<std::vector<size_t>> &group_indices,
                 const std::vector<bool> &is_sparse_gradient,
                 std::shared_ptr<imperative::ParallelContext> parallel_ctx,
                 const std::vector<size_t> &group_size_limits,
                 bool find_unused_vars)
    : vars_(vars),
      group_indices_(group_indices),
      is_sparse_gradient_(is_sparse_gradient),
      parallel_ctx_(parallel_ctx),
      group_size_limits_(group_size_limits),
      find_unused_vars_(find_unused_vars) {
  VLOG(3) << "Start construct the Reducer ...";
  nrings_ = parallel_ctx->GetNRings();
  // initialize groups
  InitializeGroups(group_indices);
  for (size_t global_var_index = 0; global_var_index < vars_.size();
       ++global_var_index) {
    auto var = vars_[global_var_index];
    var->SharedVar()->AddGradVarLeafBackwardHook(
        std::unique_ptr<LambdaGradAccumulatorPostHook>(
            new LambdaGradAccumulatorPostHook([=](VariableWrapper *grad) {
              this->AddDistHook(global_var_index);
            })));
    var_index_map_[var->GradVarBase()->SharedVar().get()] = global_var_index;
  }
}

void Reducer::InitializeDenseGroups(
    const std::vector<size_t> &variable_indices_, Group *p_group) {
  int64_t all_length = 0;
  for (size_t index = 0; index < variable_indices_.size(); ++index) {
    const auto variable_index = variable_indices_[index];
    const auto &var = vars_[variable_index];
    const auto var_name = var->Name();
    PADDLE_ENFORCE_EQ(is_sparse_gradient_[variable_index], false,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s's GRAD must be LoDTensor, but received "
                          "GRAD is SelectedRows",
                          var_name));

    auto lod_tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(lod_tensor->IsInitialized(), true,
                      platform::errors::PreconditionNotMet(
                          "Tensor %s is not initialized.", var_name));
    auto size = lod_tensor->numel();
    PADDLE_ENFORCE_GT(
        size, 0, platform::errors::PreconditionNotMet(
                     "The number of tensor %s's elements is 0.", var_name));
    all_length += size;

    p_group->length_.push_back(size);

    // check the dtype and place, it must be same.
    auto dtype = var->DataType();
    auto place = var->Place();
    if (index > 0) {
      PADDLE_ENFORCE_EQ(
          dtype, p_group->dtype_,
          platform::errors::PreconditionNotMet(
              "Tensor %s has different dtype. Expected dtype is %s, but actual "
              "dtype is %s",
              var_name, framework::DataTypeToString(p_group->dtype_),
              framework::DataTypeToString(dtype)));
      PADDLE_ENFORCE_EQ(place, place_,
                        platform::errors::PreconditionNotMet(
                            "Tensor %s has different place. Expected place is "
                            "%s, but actual place is %s",
                            var_name, place_, place));
    } else {
      p_group->dtype_ = dtype;
      place_ = place;
    }
  }
}

// Each parameter will be initialized according to the group information.
// For the sparse parameter, sparse_contents_ in the group directly points
// to the parameter. For dense parameters, first construct an empty Tensor().
// Then specify the actual memory in MarkDenseVarReady.
void Reducer::InitializeGroups(
    const std::vector<std::vector<size_t>> &group_indices) {
  VLOG(3) << "Start initialize groups ..";
  // clear the group
  groups_.clear();
  groups_.reserve(group_indices.size());
  variable_locators_.clear();
  variable_locators_.resize(vars_.size());

  auto group_nums = group_indices.size();
  for (size_t group_index = 0; group_index < group_nums; ++group_index) {
    const auto &variable_indices_ = group_indices[group_index];
    PADDLE_ENFORCE_GT(
        variable_indices_.size(), 0,
        platform::errors::PreconditionNotMet(
            "The number of group[%d]'s elements is 0.", group_index));
    Group group;

    // It's just for check the sparse or dense
    auto first_varbase = vars_[variable_indices_.front()];
    if (variable_indices_.size() == 1 &&
        is_sparse_gradient_[variable_indices_.front()]) {
      // process the sparse gradient. one sparse, one group
      group.dtype_ = first_varbase->DataType();
      group.is_sparse_ = true;
    } else {
      // process the dense gradient.
      InitializeDenseGroups(variable_indices_, &group);
    }

    // map variables to this group by VariableLocator
    size_t inside_group_index = 0;
    for (const auto var_index : variable_indices_) {
      variable_locators_[var_index] = VariableLocator{
          .group_index = group_index,
          .inside_group_index = inside_group_index++,
      };
    }
    group.variable_indices_ = std::move(variable_indices_);
    groups_.emplace_back(std::move(group));
    // Debug Message For Reducer
    VLOG(3) << "The Group[" << group_index << "]:";
    VLOG(3) << groups_.back();
  }
}

void Reducer::PrepareDeps(const std::unordered_set<GradOpNode *> &init_nodes) {
  PADDLE_ENFORCE_EQ(
      node_deps_.empty(), true,
      platform::errors::AlreadyExists("Op deps must be initialized here"));

  std::queue<GradOpNode *> q;
  std::unordered_set<GradOpNode *> visited;

  for (auto pos = init_nodes.begin(); pos != init_nodes.end(); pos++) {
    q.push(*pos);
    visited.insert(*pos);
  }

  while (!q.empty()) {
    auto *cur_node = q.front();
    q.pop();

    for (auto &cur_op : *cur_node) {
      cur_op.EnforceHasInOut();
    }

    const auto &grad_pending_nodes = cur_node->GradPendingNodes();
    for (auto &grad_pending_node : grad_pending_nodes) {
      PADDLE_ENFORCE_NOT_NULL(
          grad_pending_node,
          platform::errors::NotFound("Grad pending node should not be null"));
      ++node_deps_[grad_pending_node.get()];
      if (visited.count(grad_pending_node.get()) == 0) {
        visited.insert(grad_pending_node.get());
        q.push(grad_pending_node.get());
      }
    }
  }
}

// After each batch is calculated, the counter of each group(group.pending_)
// and allreudce sequence counter(next_group_) will be cleaned up again.
void Reducer::PrepareForBackward(
    const std::vector<std::shared_ptr<imperative::VarBase>> &outputs) {
  VLOG(3) << "start reseting count..";
  next_group_ = 0;
  std::for_each(groups_.begin(), groups_.end(), [](Group &group) {
    group.pending_ = group.variable_indices_.size();
    group.all_length_ = 0;
    group.dense_tensors_.clear();
    group.dense_tensors_.reserve(group.pending_);
    group.sparse_contents_ = nullptr;
  });

  PADDLE_ENFORCE_EQ(
      all_group_ready_, false,
      platform::errors::PreconditionNotMet(
          "Please note that all ``forward`` outputs derived from the module "
          "parameters must participate in the calculation of losses and "
          "subsequent gradient calculations. If not, the wrapper will hang, "
          "waiting for autograd to generate gradients for these parameters. "
          "you can use detach or stop_gradient to make the unused parameters "
          "detached from the autograd graph."));

  // The first var to trigger the unused parameter
  has_marked_unused_vars_ = false;
  if (!find_unused_vars_) {
    return;
  }

  // TODO(shenliang03) "find_unused_vars" interface will be exposed in the
  // future to handle control flow to process unused parameters
  find_unused_vars_ = false;

  unused_vars_.clear();
  node_deps_.clear();
  std::queue<std::shared_ptr<GradOpNode>> q;
  std::unordered_set<VariableWrapper *> var_visited;
  std::unordered_set<GradOpNode *> init_nodes;

  for (const auto &output : outputs) {
    const auto &grad_node = output->GradVarBase()->GradNode();
    if (grad_node == nullptr || output->OverridedStopGradient()) {
      VLOG(3) << "Skip auto grad since there is no grad op or output is "
                 "stop_gradient=True: "
              << output->Name();
      continue;
    } else {
      init_nodes.insert(grad_node.get());
      var_visited.insert(output->SharedVar().get());
      q.push(grad_node);
    }
  }

  PrepareDeps(init_nodes);
  // Traverse the autograd graph starting at the specified output
  while (!q.empty()) {
    auto cur_node = q.front();
    q.pop();

    for (const auto &cur_op : *cur_node) {
      cur_op.EnforceHasInOut();
      auto &bwd_outs = cur_op.GetOutsMap();
      for (const auto &pair : bwd_outs) {
        if (!pair.second.IsGrad()) {
          continue;
        }
        for (auto &var : pair.second) {
          if (!var || var->OverridedStopGradient()) {
            continue;
          } else {
            var_visited.insert(var.get());
          }
        }
      }
    }
    for (const auto &grad_pending_node : cur_node->GradPendingNodes()) {
      PADDLE_ENFORCE_NOT_NULL(grad_pending_node,
                              platform::errors::NotFound(
                                  "Grad pending node should not be nullptr"));
      auto iter = node_deps_.find(grad_pending_node.get());
      if (iter == node_deps_.end()) {
        continue;
      }
      if (--(iter->second) == 0) {
        q.push(grad_pending_node);
      }
    }
  }

  for (const auto &it : var_index_map_) {
    if (var_visited.count(it.first) == 0) {
      unused_vars_.push_back(it.second);
      VLOG(3) << "Var[" << it.second << "] [" << it.first->Name()
              << "] is not used";
    }
  }
}

// Add hook function to each leaf node. When the gradient of a leaf node is
// generated, if it is the sparse parameter, it will directly execute allreduce,
// if it is the dense parameter, it will execute three steps: 1,
// MarkDenseVarReady. Find the position of the corresponding group
// through var_index, share the gradient memory and the group dense_tensors,
// the group counter is reduced by 1. 2, MarkGroupReady: When the group
// counter is 0, it means that allreduce can be emitted, and
// concat + allreduce + split is emitted in turn according to next_group_.
// 3, FinalizeBackward: after the end, synchronize each stream.
void Reducer::AddDistHook(size_t var_index) {
  VLOG(3) << "Var[" << var_index << "] ["
          << vars_[var_index]->GradVarBase()->Name()
          << "] arrived and triggered disthook";
  if (!has_marked_unused_vars_) {
    has_marked_unused_vars_ = true;
    for (auto unused_index : unused_vars_) {
      if (NeedRebuildGroup()) {
        rebuild_vars_.push_back(vars_[unused_index]);
        rebuild_var_indices_.push_back(unused_index);
      }
      MarkVarReady(unused_index, false);
    }
  }

  if (NeedRebuildGroup()) {
    rebuild_vars_.push_back(vars_[var_index]);
    rebuild_var_indices_.push_back(var_index);
  }
  MarkVarReady(var_index, true);
}

void Reducer::MarkVarReady(const size_t var_index, const bool is_used_var) {
  all_group_ready_ = true;
  const auto &var_locator = variable_locators_[var_index];
  auto group_index = var_locator.group_index;
  auto &group = groups_[group_index];

  if (is_used_var) {
    auto var_warpper = vars_[var_index]->GradVarBase()->SharedVar();
    if (!group.is_sparse_) {
      auto grad = var_warpper->MutableVar();
      auto inside_group_index = var_locator.inside_group_index;
      auto length = group.length_[inside_group_index];

      auto tensor = grad->GetMutable<framework::LoDTensor>();
      framework::Tensor tmp;
      tmp.ShareDataWith(*tensor).Resize({static_cast<int64_t>(length)});
      group.dense_tensors_.push_back(std::move(tmp));
      group.all_length_ += length;
    } else {
      group.sparse_contents_ = var_warpper->MutableVar();
    }
  }
  if (--group.pending_ == 0) {
    // can start allreduce
    MarkGroupReady(group_index);
  }

  if (next_group_ == groups_.size()) {
    FinalizeBackward();
  }
}

void Reducer::MarkGroupReady(size_t group_index) {
  if (group_index > next_group_) {
    VLOG(3) << "It will adjust the order of group in next batch automatically";
    return;
  }

  for (; next_group_ < groups_.size() && groups_[next_group_].pending_ == 0;
       ++next_group_) {
    auto &group = groups_[next_group_];
    int run_order = next_group_ % nrings_;

    // For CUDA or XPU, compute_stream --> comm_stream.
    // For CPU, do nothing.
    // NOTE. Because concat uses the comm_stream,
    // so we expose WaitCompute() interface and call
    // it here.
    parallel_ctx_->WaitCompute(run_order);

    if (group.is_sparse_) {
      if (group.sparse_contents_ != nullptr) {
        VLOG(3) << "sparse group [" << next_group_
                << "] start allreduce in ring[" << run_order << "]";
        parallel_ctx_->AllReduceByStream(
            *group.sparse_contents_, group.sparse_contents_, run_order, false);
      } else {
        VLOG(3) << "The sparse group[" << next_group_
                << "] has no var to allreduce";
      }
    } else {
      if (!group.dense_tensors_.empty()) {
        VLOG(3) << "dense group [" << next_group_
                << "] start allreduce in ring[" << run_order << "]";
        // Select common commstream to concat tensors
        // group.dense_tensors ---> group.dense_contents_
        group.ConcatTensors(*parallel_ctx_->GetDeviceContext(run_order));

        // Start allreduce
        parallel_ctx_->AllReduceByStream(
            group.dense_contents_, &(group.dense_contents_), run_order, false);

        // Select common commstream to split tensors
        // group.dense_contents_ ---> group.dense_tensors
        group.SplitTensors(*parallel_ctx_->GetDeviceContext(run_order));
      } else {
        VLOG(3) << "The dense group[" << next_group_
                << "] has no var to allreduce";
      }
    }
  }
}

std::vector<std::vector<size_t>> Reducer::RebuildGruops() {
  VLOG(3) << "The order of parameter arrival: "
          << string::join_strings(rebuild_var_indices_, ',');

  PADDLE_ENFORCE_EQ(
      rebuild_vars_.size(), vars_.size(),
      platform::errors::PreconditionNotMet(
          "Rebuild vars's number should be equal to original vars'number, "
          "expect it to be %d, but got %d.",
          vars_.size(), rebuild_vars_.size()));
  std::reverse(rebuild_vars_.begin(), rebuild_vars_.end());
  std::reverse(rebuild_var_indices_.begin(), rebuild_var_indices_.end());
  auto rebuild_group_indices =
      AssignGroupBySize(rebuild_vars_, is_sparse_gradient_, group_size_limits_,
                        rebuild_var_indices_);
  has_rebuilt_group_ = true;
  rebuild_vars_.clear();
  rebuild_var_indices_.clear();
  std::reverse(rebuild_group_indices.begin(), rebuild_group_indices.end());
  return rebuild_group_indices;
}

void Reducer::FinalizeBackward() {
  all_group_ready_ = false;
  // Must prevent compute_stream_ starting until all comm streams have finished
  for (int i = 0; i < nrings_; ++i) {
    parallel_ctx_->WaitComm(i);
  }

  if (NeedRebuildGroup()) {
    VLOG(3) << "Start rebuilding the groups";
    auto rebuild_group_indices = RebuildGruops();
    group_indices_ = std::move(rebuild_group_indices);
    InitializeGroups(group_indices_);
  }

  VLOG(3) << "In the batch, Reducer is finished...";
}

// According to the size of each parameter, it is allocated to different groups.
// The sparse parameter occupies a group exclusively. The dense parameters of
// the same data type are assigned to the same group. When dividing groups, the
// size of each group will be limited according to each value in
// group_size_limits in turn. When it is not enough, it will be divided
// by the last value of group_size_limits. The limit value is 0, which
// means that the parameter will monopolize the group.
std::vector<std::vector<size_t>> AssignGroupBySize(
    const std::vector<std::shared_ptr<imperative::VarBase>> &vars,
    const std::vector<bool> &is_sparse_gradient,
    const std::vector<size_t> &group_size_limits,
    const std::vector<int64_t> &tensor_indices) {
  PADDLE_ENFORCE_EQ(vars.size(), is_sparse_gradient.size(),
                    platform::errors::PreconditionNotMet(
                        "vars len must be equal to is_sparse_gradient len, but "
                        "[%lu] != [%lu]",
                        vars.size(), is_sparse_gradient.size()));
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
  std::unordered_map<std::string, size_t> group_limit_index;

  // Key: the var type
  // Value: <the var index in input tensors, total numel in this group>
  std::unordered_map<std::string, std::pair<std::vector<size_t>, size_t>>
      next_group;

  for (size_t i = 0; i < vars.size(); ++i) {
    const auto &var = vars[i];

    size_t tensor_real_index = i;
    if (!tensor_indices.empty()) {
      tensor_real_index = tensor_indices[i];
    }

    if (is_sparse_gradient[tensor_real_index]) {
      // we keep sparse var a single group
      res.push_back({tensor_real_index});
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
    } else {
      VLOG(3) << "var " << var->Name()
              << " is not tensor or selected_rows, so skip it";
      continue;
    }
    group_info.first.push_back(tensor_real_index);
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
#endif

}  // namespace imperative
}  // namespace paddle
