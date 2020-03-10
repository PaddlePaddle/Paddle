// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/partial_grad_engine.h"

#include <algorithm>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/gradient_accumulator.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace imperative {

static void GetOpsBetweenTargets(
    std::unordered_set<VariableWrapper *> *input_target_grads,
    std::unordered_set<VarBase *> *output_targets,
    std::unordered_set<const OpBase *> *startup_ops_ptr,
    std::unordered_map<const OpBase *, std::unordered_set<const OpBase *>>
        *pending_ops_ptr,
    std::unordered_map<const OpBase *, size_t> *op_deps_ptr) {
  std::queue<std::pair<const OpBase *, const GradOpNode *>> q;
  std::unordered_set<GradOpNode *> visited;
  for (auto iter = output_targets->begin(); iter != output_targets->end();) {
    auto *output_target = *iter;
    PADDLE_ENFORCE_NOT_NULL(output_target);
    if (output_target->OverridedStopGradient() ||
        output_target->GradVarBase() == nullptr ||
        output_target->GradVarBase()->GradNode() == nullptr) {
      iter = output_targets->erase(iter);
      continue;
    }

    auto &grad_node = output_target->GradVarBase()->GradNode();
    if (visited.count(grad_node.get()) == 0) {
      for (auto &op : *grad_node) {
        q.emplace(&op, grad_node.get());
      }
    }
    ++iter;
  }

  std::unordered_set<VariableWrapper *> found_input_target_grads;
  std::unordered_set<const OpBase *> endpoint_ops;
  std::unordered_map<const OpBase *, std::unordered_set<const OpBase *>>
      preceding_ops;
  while (!q.empty()) {
    auto op_node_pair = q.front();
    q.pop();

    auto *op = op_node_pair.first;
    auto *node = op_node_pair.second;

    for (auto &output_pair : op->GetOutsMap()) {
      if (!output_pair.second.IsGrad()) {
        continue;
      }

      for (auto &out_var : output_pair.second) {
        if (input_target_grads->count(out_var.get()) > 0) {
          VLOG(10) << "Found endpoint op " << op->Type();
          found_input_target_grads.insert(out_var.get());
          endpoint_ops.emplace(op);
        }
      }
    }

    for (auto &pending_node : node->GradPendingNodes()) {
      if (visited.count(pending_node.get()) == 0) {
        for (auto &pending_op : *pending_node) {
          preceding_ops[&pending_op].insert(op);
          q.emplace(&pending_op, pending_node.get());
        }
      }
    }
  }

  *input_target_grads = found_input_target_grads;

  auto &pending_ops = *pending_ops_ptr;
  pending_ops.clear();

  auto &startup_ops = *startup_ops_ptr;
  startup_ops.clear();

  auto &op_deps = *op_deps_ptr;
  op_deps.clear();

  std::unordered_set<VariableWrapper *> target_vars(*input_target_grads);
  std::queue<const OpBase *> op_queue;
  for (auto &endpoint_op : endpoint_ops) {
    op_queue.push(endpoint_op);
    pending_ops[endpoint_op];  // Create empty pending ops
  }

  while (!op_queue.empty()) {
    auto *op = op_queue.front();
    op_queue.pop();

    bool is_valid = false;
    for (auto &output_pair : op->GetOutsMap()) {
      if (!output_pair.second.IsGrad()) {
        continue;
      }

      for (auto &out_var : output_pair.second) {
        if (target_vars.count(out_var.get()) > 0) {
          is_valid = true;
          break;
        }
      }

      if (is_valid) {
        break;
      }
    }

    if (!is_valid) {
      continue;
    }

    for (auto &input_pair : op->GetInsMap()) {
      if (!input_pair.second.IsGrad()) {
        continue;
      }

      for (auto &in_var : input_pair.second) {
        target_vars.insert(in_var.get());
      }
    }

    auto iter = preceding_ops.find(op);
    if (iter != preceding_ops.end()) {
      for (auto &preceding_op : iter->second) {
        pending_ops[preceding_op].insert(op);
        ++op_deps[op];
        op_queue.push(preceding_op);
      }
    } else {
      startup_ops.insert(op);
    }
  }

  for (auto iter = output_targets->begin(); iter != output_targets->end();) {
    auto &grad_node = (*iter)->GradVarBase()->GradNode();
    bool is_valid = std::find_if(grad_node->begin(), grad_node->end(),
                                 [&](const OpBase &op) {
                                   return startup_ops.count(&op) > 0;
                                 }) != grad_node->end();
    if (is_valid) {
      ++iter;
    } else {
      iter = output_targets->erase(iter);
    }
  }
}

static std::string GradOpTypes(const GradOpNode &node) {
  std::vector<std::string> node_types;
  for (auto &op : node) {
    node_types.emplace_back(op.Type());
  }
  return string::join_strings(node_types, ',');
}

static std::string GradOpTypes(const VarBase &var) {
  if (!var.GradVarBase() || !var.GradVarBase()->GradNode()) {
    return "";
  } else {
    return GradOpTypes(*(var.GradVarBase()->GradNode()));
  }
}

static std::string GradPendingOpTypes(const GradOpNode &node) {
  std::vector<std::string> node_types;
  for (auto &n : node.GradPendingNodes()) {
    node_types.emplace_back(GradOpTypes(*n));
  }
  return string::join_strings(node_types, ',');
}

class GradientAccumulationInfo {
 private:
  using PartialGradTraceIdPair =
      std::pair<std::weak_ptr<VariableWrapper>, size_t>;

 public:
  explicit GradientAccumulationInfo(const std::shared_ptr<VariableWrapper> &var,
                                    bool sort_gradient, bool create_graph)
      : mapped_grad_var_(var.get()),
        sort_gradient_(sort_gradient),
        create_graph_(create_graph) {}

  void IncreaseTotalRefCnt() {
    ++total_ref_cnt_;

    if (total_ref_cnt_ > 1) {
      if (!grad_var_) {
        grad_var_ = std::make_shared<VarBase>(true, mapped_grad_var_->Name());
        grad_var_->SetOverridedStopGradient(false);
        if (sort_gradient_) {
          accumulator_.reset(
              new SortedGradientAccumulator(grad_var_->SharedVar().get()));
        } else {
          accumulator_.reset(
              new EagerGradientAccumulator(grad_var_->SharedVar().get()));
        }
        accumulator_->IncreaseRefCnt();
      }
      accumulator_->IncreaseRefCnt();
    }
  }

  size_t TotalRefCnt() { return total_ref_cnt_; }

  const std::shared_ptr<VarBase> &GradVarBase() const { return grad_var_; }

  std::shared_ptr<VariableWrapper> GradVar() const {
    return grad_var_ == nullptr ? nullptr : grad_var_->SharedVar();
  }

  VariableWrapper *MappedGradVar() { return mapped_grad_var_; }

  std::vector<std::shared_ptr<VariableWrapper>> SumGradient(
      std::shared_ptr<VariableWrapper> grad_var_partial, size_t trace_id,
      bool *is_finished) {
    PADDLE_ENFORCE_NOT_NULL(grad_var_partial);
    VLOG(10) << "Sum begins";
    PADDLE_ENFORCE_GT(total_ref_cnt_, 1);
    ++cur_ref_cnt_;
    *is_finished = (cur_ref_cnt_ == total_ref_cnt_);
    PADDLE_ENFORCE_NOT_NULL(accumulator_);

    PADDLE_ENFORCE_LE(cur_ref_cnt_, total_ref_cnt_);
    accumulator_->Add(grad_var_partial, trace_id);

    if (create_graph_) {
      VLOG(10) << "Store partial grad when create_graph = True";
      partial_grads_.emplace_back(grad_var_partial, trace_id);
    }

    if (!(*is_finished) || !create_graph_) {
      VLOG(10) << "Return empty";
      return {};
    }

    if (sort_gradient_) {
      std::sort(partial_grads_.begin(), partial_grads_.end(),
                [](const PartialGradTraceIdPair &p1,
                   const PartialGradTraceIdPair &p2) {
                  return p1.second > p2.second;
                });
    }

    std::vector<std::shared_ptr<VariableWrapper>> result;
    result.reserve(partial_grads_.size());
    for (auto &pair : partial_grads_) {
      if (auto var = pair.first.lock()) {
        result.emplace_back(var);
      }
    }

    return result;
  }

 private:
  std::shared_ptr<VarBase> grad_var_;
  VariableWrapper *mapped_grad_var_;
  std::unique_ptr<GradientAccumulator> accumulator_;
  std::vector<PartialGradTraceIdPair> partial_grads_;
  size_t total_ref_cnt_{0};
  size_t cur_ref_cnt_{0};
  bool sort_gradient_;
  bool create_graph_;
};

static void FillConstantLike(const VariableWrapper &ref_var,
                             VariableWrapper *dst_var,
                             const platform::Place &place, float value) {
  auto &ref_tensor = ref_var.Var().Get<framework::LoDTensor>();
  auto *dst_tensor = dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
  auto *dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  dst_tensor->Resize(ref_tensor.dims());
  dst_tensor->mutable_data(place, ref_var.DataType());
  operators::math::set_constant(*dev_ctx, dst_tensor, value);
}

class ReadyGradVarInfoMap {
 private:
  struct ReadyVarInfo {
    std::shared_ptr<VarBase> var;
    size_t cur_ref_cnt{0};
    size_t total_ref_cnt{0};
  };

 public:
  void IncreaseRefCnt(const VariableWrapper *var) {
    ++(vars_[var].total_ref_cnt);
  }

  std::shared_ptr<VarBase> Get(const VariableWrapper *var,
                               const platform::Place &place, bool *is_last) {
    auto iter = vars_.find(var);
    PADDLE_ENFORCE_EQ(iter != vars_.end(), true);
    auto &ready_var = iter->second;
    PADDLE_ENFORCE_LT(ready_var.cur_ref_cnt, ready_var.total_ref_cnt);

    if (ready_var.var == nullptr && ready_var.cur_ref_cnt == 0) {
      ready_var.var = std::make_shared<VarBase>(var->Name());
      VLOG(10) << "Fill zero for " << var->Name();
      FillConstantLike(*var, ready_var.var->SharedVar().get(), place, 0.0f);
    } else {
      PADDLE_ENFORCE_NOT_NULL(ready_var.var);
    }

    if (++ready_var.cur_ref_cnt == ready_var.total_ref_cnt) {
      *is_last = true;
      return std::move(ready_var.var);  // move to set ready_var.var to nullptr
    } else {
      *is_last = false;
      return ready_var.var;
    }
  }

  bool Set(const VariableWrapper *mapped_var,
           const std::shared_ptr<VarBase> &var) {
    PADDLE_ENFORCE_NOT_NULL(var);
    {
      auto target_iter = target_vars_.find(mapped_var);
      if (target_iter != target_vars_.end()) {
        PADDLE_ENFORCE_EQ(target_iter->second, nullptr);
        target_iter->second = var;
      }
    }

    auto iter = vars_.find(mapped_var);
    if (iter != vars_.end()) {  // This var is ready for next op's input
      auto &ready_var = iter->second;
      PADDLE_ENFORCE_EQ(ready_var.var, nullptr);
      PADDLE_ENFORCE_EQ(ready_var.cur_ref_cnt, 0);
      ready_var.var = var;
      return true;
    } else {
      VLOG(10) << "Do not record " << mapped_var->Name()
               << " because it is not input of any following ops";
      return false;
    }
  }

  void Clear() {
    vars_.clear();
    target_vars_.clear();
  }

  void SetTarget(const VariableWrapper *var) {
    PADDLE_ENFORCE_EQ(target_vars_[var], nullptr);
  }

  const std::shared_ptr<VarBase> &GetTarget(const VariableWrapper *var) const {
    auto iter = target_vars_.find(var);
    PADDLE_ENFORCE_EQ(iter != target_vars_.end(), true);
    PADDLE_ENFORCE_NOT_NULL(iter->second);
    return iter->second;
  }

 private:
  std::unordered_map<const VariableWrapper *, ReadyVarInfo> vars_;
  std::unordered_map<const VariableWrapper *, std::shared_ptr<VarBase>>
      target_vars_;
};

class PartialGradTask {
 public:
  PartialGradTask(const std::vector<std::shared_ptr<VarBase>> &input_targets,
                  const std::vector<std::shared_ptr<VarBase>> &output_targets,
                  const std::vector<std::shared_ptr<VarBase>> &output_grads,
                  const platform::Place &place,
                  const detail::BackwardStrategy &strategy, bool create_graph);

  std::vector<std::shared_ptr<VarBase>> Run();

 private:
  void RunEachOp(const OpBase *op);

  void PrepareInitialReadyVarsMap(const OpBase *op);

  void PrepareInitialGradientAccumulators(const OpBase *op);

  std::vector<std::shared_ptr<VarBase>> CreateResult();

 private:
  std::unordered_set<const OpBase *> startup_ops_;
  std::unordered_map<const OpBase *, std::unordered_set<const OpBase *>>
      pending_ops_;
  std::unordered_map<const OpBase *, size_t> op_deps_;

  ReadyGradVarInfoMap ready_grad_vars_;

  std::unordered_map<VariableWrapper *,
                     std::unique_ptr<GradientAccumulationInfo>>
      grad_accumulators_;

  std::vector<std::shared_ptr<GradOpNode>> double_grad_nodes_;

  // Input targets that are reachable
  std::vector<std::shared_ptr<VarBase>> input_targets_;
  std::unordered_set<VariableWrapper *> input_target_grads_;

  platform::Place place_;
  bool create_graph_;
  detail::BackwardStrategy strategy_;
};

PartialGradTask::PartialGradTask(
    const std::vector<std::shared_ptr<VarBase>> &input_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_grads,
    const platform::Place &place, const detail::BackwardStrategy &strategy,
    bool create_graph) {
  input_targets_ = input_targets;
  place_ = place;
  create_graph_ = create_graph;
  strategy_ = strategy;

  PADDLE_ENFORCE_EQ(input_targets.empty(), false);
  PADDLE_ENFORCE_EQ(output_targets.empty(), false);

  std::unordered_set<VarBase *> out_set;
  for (auto &output : output_targets) {
    PADDLE_ENFORCE_NOT_NULL(output);
    PADDLE_ENFORCE_NOT_NULL(output->GradVarBase());
    PADDLE_ENFORCE_EQ(out_set.count(output.get()), 0);
    out_set.insert(output.get());
  }

  std::unordered_set<VarBase *> in_set;
  std::unordered_set<VariableWrapper *> one_grad_vars;
  for (auto &input : input_targets) {
    PADDLE_ENFORCE_NOT_NULL(input);
    PADDLE_ENFORCE_NOT_NULL(input->GradVarBase());
    PADDLE_ENFORCE_EQ(in_set.count(input.get()), 0);
    in_set.insert(input.get());
    input_target_grads_.insert(input->GradVarBase()->SharedVar().get());
    if (out_set.count(input.get()) > 0) {
      one_grad_vars.insert(input->GradVarBase()->SharedVar().get());
    }
  }

  GetOpsBetweenTargets(&input_target_grads_, &out_set, &startup_ops_,
                       &pending_ops_, &op_deps_);

  for (auto &op_pair : pending_ops_) {
    auto *op = op_pair.first;
    PrepareInitialReadyVarsMap(op);
    PrepareInitialGradientAccumulators(op);
  }

  for (auto &input_grad : input_target_grads_) {
    ready_grad_vars_.SetTarget(input_grad);
  }

  for (auto &one_grad : one_grad_vars) {
    VLOG(10) << "Add one target " << one_grad->Name();
    input_target_grads_.insert(one_grad);
    ready_grad_vars_.SetTarget(one_grad);
  }

  VLOG(10) << "Valid op number " << pending_ops_.size();

  if (!output_grads.empty()) {
    PADDLE_ENFORCE_EQ(output_targets.size(), output_grads.size());
  }

  for (size_t i = 0; i < output_targets.size(); ++i) {
    auto *mapped_out_grad_var =
        output_targets[i]->GradVarBase()->SharedVar().get();

    if (out_set.count(output_targets[i].get()) == 0 &&
        one_grad_vars.count(mapped_out_grad_var) == 0) {
      VLOG(10) << mapped_out_grad_var->Name() << " should be None";
      continue;
    }

    std::shared_ptr<VariableWrapper> out_grad_var;
    if (output_grads.empty() || output_grads[i] == nullptr) {
      out_grad_var = std::make_shared<VariableWrapper>(
          framework::GradVarName(output_targets[i]->Name()));
      FillConstantLike(*(output_targets[i]->SharedVar()), out_grad_var.get(),
                       place_, 1.0f);
    } else {
      const auto &out_tensor =
          output_targets[i]->Var().Get<framework::LoDTensor>();
      const auto &grad_tensor =
          output_grads[i]->Var().Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(grad_tensor.dims(), out_tensor.dims());
      PADDLE_ENFORCE_EQ(grad_tensor.type(), out_tensor.type());
      out_grad_var = output_grads[i]->SharedVar();
    }

    out_grad_var->SetOverridedStopGradient(false);
    auto grad_accumulator_iter = grad_accumulators_.find(mapped_out_grad_var);
    if (grad_accumulator_iter == grad_accumulators_.end()) {
      ready_grad_vars_.Set(mapped_out_grad_var,
                           std::make_shared<VarBase>(false, out_grad_var));
      VLOG(10) << "Fill 1.0f for " << out_grad_var->Name();
    } else {
      auto &accumulator = grad_accumulator_iter->second;
      accumulator->IncreaseTotalRefCnt();
      bool is_finished = false;
      accumulator->SumGradient(out_grad_var, 0, &is_finished);
      PADDLE_ENFORCE_EQ(is_finished, false);
      VLOG(10) << "Add 1.0f to accumulator of " << out_grad_var->Name() << " "
               << accumulator->GradVarBase().get();
    }
  }
}

std::vector<std::shared_ptr<VarBase>> PartialGradTask::Run() {
  VLOG(10) << "Startup op number " << startup_ops_.size();
  std::queue<const OpBase *> q;
  for (auto *op : startup_ops_) {
    q.push(op);
  }

  while (!q.empty()) {
    auto *op = q.front();
    q.pop();
    VLOG(10) << "Start to run " << op->Type();
    RunEachOp(op);
    VLOG(10) << "End to run " << op->Type();

    auto iter = pending_ops_.find(op);
    if (iter == pending_ops_.end()) {
      VLOG(10) << "Finish running because " << op->Type()
               << " has no pending ops";
      continue;
    }

    for (auto &pending_op : iter->second) {
      auto dep_iter = op_deps_.find(pending_op);
      PADDLE_ENFORCE_EQ(dep_iter != op_deps_.end(), true);
      if (--(dep_iter->second) == 0) {
        q.push(pending_op);
      }
    }
  }

  return CreateResult();
}

void PartialGradTask::RunEachOp(const OpBase *op) {
  // Prepare new inputs
  VLOG(10) << "Preparing input of " << op->Type();
  NameVarMap<VarBase> tmp_ins;
  for (auto &input_pair : op->GetInsMap()) {
    auto &new_inputs = tmp_ins[input_pair.first];
    new_inputs.reserve(input_pair.second.size());

    if (!input_pair.second.IsGrad()) {
      for (auto &fwd_var : input_pair.second) {
        if (fwd_var) {
          // unpack forward var
          VLOG(10) << "Unpacking forward var " << fwd_var->Name();
          new_inputs.emplace_back(new VarBase(true, fwd_var));
          VLOG(10) << "Unpacked forward var " << fwd_var->Name()
                   << ", grad ops: " << GradOpTypes(*new_inputs.back());
        } else {
          new_inputs.emplace_back();
        }
      }
    } else {
      for (auto &grad_var : input_pair.second) {
        if (grad_var) {
          bool is_last;
          VLOG(10) << "Getting ready grad var " << grad_var->Name();
          new_inputs.emplace_back(
              ready_grad_vars_.Get(grad_var.get(), op->place(), &is_last));
          VLOG(10) << "Got ready grad var " << grad_var->Name() << " "
                   << new_inputs.back().get();
        } else {
          new_inputs.emplace_back();
        }
      }
    }
  }

  VLOG(10) << "Prepared input of " << op->Type();

  VLOG(10) << "Preparing output of " << op->Type();
  // Prepare new outputs
  NameVarMap<VarBase> tmp_outs;
  std::unordered_map<GradientAccumulationInfo *,
                     std::vector<std::shared_ptr<VariableWrapper>>>
      grads_to_accumulate;
  for (auto &output_pair : op->GetOutsMap()) {
    auto &new_outputs = tmp_outs[output_pair.first];
    if (!output_pair.second.IsGrad()) {
      for (auto &fwd_var : output_pair.second) {
        // unpack forward var
        if (fwd_var) {
          VLOG(10) << "Unpacking forward var " << fwd_var->Name();
          new_outputs.emplace_back(new VarBase(true, fwd_var));
          VLOG(10) << "Unpacked forward var " << fwd_var->Name();
        } else {
          new_outputs.emplace_back();
        }
      }
    } else {
      for (auto &grad_var : output_pair.second) {
        if (grad_var) {
          VLOG(10) << "Creating output grad var " << grad_var->Name();
          auto new_grad_var_iter = grad_accumulators_.find(grad_var.get());
          PADDLE_ENFORCE_EQ(new_grad_var_iter != grad_accumulators_.end(), true,
                            platform::errors::Fatal(
                                "Cannot find gradient accumulator of %s %p",
                                grad_var->Name(), grad_var.get()));

          auto new_grad_var = std::make_shared<VarBase>(true, grad_var->Name());
          new_grad_var->SetOverridedStopGradient(false);
          if (new_grad_var_iter->second->TotalRefCnt() > 1) {
            grads_to_accumulate[new_grad_var_iter->second.get()].emplace_back(
                new_grad_var->SharedVar());
          } else {
            PADDLE_ENFORCE_EQ(new_grad_var_iter->second->GradVar(), nullptr);
            grad_accumulators_.erase(new_grad_var_iter);
            ready_grad_vars_.Set(grad_var.get(), new_grad_var);
          }
          VLOG(10) << "Created output grad var " << grad_var->Name();
          new_outputs.emplace_back(std::move(new_grad_var));
        } else {
          new_outputs.emplace_back();
        }
      }
    }
  }

  VLOG(10) << "Prepared output of " << op->Type();

  // Run op
  VLOG(10) << "Really run op " << op->Type();
  OpBase::Run(op->InnerOp(), tmp_ins, tmp_outs, op->Attrs(), op->place());
  VLOG(10) << "Really run op " << op->Type() << " done";

  if (create_graph_) {
    VLOG(10) << "Prepare to create double grad op";
    auto double_grad_node = CreateGradOpNode(op->InnerOp(), tmp_ins, tmp_outs,
                                             op->Attrs(), op->place());
    if (double_grad_node) {
      double_grad_nodes_.emplace_back(double_grad_node);
      VLOG(10) << "Create " << double_grad_node->size()
               << " double grad op(s) for " << op->Type() << ", pending ops: "
               << GradPendingOpTypes(*double_grad_nodes_.back())
               << ", Output number: "
               << (*double_grad_node)[0].GetOutsMap().size();
    }
  }

  VLOG(10) << "There are " << grads_to_accumulate.size() << " to sum gradient";

  // Gradient accumulation and add assign op
  for (auto &pair : grads_to_accumulate) {
    auto *accumulator_info = pair.first;
    PADDLE_ENFORCE_NOT_NULL(accumulator_info);
    for (auto &grad_var : pair.second) {
      PADDLE_ENFORCE_NOT_NULL(grad_var);
      bool is_finished = false;
      VLOG(10) << "Start to sum " << accumulator_info->MappedGradVar()->Name();
      auto partial_grads = accumulator_info->SumGradient(
          std::move(grad_var), op->id(), &is_finished);

      if (is_finished) {
        VLOG(10) << "Sum has finished for "
                 << accumulator_info->MappedGradVar()->Name() << " "
                 << accumulator_info->GradVarBase();
        ready_grad_vars_.Set(accumulator_info->MappedGradVar(),
                             accumulator_info->GradVarBase());
      }

      if (partial_grads.empty()) {
        continue;
      }

      VLOG(10) << "Create new varbase";
      auto sum_grad_var_grad =
          accumulator_info->GradVarBase()->MutableGradVarBase();
      sum_grad_var_grad->SetOverridedStopGradient(false);
      VLOG(10) << "Create new varbase done";
      auto assign_node = std::make_shared<GradOpNode>();
      sum_grad_var_grad->SetGradNode(assign_node);
      for (auto &grad : partial_grads) {
        auto partial_grad_var_grad = grad->GetGradVar();
        if (partial_grad_var_grad == nullptr) {
          continue;
        }

        auto *assign_op = &(assign_node->emplace_back());
        VLOG(10) << "Add assign op for " << sum_grad_var_grad->Name();
        assign_op->SetType("assign");  // Can use "scale" as static graph mode
        assign_op->SetInput("X", {sum_grad_var_grad->SharedVar()}, true);
        assign_op->SetOutput("Out", {partial_grad_var_grad}, true);
        assign_op->CheckAttrs();

        if (auto grad_pending_node = partial_grad_var_grad->GetGradNode()) {
          assign_node->InsertGradPendingNode(std::move(grad_pending_node));
        }
        // PrintPendingOps(assign_op);
      }
      VLOG(10) << "Pending ops of assign is "
               << GradPendingOpTypes(*assign_node);
      grad_accumulators_.erase(accumulator_info->MappedGradVar());
      if (!assign_node->empty()) {
        double_grad_nodes_.emplace_back(assign_node);
      }
    }
  }
}

void PartialGradTask::PrepareInitialReadyVarsMap(const OpBase *op) {
  for (auto &in_var_pair : op->GetInsMap()) {
    if (!in_var_pair.second.IsGrad()) {
      continue;
    }

    for (auto &var : in_var_pair.second) {
      if (var) {
        ready_grad_vars_.IncreaseRefCnt(var.get());
      }
    }
  }
}

void PartialGradTask::PrepareInitialGradientAccumulators(const OpBase *op) {
  for (auto &out_var_pair : op->GetOutsMap()) {
    if (!out_var_pair.second.IsGrad()) {
      continue;
    }

    for (auto &var : out_var_pair.second) {
      if (var == nullptr) {
        continue;
      }

      auto &accumulator = grad_accumulators_[var.get()];

      if (!accumulator) {
        VLOG(10) << "Add gradient accumulator for " << op->Type() << "("
                 << out_var_pair.first << "): " << var->Name() << ":"
                 << var.get();
        accumulator.reset(new GradientAccumulationInfo(
            var, strategy_.sorted_sum_gradient_, create_graph_));
      }

      accumulator->IncreaseTotalRefCnt();
    }
  }
}

std::vector<std::shared_ptr<VarBase>> PartialGradTask::CreateResult() {
  std::vector<std::shared_ptr<VarBase>> result;
  result.reserve(input_targets_.size());
  for (auto &input_target : input_targets_) {
    PADDLE_ENFORCE_NOT_NULL(input_target->GradVarBase());
    auto *original_grad_var = input_target->GradVarBase()->SharedVar().get();
    PADDLE_ENFORCE_NOT_NULL(original_grad_var);
    auto iter = input_target_grads_.find(original_grad_var);
    if (iter != input_target_grads_.end()) {
      const auto &ready_var = ready_grad_vars_.GetTarget(original_grad_var);
      PADDLE_ENFORCE_NOT_NULL(ready_var);
      if (create_graph_) {
        ready_var->SetOverridedStopGradient(false);
      } else {
        ready_var->SetOverridedStopGradient(true);
      }
      result.emplace_back(std::move(ready_var));
    } else {  // return None if it does not appear in the graph
      result.emplace_back();
    }
  }

  ready_grad_vars_.Clear();
  grad_accumulators_.clear();
  double_grad_nodes_.clear();
  return result;
}

PartialGradEngine::PartialGradEngine(
    const std::vector<std::shared_ptr<VarBase>> &input_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_targets,
    const std::vector<std::shared_ptr<VarBase>> &output_grads,
    const platform::Place &place, const detail::BackwardStrategy &strategy,
    bool create_graph)
    : input_targets_(input_targets),
      output_targets_(output_targets),
      output_grads_(output_grads),
      place_(place),
      strategy_(strategy),
      create_graph_(create_graph) {}

std::vector<std::shared_ptr<VarBase>> PartialGradEngine::GetResult() const {
  return results_;
}

void PartialGradEngine::Clear() {
  input_targets_.clear();
  output_targets_.clear();
  output_grads_.clear();
}

void PartialGradEngine::Execute() {
  VLOG(10) << "Starts to execute PartialGradEngine";
  PartialGradTask task(input_targets_, output_targets_, output_grads_, place_,
                       strategy_, create_graph_);
  results_ = task.Run();
  Clear();
}

}  // namespace imperative
}  // namespace paddle
