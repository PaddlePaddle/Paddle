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

#include "paddle/fluid/imperative/layer.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <random>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace imperative {

void ThreadSafeNameSet::Insert(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  set_.insert(name);
}

void ThreadSafeNameSet::Remove(const std::string& name) {
  std::lock_guard<std::mutex> guard(mtx_);
  auto iter = set_.find(name);
  PADDLE_ENFORCE(iter != set_.end(), "%s does not exist", name);
  set_.erase(iter);
}

std::vector<std::string> ThreadSafeNameSet::Names() const {
  std::lock_guard<std::mutex> guard(mtx_);
  return std::vector<std::string>(set_.begin(), set_.end());
}

ThreadSafeNameSet VarBase::name_set_;

std::vector<std::string> VarBase::AliveVarNames() { return name_set_.Names(); }

using framework::Variable;

namespace detail {

template <typename T>
class TensorAddToFunctor : public boost::static_visitor<> {
 public:
  TensorAddToFunctor(int64_t numel, const T* x, T* y)
      : numel_(numel), x_(x), y_(y) {}

  void operator()(const platform::CPUPlace& place) {
    platform::CPUDeviceContext* ctx = dynamic_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CPUDeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }

#ifdef PADDLE_WITH_CUDA
  void operator()(const platform::CUDAPlace& place) {
    platform::CUDADeviceContext* ctx =
        dynamic_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CUDADeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const platform::CUDAPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }
#endif

  // there is NO blas in CUDAPinnedPlace
  void operator()(const platform::CUDAPinnedPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

}  // namespace detail

void AddTo(std::shared_ptr<VarBase> src, std::shared_ptr<VarBase> dst,
           platform::Place place, GradientRef* grad_ref) {
  PADDLE_ENFORCE(grad_ref->find(dst.get()) != grad_ref->end(),
                 "gradient %s are not found in grad_ref", dst->Name());
  if ((*grad_ref)[dst.get()].second) {
    PADDLE_ENFORCE(src->IsInitialize(), "Using uninitialized VarBase");
    dst->var_ = std::move(src->var_);
    (*grad_ref)[dst.get()].second = false;
    if (!dst->IsInitialize()) {
      dst->SetInitialize(true);
    }
    return;
  } else {
    framework::Tensor* dst_tensor =
        dst->var_->GetMutable<framework::LoDTensor>();
    framework::Tensor* src_tensor =
        src->var_->GetMutable<framework::LoDTensor>();

    // FIXME(minqiyang): loss_grad op will pass a zero grad of label
    // ugly fix for it
    if (src_tensor->numel() == 0) {
      return;
    }

    PADDLE_ENFORCE(dst_tensor->numel() == src_tensor->numel(),
                   "dst_numel %lld vs. src_numel %lld", dst_tensor->numel(),
                   src_tensor->numel());

    detail::TensorAddToFunctor<float> func(
        src_tensor->numel(), src_tensor->data<float>(),
        dst_tensor->mutable_data<float>(place));
    boost::apply_visitor(func, place);
  }
}

void ZeroGrads(const std::shared_ptr<imperative::VarBase> vb,
               const platform::Place& place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.Get(place);
  auto grad_t = vb->var_->GetMutable<framework::LoDTensor>();
  operators::math::set_constant(*dev_ctx, grad_t, 0.0);
}

void AddGradBySort(BackwardSumMap* bck_map,
                   std::shared_ptr<imperative::VarBase> target,
                   GradientRef* grad_ref) {
  PADDLE_ENFORCE(bck_map->find(target.get()) != bck_map->end(),
                 "Can't find %s in backward grad map", target->Name());
  std::pair<platform::Place,
            std::vector<std::pair<int, std::shared_ptr<imperative::VarBase>>>>&
      current = bck_map->at(target.get());
  std::sort(current.second.begin(), current.second.end(),
            [](const std::pair<int, std::shared_ptr<imperative::VarBase>>& a,
               const std::pair<int, std::shared_ptr<imperative::VarBase>>& b) {
              return a.first > b.first;
            });
  for (auto& var_pair : current.second) {
    VLOG(10) << "add origin_grad: " << target->Name();
    VLOG(10) << "added grad: " << var_pair.second->Name()
             << " trace id is: " << var_pair.first;
    AddTo(var_pair.second, target, current.first, grad_ref);
    var_pair.second.reset();
  }
}

class Autograd {
 public:
  Autograd() {}

  void RunBackward(VarBase* var, const detail::BackwardStrategy& bck_stratedy) {
    if (var->IsStopGradient()) {
      return;
    }
    VLOG(2) << "start autograd";
    BackwardSumMap bck_map;
    std::deque<OpBase*> ready;
    ready.push_back(var->PreOp());

    std::map<OpBase*, int> dep_counts =
        ComputeDepCounts(var->PreOp(), bck_stratedy, &grad_ref);

    while (!ready.empty()) {
      OpBase* ready_op = ready.front();
      ready.pop_front();
      std::vector<VarBasePtrMap> grads_outputs =
          ready_op->ApplyGrad(&bck_map, &grad_ref, bck_stratedy);

      for (const auto& map : grads_outputs) {
        for (auto it = map.rbegin(); it != map.rend(); ++it) {
          const std::vector<std::shared_ptr<VarBase>>& grad_outs = it->second;
          for (size_t i = 0; i < grad_outs.size(); ++i) {
            if (!grad_outs[i] || grad_outs[i]->IsStopGradient()) continue;
            OpBase* pre_op = grad_outs[i]->PreOp();
            if (!pre_op) continue;
            dep_counts[pre_op] -= 1;
            PADDLE_ENFORCE(dep_counts[pre_op] >= 0);
            bool pre_op_ready = dep_counts[pre_op] == 0;
            if (pre_op_ready) {
              ready.push_back(pre_op);
            }
          }
        }
      }

      ready_op->InvokeBackwardHooks();
    }
  }

 private:
  std::map<OpBase*, int> ComputeDepCounts(
      OpBase* op, const detail::BackwardStrategy& bck_stratedy,
      GradientRef* grad_ref) {
    if (bck_stratedy.sorted_sum_gradient_) {
      PADDLE_ENFORCE_NOT_NULL(grad_ref,
                              "grad_ref should not be null when "
                              "using sorted grad backward strategy");
    }
    std::map<OpBase*, int> ret;

    std::deque<OpBase*> queue;
    queue.push_back(op);
    std::unordered_set<OpBase*> visited;
    visited.insert(op);
    while (!queue.empty()) {
      OpBase* candidate = queue.front();
      queue.pop_front();
      for (const auto& map : candidate->grad_output_vars_) {
        for (const auto& it : map) {
          for (const auto& vb : it.second) {
            if (bck_stratedy.sorted_sum_gradient_) {
              ++(*grad_ref)[vb.get()].first;
            }
            // init the state of the grad_
            (*grad_ref)[vb.get()].second = true;
          }
        }
      }
      for (auto it : candidate->pre_ops_) {
        for (OpBase* pre_op : it.second) {
          if (!pre_op) continue;
          VLOG(2) << "op dep " << candidate->Type() << " trace id "
                  << candidate->trace_id_ << " <---- " << it.first << " <---- "
                  << pre_op->Type() << " trace id " << pre_op->trace_id_;
          if (visited.find(pre_op) == visited.end()) {
            visited.insert(pre_op);
            queue.push_back(pre_op);
          }
          ret[pre_op] += 1;
        }
      }
    }
    return ret;
  }

  GradientRef grad_ref;
};

std::unique_ptr<VarBase> VarBase::NewVarBase(const platform::Place& dst_place,
                                             const bool blocking) const {
  PADDLE_ENFORCE(var_->IsInitialized(),
                 "Variable must be initialized when getting numpy tensor");

  // TODO(minqiyang): change this after move unique_name generator to CXX
  const framework::LoDTensor& self_tensor = var_->Get<framework::LoDTensor>();
  std::unique_ptr<VarBase> new_var(new VarBase(
      "Itmp", self_tensor.type(), self_tensor.dims(), dst_place, true, false));
  framework::LoDTensor* tensor =
      new_var->var_->GetMutable<framework::LoDTensor>();
  tensor->set_lod(var_->Get<framework::LoDTensor>().lod());

  const auto& src_tensor = var_->Get<framework::LoDTensor>();
  framework::TensorCopy(src_tensor, dst_place, tensor);
  if (blocking) {
    platform::DeviceContextPool::Instance().Get(dst_place)->Wait();
    auto src_place = src_tensor.place();
    if (!(src_place == dst_place)) {
      platform::DeviceContextPool::Instance().Get(src_place)->Wait();
    }
  }

  if (platform::is_gpu_place(dst_place)) {
    VLOG(3) << "copy tensor " << Name() << " from gpu";
  }

  return new_var;
}

framework::LoDTensor& VarBase::GradValue() {
  VLOG(3) << "get var grad " << Name();
  PADDLE_ENFORCE_NOT_NULL(grads_,
                          "Could not get grad value from no grad variable");
  return *(grads_->var_->GetMutable<framework::LoDTensor>());
}

std::vector<VarBasePtrMap> OpBase::ApplyGrad(
    BackwardSumMap* bck_map, GradientRef* grad_ref,
    const detail::BackwardStrategy& bck_stratedy) {
  PADDLE_ENFORCE(!grad_op_descs_.empty(), "%s has no backward implementation",
                 Type());
  VLOG(3) << "apply op grad: " << Type();
  std::vector<VarBasePtrMap> tmp_grad_outputs;
  const size_t grad_op_count = grad_op_descs_.size();

  tmp_grad_outputs.resize(grad_op_count);
  for (size_t k = 0; k < grad_op_count; ++k) {
    framework::OpDesc* grad_op_desc = grad_op_descs_[k];
    platform::RecordEvent record_event(grad_op_desc->Type());
    auto& grad_output_variable_map = grad_output_vars_[k];
    VLOG(3) << "apply grad op " << grad_op_desc->Type();

    // Allocate tmp grad output variable
    for (const auto& it : grad_output_variable_map) {
      auto& outputs = tmp_grad_outputs[k][it.first];
      outputs.reserve(it.second.size());
      for (const std::shared_ptr<imperative::VarBase>& origin_grad_var_base :
           it.second) {
        // Allocate a new variable
        std::shared_ptr<imperative::VarBase> tmp_grad_var_base(new VarBase(
            string::Sprintf("%s@IGrad", origin_grad_var_base->Name()),
            origin_grad_var_base->DataType(), origin_grad_var_base->Dims(),
            place_, true, false));
        outputs.emplace_back(std::move(tmp_grad_var_base));
      }
    }

    // No need to do compile time infer shape here.
    // grad_op_desc_->InferShape(*block_);
    // grad_op_desc->InferVarType(block_);

    std::unique_ptr<framework::OperatorBase> opbase =
        framework::OpRegistry::CreateOp(*grad_op_desc);

    auto& info = framework::OpInfoMap::Instance().Get(grad_op_desc->Type());
    if (info.infer_var_type_) {
      RuntimeInferVarTypeContext infer_var_type_ctx(
          &grad_input_vars_[k], &tmp_grad_outputs[k], &(opbase->Attrs()));
      info.infer_var_type_(&infer_var_type_ctx);
    }

    framework::OperatorWithKernel* op_kernel =
        dynamic_cast<framework::OperatorWithKernel*>(opbase.get());
    PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");

    // Run grad op
    framework::VariableValueMap grad_invars_map;
    framework::VariableValueMap grad_outvars_map;

    for (const auto& it : grad_input_vars_[k]) {
      auto& grad_invars = grad_invars_map[it.first];
      grad_invars.reserve(it.second.size());
      for (const std::shared_ptr<imperative::VarBase>& grad_inp : it.second) {
        PADDLE_ENFORCE_NOT_NULL(grad_inp->var_, "op %s input %s nullptr",
                                grad_op_desc->Type(), grad_inp->Name());
        if (!grad_inp->IsInitialize()) {
          grad_inp->InitBuffer();
          ZeroGrads(grad_inp, place_);
        }
        const std::shared_ptr<imperative::VarBase>& const_grad_inp = grad_inp;
        grad_invars.emplace_back(const_grad_inp->var_.get());
      }
    }

    for (const auto& it : tmp_grad_outputs[k]) {
      auto& grad_outvars = grad_outvars_map[it.first];
      grad_outvars.reserve(it.second.size());
      for (const std::shared_ptr<imperative::VarBase>& grad_out : it.second) {
        PADDLE_ENFORCE_NOT_NULL(grad_out->var_, "op %s output %s nullptr",
                                grad_op_desc->Type(), grad_out->Name());

        grad_outvars.emplace_back(grad_out->var_.get());
      }
    }

    framework::RuntimeContext ctx(grad_invars_map, grad_outvars_map);
    framework::Scope scope;
    PreparedOp p = PreparedOp::Prepare(ctx, *op_kernel, place_);
    p.op.RuntimeInferShape(scope, place_, ctx);
    p.func(
        framework::ExecutionContext(p.op, scope, *p.dev_ctx, p.ctx, nullptr));
  }

  platform::RecordEvent record_event("merge_grads");
  // Add tmp grad outputs to original grad vars
  for (size_t k = 0; k < grad_output_vars_.size(); ++k) {
    for (const auto& it : grad_output_vars_[k]) {
      auto& outputs = tmp_grad_outputs[k][it.first];
      const auto& origin_outputs = it.second;
      PADDLE_ENFORCE_EQ(outputs.size(), origin_outputs.size());

      for (size_t i = 0; i < outputs.size(); ++i) {
        // track outputs used by sum
        if (bck_stratedy.sorted_sum_gradient_) {
          if (bck_map->find(origin_outputs[i].get()) != bck_map->end()) {
            VLOG(10) << "add sub grad to " << origin_outputs[i]->Name();
            bck_map->at(origin_outputs[i].get())
                .second.emplace_back(
                    std::pair<int, std::shared_ptr<imperative::VarBase>>(
                        this->trace_id_, std::move(outputs[i])));
          } else {
            VLOG(10) << "insert new map for " << origin_outputs[i]->Name();
            std::pair<platform::Place,
                      std::vector<
                          std::pair<int, std::shared_ptr<imperative::VarBase>>>>
                tmp(place_,
                    {std::make_pair(this->trace_id_, std::move(outputs[i]))});
            bck_map->insert(std::make_pair(origin_outputs[i].get(), tmp));
          }

          PADDLE_ENFORCE(
              grad_ref->find(origin_outputs[i].get()) != grad_ref->end(),
              "Can't find  %s in grad_reference count map",
              origin_outputs[i]->Name());
          PADDLE_ENFORCE(grad_ref->at(origin_outputs[i].get()).first >= 1,
                         "Backward error when calculate grad reference");
          if (grad_ref->at(origin_outputs[i].get()).first > 1) {
            VLOG(10) << "remove ref for " << origin_outputs[i]->Name();
            grad_ref->at(origin_outputs[i].get()).first--;
          } else {
            VLOG(10) << "Add grad for: " << origin_outputs[i]->Name();
            AddGradBySort(bck_map, origin_outputs[i], grad_ref);
            grad_ref->at(origin_outputs[i].get()).first--;
          }
        } else {
          VLOG(10) << "AddTo Called with orig_grad is: "
                   << origin_outputs[i]->name_ << " Grad to be added is "
                   << outputs[i]->name_;
          AddTo(outputs[i], origin_outputs[i], place_, grad_ref);
          outputs[i].reset();
        }
      }
    }
  }

  return grad_output_vars_;
}

void OpBase::InvokeBackwardHooks() {
  VLOG(3) << "call backward hooks, hooks num: " << backward_hooks_.size();

  // call backward hooks
  for (py::object& callable : backward_hooks_) {
    callable(this);
  }
}

void OpBase::RegisterBackwardHooks(const py::object& callable) {
  VLOG(3) << "Register backward hooks " << trace_id_;

  // TODO(minqiyang): check the callable format
  backward_hooks_.push_back(callable);
}

void VarBase::RunBackward(const detail::BackwardStrategy& bck_stratedy) {
  if (!pre_op_) return;
  platform::RecordEvent record_event("Imperative Backward");
  VLOG(3) << "start backward";
  grads_->InitBuffer();
  auto grads_t = grads_->var_->GetMutable<framework::LoDTensor>();
  operators::math::set_constant(
      *(platform::DeviceContextPool::Instance().Get(
          var_->GetMutable<framework::LoDTensor>()->place())),
      grads_t, 1.0);

  Autograd().RunBackward(this, bck_stratedy);
}

}  // namespace imperative
}  // namespace paddle
