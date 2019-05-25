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

const char* PyLayer::kFwdInp = "X";
const char* PyLayer::kFwdOut = "Out";

std::map<int, py::object> py_funcs_;

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

void AddTo(Variable* src, Variable* dst, platform::Place place) {
  framework::Tensor* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  framework::Tensor* src_tensor = src->GetMutable<framework::LoDTensor>();

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

void AddGradBySort(BackwardSumMap* bck_map, VarBase* target) {
  PADDLE_ENFORCE(bck_map->find(target) != bck_map->end(),
                 "Can't find %s in backward grad map", target->Name());
  std::pair<platform::Place, std::vector<std::pair<int, VarBase*>>>& current =
      bck_map->at(target);
  std::sort(
      current.second.begin(), current.second.end(),
      [](const std::pair<int, VarBase*>& a, const std::pair<int, VarBase*>& b) {
        return a.first > b.first;
      });
  for (auto& var_pair : current.second) {
    Variable* origin_grad = target->var_.get();
    Variable* grad_to_add = var_pair.second->var_.get();
    VLOG(2) << "add origin_grad: " << target->Name();
    VLOG(2) << "added grad: " << var_pair.second->Name()
            << " trace id is: " << var_pair.first;
    AddTo(grad_to_add, origin_grad, current.first);
    delete var_pair.second;
    var_pair.second = nullptr;
  }
}

class Autograd {
 public:
  Autograd() {}

  void RunBackward(VarBase* var, const detail::BackwardStrategy& bck_stratedy) {
    if (var->IsStopGradient()) {
      return;
    }
    VLOG(3) << "start autograd";
    BackwardSumMap bck_map;
    GradientRef grad_ref;
    std::deque<OpBase*> ready;
    ready.push_back(var->PreOp());

    std::map<OpBase*, int> dep_counts =
        ComputeDepCounts(var->PreOp(), bck_stratedy, &grad_ref);

    while (!ready.empty()) {
      OpBase* ready_op = ready.front();
      ready.pop_front();
      std::map<std::string, std::vector<VarBase*>> input_grads =
          ready_op->ApplyGrad(&bck_map, &grad_ref, bck_stratedy);

      for (auto it = input_grads.rbegin(); it != input_grads.rend(); ++it) {
        const std::vector<VarBase*>& ingrads = it->second;
        for (size_t i = 0; i < ingrads.size(); ++i) {
          if (!ingrads[i]) continue;
          auto p = ready_op->input_vars_[it->first][i];

          if (p->IsStopGradient()) continue;
          OpBase* pre_op = ready_op->pre_ops_[it->first][i];
          if (!pre_op) continue;

          dep_counts[pre_op] -= 1;
          PADDLE_ENFORCE(dep_counts[pre_op] >= 0);
          bool pre_op_ready = dep_counts[pre_op] == 0;
          if (pre_op_ready) {
            ready.push_back(pre_op);
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
      if (bck_stratedy.sorted_sum_gradient_) {
        for (const auto& map : candidate->grad_output_vars_) {
          for (const auto& it : map) {
            for (const auto& vb : it.second) {
              ++(*grad_ref)[vb];
            }
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

std::map<std::string, std::vector<VarBase*>> OpBase::ApplyGrad(
    BackwardSumMap* bck_map, GradientRef* grad_ref,
    const detail::BackwardStrategy& bck_stratedy) {
  PADDLE_ENFORCE(!grad_op_descs_.empty() || backward_id_ > 0,
                 "%s has no backward implementation", Type());
  VLOG(3) << "apply op grad: " << Type();
  std::vector<VarBasePtrMap> tmp_grad_outputs;
  if (backward_id_ > 0) {
    VLOG(3) << "py_layer_grad";
    tmp_grad_outputs.resize(1);
    tmp_grad_outputs[0][framework::GradVarName(PyLayer::kFwdOut)] =
        PyLayer::ApplyGrad(
            backward_id_,
            grad_input_vars_[0][framework::GradVarName(PyLayer::kFwdInp)]);
  } else {
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
        for (size_t i = 0; i < it.second.size(); ++i) {
          VarBase* origin_grad_var_base = it.second[i];

          // Allocate a new variable
          VarBase* tmp_grad_var_base = new VarBase(
              string::Sprintf("%s@IGrad", origin_grad_var_base->Name()),
              origin_grad_var_base->DataType(), origin_grad_var_base->Dims(),
              place_, true, false);
          outputs.emplace_back(tmp_grad_var_base);
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
            &grad_input_vars_[k], &tmp_grad_outputs[k], &attrs_);
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
        for (const VarBase* grad_inp : it.second) {
          PADDLE_ENFORCE_NOT_NULL(grad_inp->var_, "op %s input %s nullptr",
                                  grad_op_desc->Type(), grad_inp->Name());

          grad_invars.emplace_back(grad_inp->var_.get());
        }
      }

      for (const auto& it : tmp_grad_outputs[k]) {
        auto& grad_outvars = grad_outvars_map[it.first];
        grad_outvars.reserve(it.second.size());
        for (VarBase* grad_out : it.second) {
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
#ifndef PADDLE_WITH_CUDA
          VLOG(2) << "origin_outputs is : " << origin_outputs[i]->Name() << " ";
          VLOG(2) << origin_outputs[i]
                         ->var_->GetMutable<framework::LoDTensor>()
                         ->data<float>()[0];
          VLOG(2) << "outputs is : " << outputs[i]->Name() << " ";
          VLOG(2) << outputs[i]
                         ->var_->GetMutable<framework::LoDTensor>()
                         ->data<float>()[0];
#endif
          if (bck_map->find(origin_outputs[i]) != bck_map->end()) {
            VLOG(2) << "add sub grad to " << origin_outputs[i]->Name();
            bck_map->at(origin_outputs[i])
                .second.emplace_back(
                    std::pair<int, VarBase*>(this->trace_id_, outputs[i]));
          } else {
            VLOG(2) << "insert new map for " << origin_outputs[i]->Name();
            std::pair<platform::Place, std::vector<std::pair<int, VarBase*>>>
                tmp(place_, {std::make_pair(this->trace_id_, outputs[i])});
            bck_map->insert(std::make_pair(origin_outputs[i], tmp));
          }

          PADDLE_ENFORCE(grad_ref->find(origin_outputs[i]) != grad_ref->end(),
                         "Can't find  %s in grad_reference count map",
                         origin_outputs[i]->Name());
          PADDLE_ENFORCE(grad_ref->at(origin_outputs[i]) >= 1,
                         "Backward error when calculate grad reference");
          if (grad_ref->at(origin_outputs[i]) > 1) {
            VLOG(2) << "remove ref for " << origin_outputs[i]->Name();
            grad_ref->at(origin_outputs[i])--;
          } else {
            VLOG(2) << "Add grad for: " << origin_outputs[i]->Name();
            AddGradBySort(bck_map, origin_outputs[i]);
            grad_ref->at(origin_outputs[i])--;
          }
        } else {
          framework::Variable* grad = outputs[i]->var_.get();
          framework::Variable* orig_grad = origin_outputs[i]->var_.get();
          VLOG(2) << "AddTo Called with orig_grad is: "
                  << origin_outputs[i]->name_ << " Grad to be added is "
                  << outputs[i]->name_;
          AddTo(grad, orig_grad, place_);
          delete outputs[i];
        }
      }
    }
  }

  return input_vars_;
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
  auto grads_t = grads_->var_->GetMutable<framework::LoDTensor>();
  operators::math::set_constant(
      *(platform::DeviceContextPool::Instance().Get(
          var_->GetMutable<framework::LoDTensor>()->place())),
      grads_t, 1.0);

  PADDLE_ENFORCE(
      grads_ ==
      pre_op_->output_vars_[pre_op_out_name_][pre_op_out_idx_]->grads_);
  Autograd().RunBackward(this, bck_stratedy);
}

void PyLayer::RegisterFunc(int func_id, const py::object& py_func) {
  py_funcs_[func_id] = py_func;
}

int PyLayer::NumFuncs() { return py_funcs_.size(); }

std::vector<std::unique_ptr<framework::Variable>> PyLayer::Apply(
    int func_id, const std::vector<VarBase*>& inputs) {
  PADDLE_ENFORCE(py_funcs_.find(func_id) != py_funcs_.end());
  return CallPythonFunc(py_funcs_[func_id], inputs);
}

std::vector<VarBase*> PyLayer::ApplyGrad(int func_id,
                                         const std::vector<VarBase*>& inputs) {
  PADDLE_ENFORCE(py_funcs_.find(func_id) != py_funcs_.end());
  auto rets = CallPythonFunc(py_funcs_[func_id], inputs);

  std::vector<VarBase*> outs;
  outs.reserve(rets.size());
  for (size_t i = 0U; i != rets.size(); ++i) {
    outs.emplace_back(new VarBase(
        string::Sprintf("%s_out_%d", framework::GradVarName(PyLayer::kFwdOut),
                        i),
        std::move(rets[i]), nullptr, true));
  }

  return outs;
}

std::vector<std::unique_ptr<framework::Variable>> PyLayer::CallPythonFunc(
    const py::object& callable, const std::vector<VarBase*>& ins) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    const framework::LoDTensor& t = ins[i]->var_->Get<framework::LoDTensor>();
    in_args[i] = t.IsInitialized() ? py::cast(t) : py::cast(nullptr);
  }
  VLOG(3) << "pyfunc in " << py::len(in_args);

  // TODO(panyx0718): Who owns the returned LoDTensor.
  auto ret = callable(in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  std::vector<std::unique_ptr<framework::Variable>> outs;
  outs.reserve(ret_num);
  VLOG(3) << "pyfunc out " << ret_num;
  for (size_t i = 0; i < ret_num; ++i) {
    try {
      auto* py_out_tensor = py::cast<framework::LoDTensor*>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_tensor,
                              "Output tensor %d should not be nullptr", i);
      auto var =
          std::unique_ptr<framework::Variable>(new framework::Variable());
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      tensor->ShareDataWith(*py_out_tensor);
      tensor->set_lod(py_out_tensor->lod());
      outs.emplace_back(std::move(var));
    } catch (py::cast_error&) {
      PADDLE_THROW("The %d-th output must be LoDTensor", i);
    }
  }
  return outs;
}

}  // namespace imperative
}  // namespace paddle
