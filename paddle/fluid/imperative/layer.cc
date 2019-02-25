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
#include <utility>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"
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

class Autograd {
 public:
  Autograd() {}

  void RunBackward(VarBase* var) {
    if (var->IsStopGradient()) {
      return;
    }
    VLOG(3) << "start autograd";

    std::deque<OpBase*> ready;
    ready.push_back(var->PreOp());

    std::map<OpBase*, int> dep_counts = ComputeDepCounts(var->PreOp());

    while (!ready.empty()) {
      OpBase* ready_op = ready.front();
      ready.pop_front();
      std::map<std::string, std::vector<VarBase*>> input_grads =
          ready_op->ApplyGrad();

      for (auto it : input_grads) {
        const std::vector<VarBase*>& ingrads = it.second;
        for (size_t i = 0; i < ingrads.size(); ++i) {
          if (!ingrads[i]) continue;
          if (ready_op->input_vars_[it.first][i]->IsStopGradient()) {
            continue;
          }
          OpBase* pre_op = ready_op->pre_ops_[it.first][i];
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
  }

 private:
  std::map<OpBase*, int> ComputeDepCounts(OpBase* op) {
    std::map<OpBase*, int> ret;

    std::deque<OpBase*> queue;
    queue.push_back(op);
    std::unordered_set<OpBase*> visited;
    visited.insert(op);
    while (!queue.empty()) {
      OpBase* candidate = queue.front();
      queue.pop_front();
      for (auto it : candidate->pre_ops_) {
        for (OpBase* pre_op : it.second) {
          if (!pre_op) continue;
          VLOG(5) << "op dep " << candidate->op_desc_->Type() << " <---- "
                  << it.first << " <---- " << pre_op->op_desc_->Type();
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

  std::unique_ptr<VarBase> new_var(new VarBase());
  framework::LoDTensor* tensor =
      new_var->var_->GetMutable<framework::LoDTensor>();
  tensor->Resize(var_->Get<framework::LoDTensor>().dims());
  tensor->set_lod(var_->Get<framework::LoDTensor>().lod());

  if (blocking) {
    platform::DeviceContext* dev_ctx =
        platform::DeviceContextPool::Instance().Get(dst_place);

    framework::TensorCopySync(var_->Get<framework::LoDTensor>(), dst_place,
                              tensor);

    dev_ctx->Wait();
  } else {
    framework::TensorCopy(var_->Get<framework::LoDTensor>(), dst_place, tensor);
  }

  if (platform::is_gpu_place(dst_place)) {
    VLOG(3) << "copy tensor " << var_desc_->Name() << " from gpu";
  }

  return new_var;
}

framework::LoDTensor& VarBase::GradValue() {
  VLOG(3) << "get var grad " << var_desc_->Name();
  return *(grads_->var_->GetMutable<framework::LoDTensor>());
}

std::map<std::string, std::vector<VarBase*>> OpBase::ApplyGrad() {
  if (grad_op_descs_.empty() && backward_id_ <= 0) {
    VLOG(3) << "op with no grad: " << op_desc_->Type();
    return {};
  }

  std::vector<framework::VariableValueMap> grad_outputs;
  if (backward_id_ > 0) {
    VLOG(3) << "py_layer_grad";
    grad_outputs.resize(1);
    grad_outputs[0][framework::GradVarName(PyLayer::kFwdOut)] =
        PyLayer::ApplyGrad(
            backward_id_,
            grad_input_vars_[0][framework::GradVarName(PyLayer::kFwdInp)]);
  } else {
    grad_outputs.resize(grad_op_descs_.size());
    for (size_t k = 0; k < grad_op_descs_.size(); ++k) {
      framework::OpDesc* grad_op_desc = grad_op_descs_[k];
      VLOG(3) << "op grad " << grad_op_desc->Type();
      for (auto it : grad_output_vars_[k]) {
        auto& outputs = grad_outputs[k][it.first];
        for (size_t i = 0; i < it.second.size(); ++i) {
          // Allocate a new variable
          Variable* tmp_var = new framework::Variable();
          tmp_var->GetMutable<framework::LoDTensor>();
          outputs.push_back(tmp_var);
        }
      }

      framework::RuntimeContext ctx(grad_input_vars_[k], grad_outputs[k]);

      // No need to do compile time infer shape here.
      // grad_op_desc_->InferShape(*block_);
      grad_op_desc->InferVarType(block_);

      std::unique_ptr<framework::OperatorBase> opbase =
          framework::OpRegistry::CreateOp(*grad_op_desc);
      framework::OperatorWithKernel* op_kernel =
          dynamic_cast<framework::OperatorWithKernel*>(opbase.get());
      PADDLE_ENFORCE_NOT_NULL(op_kernel, "only support op with kernel");

      framework::Scope scope;
      PreparedOp p = PreparedOp::Prepare(ctx, *op_kernel, place_);
      p.op.RuntimeInferShape(scope, place_, ctx);
      p.func(framework::ExecutionContext(p.op, scope, *p.dev_ctx, p.ctx));
    }
  }

  for (size_t k = 0; k < grad_output_vars_.size(); ++k) {
    for (auto it : grad_output_vars_[k]) {
      auto& outputs = grad_outputs[k][it.first];
      auto& origin_outputs = it.second;
      PADDLE_ENFORCE_EQ(outputs.size(), origin_outputs.size());

      for (size_t i = 0; i < outputs.size(); ++i) {
        framework::Variable* grad = outputs[i];
        framework::Variable* orig_grad = origin_outputs[i];
        AddTo(grad, orig_grad, place_);
        delete grad;
      }
    }
  }

  return input_vars_;
}

void VarBase::RunBackward() {
  if (!pre_op_) return;

  VLOG(3) << "start backward";
  auto grads_t = grads_->var_->GetMutable<framework::LoDTensor>();
  operators::math::set_constant(
      *(platform::DeviceContextPool::Instance().Get(
          var_->GetMutable<framework::LoDTensor>()->place())),
      grads_t, 1.0);

  PADDLE_ENFORCE(
      grads_ ==
      pre_op_->output_vars_[pre_op_out_name_][pre_op_out_idx_]->grads_);
  Autograd().RunBackward(this);
}

void PyLayer::RegisterFunc(int func_id, const py::object& py_func) {
  py_funcs_[func_id] = py_func;
}

int PyLayer::NumFuncs() { return py_funcs_.size(); }

std::vector<VarBase*> PyLayer::Apply(int func_id,
                                     const std::vector<VarBase*>& inputs) {
  std::vector<framework::Variable*> invars;
  for (const VarBase* in : inputs) {
    invars.push_back(in->var_);
  }
  PADDLE_ENFORCE(py_funcs_.find(func_id) != py_funcs_.end());
  std::vector<Variable*> outvars = CallPythonFunc(py_funcs_[func_id], invars);
  std::vector<VarBase*> ret;
  for (Variable* v : outvars) {
    ret.push_back(new VarBase(v, new VarBase(true)));
  }
  return ret;
}

std::vector<Variable*> PyLayer::ApplyGrad(
    int func_id, const std::vector<framework::Variable*>& inputs) {
  PADDLE_ENFORCE(py_funcs_.find(func_id) != py_funcs_.end());
  return CallPythonFunc(py_funcs_[func_id], inputs);
}

std::vector<framework::Variable*> PyLayer::CallPythonFunc(
    const py::object& callable, const std::vector<framework::Variable*>& ins) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    const framework::LoDTensor& t = ins[i]->Get<framework::LoDTensor>();
    in_args[i] = t.IsInitialized() ? py::cast(t) : py::cast(nullptr);
  }
  VLOG(3) << "pyfunc in " << py::len(in_args);

  // TODO(panyx0718): Who owns the returned LoDTensor.
  auto ret = callable(in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  std::vector<framework::Variable*> outs;
  VLOG(3) << "pyfunc out " << ret_num;
  for (size_t i = 0; i < ret_num; ++i) {
    try {
      auto* py_out_tensor = py::cast<framework::LoDTensor*>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_tensor,
                              "Output tensor %d should not be nullptr", i);
      auto* var = new framework::Variable();
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      tensor->ShareDataWith(*py_out_tensor);
      tensor->set_lod(py_out_tensor->lod());
      outs.push_back(var);
    } catch (py::cast_error&) {
      PADDLE_THROW("The %d-th output must be LoDTensor", i);
    }
  }
  return outs;
}

}  // namespace imperative
}  // namespace paddle
