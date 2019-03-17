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

#pragma once

// clang-format off
#include "paddle/fluid/framework/python_headers.h"
// clang-format on

#include <map>     // NOLINT
#include <string>  // NOLINT
#include <vector>  // NOLINT
#include <memory>  // NOLINT

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace imperative {

class VarBase;

namespace py = ::pybind11;

class PreparedOp {
 public:
  PreparedOp(const framework::OperatorBase& op,
             const framework::RuntimeContext& ctx,
             framework::OperatorWithKernel::OpKernelFunc func,
             platform::DeviceContext* dev_ctx,
             std::vector<framework::KernelConfig>* kernel_configs)
      : op(op),
        ctx(ctx),
        func(func),
        dev_ctx(dev_ctx),
        kernel_configs(kernel_configs) {}

  static PreparedOp Prepare(const framework::RuntimeContext& ctx,
                            const framework::OperatorWithKernel& op,
                            const platform::Place& place) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);

    // check if op[type] has kernel registered.
    auto& all_op_kernels = op.AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op.Type());
    if (kernels_iter == all_op_kernels.end()) {
      PADDLE_THROW(
          "There are no kernels which are registered in the %s operator.",
          op.Type());
    }

    framework::OperatorWithKernel::OpKernelMap& kernels = kernels_iter->second;

    auto expected_kernel_key =
        op.GetExpectedKernelType(framework::ExecutionContext(
            op, framework::Scope(), *dev_ctx, ctx, nullptr));
    VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

    auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
    // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
    if (kernel_iter == kernels.end() &&
        expected_kernel_key.library_type_ == framework::LibraryType::kMKLDNN) {
      VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
      expected_kernel_key.library_type_ = framework::LibraryType::kPlain;
      expected_kernel_key.data_layout_ = framework::DataLayout::kAnyLayout;
      kernel_iter = kernels.find(expected_kernel_key);
    }
#endif
    if (kernel_iter == kernels.end()) {
      PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                   KernelTypeToString(expected_kernel_key));
    }
    std::vector<framework::KernelConfig>* kernel_configs =
        op.GetKernelConfig(expected_kernel_key);
    return PreparedOp(op, ctx, kernel_iter->second, dev_ctx, kernel_configs);
  }

  inline platform::DeviceContext* GetDeviceContext() const { return dev_ctx; }

  const framework::OperatorBase& op;
  const framework::RuntimeContext& ctx;
  framework::OperatorWithKernel::OpKernelFunc func;
  platform::DeviceContext* dev_ctx;
  std::vector<framework::KernelConfig>* kernel_configs;
};

class OpBase;

/* The wrapper for Variable which holds a Variable and a VarBase of its
 * gradient. This object should be managed totally by Python intepreter.
 *
 * Nearly all interface should be implemented in C++.
 */
class VarBase {
 public:
  // Internal interface, create VarBase from exist variable
  VarBase(const std::string& name, framework::Variable* var, VarBase* grad,
          bool stop_gradient)
      : VarBase(name, var->Get<framework::LoDTensor>().type(),
                var->Get<framework::LoDTensor>().dims(),
                var->Get<framework::LoDTensor>().place(), var, grad,
                stop_gradient, false) {}

  // Python interface
  VarBase(const std::string& name, const framework::proto::VarType::Type dtype,
          const std::vector<int64_t>& shape, const platform::Place& place,
          bool stop_gradient, bool persistable)
      : VarBase(name, dtype, framework::make_ddim(shape), place, stop_gradient,
                persistable) {}

  // Internal interface, create VarBase from with ddim
  VarBase(const std::string& name, const framework::proto::VarType::Type dtype,
          const framework::DDim& shape, const platform::Place& place,
          bool stop_gradient, bool persistable)
      : VarBase(name, dtype, shape, place, nullptr, nullptr, stop_gradient,
                persistable) {}

 private:
  VarBase(const std::string& name, framework::proto::VarType::Type dtype,
          const framework::DDim& shape, const platform::Place& place,
          framework::Variable* var, VarBase* grad, bool stop_gradient,
          bool persistable)
      : name_(name),
        dtype_(dtype),
        place_(place),
        var_(var),
        grads_(grad),
        stop_gradient_(stop_gradient),
        persistable_(persistable),
        pre_op_(nullptr),
        pre_op_out_name_(),
        pre_op_out_idx_(-1) {
    if (!var_) {
      var_ = new framework::Variable();
      auto tensor = var_->GetMutable<framework::LoDTensor>();
      tensor->Resize(shape);
      tensor->mutable_data(place_, dtype_);
    }
  }

 public:
  virtual ~VarBase() {
    if (var_) {
      delete var_;
      var_ = nullptr;
    }

    if (grads_) {
      delete grads_;
      grads_ = nullptr;
    }

    pre_op_ = nullptr;
    pre_op_out_idx_ = -1;
  }

  inline void SetName(const std::string& name) { name_ = name; }
  inline std::string Name() const { return name_; }

  inline std::vector<int64_t> Shape() const {
    if (var_->IsInitialized()) {
      return framework::vectorize(var_->Get<framework::LoDTensor>().dims());
    } else {
      return {};
    }
  }

  inline framework::proto::VarType::Type DType() const { return dtype_; }

  inline void SetStopGradient(bool stop_gradient) {
    stop_gradient_ = stop_gradient;
  }
  inline bool IsStopGradient() const { return stop_gradient_; }

  inline void SetPersistable(bool persistable) { persistable_ = persistable; }
  inline bool IsPersistable() const { return persistable_; }

  inline OpBase* PreOp() const { return pre_op_; }
  inline int PreOpOutIdx() const { return pre_op_out_idx_; }

  void RunBackward();

  inline void ResetPreOp(OpBase* op) {
    if (op == pre_op_) {
      // clear pre_op info when op equals to var's pre_op
      pre_op_ = nullptr;
      pre_op_out_idx_ = -1;
    }
  }

  void TrackPreOp(OpBase* pre_op, const std::string& pre_op_out_name,
                  int pre_op_out_idx, bool pre_op_stop_gradient) {
    pre_op_ = pre_op;
    pre_op_out_name_ = pre_op_out_name;
    pre_op_out_idx_ = pre_op_out_idx;
    if (pre_op_stop_gradient) {
      stop_gradient_ = pre_op_stop_gradient;
    }
  }

  void ClearGradient() {
    VLOG(1) << "clear gradient of " << Name();
    if (grads_ && grads_->var_ && grads_->var_->IsInitialized()) {
      auto grads_t = grads_->var_->GetMutable<framework::LoDTensor>();
      operators::math::set_constant(
          *(platform::DeviceContextPool::Instance().Get(
              grads_->var_->Get<framework::LoDTensor>().place())),
          grads_t, 0.0);
    }
  }

  framework::LoDTensor& GradValue();

  std::unique_ptr<VarBase> NewVarBase(const platform::Place& dst_place,
                                      const bool blocking) const;

  inline std::string GradName() const {
    return string::Sprintf("%s@IGrad", Name());
  }

  std::string name_;
  framework::proto::VarType::Type dtype_;
  platform::Place place_;

  framework::Variable* var_;
  VarBase* grads_;

 private:
  bool stop_gradient_;
  bool persistable_;

  OpBase* pre_op_;
  std::string pre_op_out_name_;
  int pre_op_out_idx_;
};

/* The wrapper for OpDesc which holds a OpDesc and a OpDesc of its
 * gradient. This object should be managed totally by Python intepreter.
 */
class PYBIND11_HIDDEN OpBase {
 public:
  OpBase(const std::string& type)
      : type_(type),
        trace_id_(-1),
        forward_id_(-1),
        backward_id_(-1),
        place_(platform::CPUPlace()),
        backward_hooks_() {}

  virtual ~OpBase() {
    // TODO(minqiyang): remove op_desc from block_desc in tracer
    //
    // reset all output vars' pre op
    for (auto iter : output_vars_) {
      for (VarBase* var : iter.second) {
        var->ResetPreOp(this);
      }
    }

    // release resource
    for (framework::OpDesc* desc : grad_op_descs_) {
      delete desc;
    }
  }

  std::map<std::string, std::vector<VarBase*>> ApplyGrad();

  inline std::string Type() const { return type_; }
  inline std::string GradOpType(size_t index) const {
    PADDLE_ENFORCE_NOT_NULL(grad_op_descs_[index]);
    return grad_op_descs_[index]->Type();
  }

  void RegisterBackwardHooks(const py::object& callable);

  void InvokeBackwardHooks();

  void TrackPreOp(const VarBase* inp_var, const std::string& inp_name) {
    if (inp_var->PreOp() && !inp_var->IsStopGradient()) {
      VLOG(3) << "add pre op " << inp_var->PreOp()->Type() << " in slot "
              << inp_name;
      pre_ops_[inp_name].push_back(inp_var->PreOp());
      pre_ops_out_idx_[inp_name].push_back(inp_var->PreOpOutIdx());
    } else {
      VLOG(3) << "no pre op in slot " << inp_name
              << " input var stop_gradient: " << inp_var->IsStopGradient();
      pre_ops_[inp_name].push_back(nullptr);
      // pre_ops_out_idx_[inp_name].push_back(-1);
    }
  }

  std::string type_;
  // One of `trace_id_` or `forward_id_` is set, not both.
  // For pure python PyLayer, use `forward_id_`, otherwise, use trace_id_.
  int trace_id_;
  int forward_id_;

  // When has backward, one of `grad_op_descs_` or `backward_id_` is set,
  // not both.
  // Note: each fwd op corresponds to a vector of bwd ops.
  std::vector<framework::OpDesc*> grad_op_descs_;
  int backward_id_;

  platform::Place place_;

  VarBasePtrMap input_vars_;
  VarBasePtrMap output_vars_;
  OpBasePtrMap pre_ops_;
  std::map<std::string, std::vector<int>> pre_ops_out_idx_;

  // Inputs to a vector of bwd ops.
  std::vector<framework::VariableValueMap> grad_input_vars_;
  // Outputs to a vector of bwd ops.
  std::vector<framework::VariableValueMap> grad_output_vars_;

  std::vector<py::object> backward_hooks_;
};

class Layer {
 public:
  virtual ~Layer() {}

  virtual std::vector<VarBase> Forward(const std::vector<VarBase>& inputs) {
    std::vector<VarBase> vars;
    return vars;
  }
};

class PyLayer {
 public:
  virtual ~PyLayer() {}

  static const char* kFwdInp;
  static const char* kFwdOut;

  static void RegisterFunc(int func_id, const py::object& py_func);

  static int NumFuncs();

  static std::vector<framework::Variable*> Apply(
      int func_id, const std::vector<VarBase*>& inputs);

  static std::vector<framework::Variable*> ApplyGrad(
      int func_id, const std::vector<framework::Variable*>& inputs);

 private:
  static std::vector<framework::Variable*> CallPythonFunc(
      const py::object& callable, const std::vector<framework::Variable*>& ins);
};

}  // namespace imperative
}  // namespace paddle
