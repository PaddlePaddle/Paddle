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
  VarBase() : VarBase(new framework::Variable(), new VarBase(true)) {}

  explicit VarBase(bool stop_gradient)
      : VarBase(new framework::Variable(),
                stop_gradient ? nullptr : new VarBase(true), stop_gradient) {}

  VarBase(framework::Variable* var, VarBase* grad)
      : VarBase(var, grad, false) {}

 private:
  VarBase(framework::Variable* var, VarBase* grad, bool stop_gradient)
      : var_desc_(nullptr),
        var_(var),
        grads_(grad),
        stop_gradient_(stop_gradient),
        pre_op_(nullptr),
        pre_op_out_idx_(-1) {}

 public:
  virtual ~VarBase() {
    if (var_) {
      delete var_;
    }

    if (grads_) {
      delete grads_;
    }
  }

  inline OpBase* PreOp() const { return pre_op_; }
  inline int PreOpOutIdx() const { return pre_op_out_idx_; }

  inline void SetStopGradient(bool stop_gradient) {
    stop_gradient_ = stop_gradient;
  }
  inline bool IsStopGradient() const { return stop_gradient_; }

  void RunBackward();

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
    VLOG(1) << "clear gradient of " << var_desc_->Name();
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
    PADDLE_ENFORCE(
        var_desc_,
        "Couldn't get gradient variable's name, please call backward() first");
    return string::Sprintf("%s@IGrad", var_desc_->Name());
  }

  framework::VarDesc* var_desc_;

  framework::Variable* var_;
  VarBase* grads_;

 private:
  bool stop_gradient_;
  OpBase* pre_op_;
  std::string pre_op_out_name_;
  int pre_op_out_idx_;
};

/* The wrapper for OpDesc which holds a OpDesc and a OpDesc of its
 * gradient. This object should be managed totally by Python intepreter.
 */
class OpBase {
 public:
  OpBase()
      : op_desc_(nullptr),
        forward_id_(-1),
        backward_id_(-1),
        place_(platform::CPUPlace()) {}

  virtual ~OpBase() {
    for (framework::OpDesc* desc : grad_op_descs_) {
      delete desc;
    }
  }

  std::map<std::string, std::vector<VarBase*>> ApplyGrad();

  // One of `op_desc_` or `forward_id_` is set, not both.
  // For pure python PyLayer, use `forward_id_`, otherwise, use op_desc_.
  framework::OpDesc* op_desc_;
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

  framework::BlockDesc* block_;
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

  static std::vector<VarBase*> Apply(int func_id,
                                     const std::vector<VarBase*>& inputs);

  static std::vector<framework::Variable*> ApplyGrad(
      int func_id, const std::vector<framework::Variable*>& inputs);

 private:
  static std::vector<framework::Variable*> CallPythonFunc(
      const py::object& callable, const std::vector<framework::Variable*>& ins);
};

}  // namespace imperative
}  // namespace paddle
