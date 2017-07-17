/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <paddle/framework/attr_checker.h>
#include <paddle/framework/op_desc.pb.h>
#include <paddle/framework/scope.h>
#include <paddle/framework/tensor.h>
#include <paddle/platform/device_context.h>
#include <paddle/platform/place.h>
#include <paddle/utils/Error.h>
#include <boost/variant.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

template <typename T>
struct EigenDeviceConverter;

template <>
struct EigenDeviceConverter<platform::CPUPlace> {
  using EigenDeviceType = Eigen::DefaultDevice;
};

#ifndef PADDLE_ONLY_CPU
template <>
struct EigenDeviceConverter<platform::GPUPlace> {
  using EigenDeviceType = Eigen::GpuDevice;
};
#endif

class OperatorBase;

/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  virtual ~OperatorBase() {}

  template <typename T>
  inline const T& GetAttr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  std::string DebugString() const;

  /// Init will be called after CreateOperator, you can put some initialization
  /// logic here.
  virtual void Init() {}

  /// InferShape infer the size of Variables used by this Operator with
  /// information inside scope
  virtual void InferShape(const std::shared_ptr<Scope>& scope) const = 0;

  /// Net will call this function to Run an op.
  virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const = 0;

 protected:
  std::string Type() const { return desc_.type(); }

 public:
  OpDesc desc_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  AttributeMap attrs_;
};

class OpKernel {
 public:
  /**
   * KernelContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * KernelContext. User should construct it before run the Operator.
   */
  class KernelContext {
   public:
    KernelContext(const OperatorBase* op, const std::shared_ptr<Scope>& scope,
                  const platform::DeviceContext& device_context)
        : op_(*op), scope_(scope), device_context_(device_context) {}

    const Variable* Input(int index) const {
      return scope_->GetVariable(op_.inputs_[index]);
    }

    Variable* Output(int index) const {
      return scope_->GetVariable(op_.outputs_[index]);
    }

    template <typename PlaceType,
              typename DeviceType =
                  typename EigenDeviceConverter<PlaceType>::EigenDeviceType>
    DeviceType* get_eigen_device() const;

    const OperatorBase& op_;
    const std::shared_ptr<Scope>& scope_;
    const platform::DeviceContext& device_context_;
  };

  virtual void Compute(const KernelContext& context) const = 0;

  virtual ~OpKernel() {}
};

template <typename T>
struct VarToTensor {};

template <>
struct VarToTensor<Tensor*> {
  Tensor* operator()(Variable* var) { return var->GetMutable<Tensor>(); }
};

template <>
struct VarToTensor<const Tensor*> {
  const Tensor* operator()(Variable* var) { return &var->Get<Tensor>(); }
};

class OperatorWithKernel : public OperatorBase {
 public:
  struct OpKernelKey {
    platform::Place place_;

    OpKernelKey() = default;
    OpKernelKey(const platform::DeviceContext& dev_ctx) {
      place_ = dev_ctx.GetPlace();
    }

    bool operator==(const OpKernelKey& o) const { return place_ == o.place_; }
  };

  struct OpKernelHash {
    std::hash<bool> hash_;
    size_t operator()(const OpKernelKey& key) const {
      return hash_(platform::is_gpu_place(key.place_));
    }
  };

  using OpKernelMap =
      std::unordered_map<OpKernelKey, std::unique_ptr<OpKernel>, OpKernelHash>;

  void Run(const std::shared_ptr<Scope>& scope,
           const platform::DeviceContext& dev_ctx) const final {
    auto& opKernel = AllOpKernels().at(Type()).at(OpKernelKey(dev_ctx));
    opKernel->Compute(OpKernel::KernelContext(this, scope, dev_ctx));
  }

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }
  void InferShape(const std::shared_ptr<Scope>& scope) const final {
    std::vector<const Tensor*> ins;
    VarNamesToTensors(scope, inputs_, &ins);
    std::vector<Tensor*> outs;
    VarNamesToTensors(scope, outputs_, &outs);
    InferShape(ins, outs);
  };

 private:
  template <typename T>
  void VarNamesToTensors(const std::shared_ptr<Scope>& scope,
                         const std::vector<std::string>& var_names,
                         std::vector<T>* container) const {
    container->reserve(var_names.size());
    VarToTensor<T> convert;
    for (auto& name : var_names) {
      auto var = scope->GetVariable(name);
      if (var != nullptr) {
        container->push_back(convert(var));
      } else {
        container->push_back(nullptr);
      }
    }
  }

 protected:
  virtual void InferShape(const std::vector<const Tensor*>& inputs,
                          const std::vector<Tensor*>& outputs) const = 0;
};

}  // namespace framework
}  // namespace paddle
