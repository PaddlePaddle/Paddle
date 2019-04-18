// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/core/type_system.h"
#include "paddle/fluid/lite/core/types.h"
#include "paddle/fluid/lite/operators/op_params.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// An base with virtual functions to unify all the kernel implementation on
// different targets.
class KernelBase {
 public:
  virtual void Run() = 0;

  template <TargetType Target>
  void SetContext(std::unique_ptr<Context<Target>>&& ctx) {
    context_.set<std::unique_ptr<Context<Target>>>(std::move(ctx));
  }

  template <typename T>
  void SetParam(T param) {
    param_.set<T>(param);
  }

  template <typename Param>
  Param& param() const {
    return param_.get<Param>();
  }

  void set_op_type(const std::string& type) { op_type_ = type; }
  const std::string& op_type() const { return op_type_; }

  void Torch() {}

  virtual TargetType target() const = 0;
  virtual PrecisionType precision() const = 0;
  virtual DataLayoutType layout() const = 0;

  virtual std::string name() const = 0;

  virtual ~KernelBase() = default;

 protected:
  core::any_context_t context_;
  mutable operators::param_t param_;
  // The corresponding op type.
  std::string op_type_;
};

/*
 * ParamType is used to represent a data type of a parameter for the kernel. It
 * can represent any Variable data type.
 * The element_type_hash is the hash code of the element, it should be
 * registered in the `TypeSystem`.
 */
struct ParamType {
  // For unsupported types.
  size_t element_type_hash{};
  Place tensor_place{};
  const Type* type_;

  explicit ParamType() = default;
  explicit ParamType(size_t element_type_hash)
      : element_type_hash(element_type_hash) {}
  ParamType(size_t element_type_hash, const Place& place)
      : element_type_hash(element_type_hash), tensor_place(place) {}
  ParamType(const Type* type) : type_(type) {}
};

/*
 * The data types of kernel parameters. It is used to track the type of kernel's
 * inputs and outputs.
 */
struct ParamTypeRecorder {
  std::map<std::string, ParamType> inputs;
  std::map<std::string, ParamType> outputs;

  void RegisterInputType(const std::string& arg_name, const ParamType& type) {
    Register(&inputs, arg_name, type);
  }

  void RegisterOutputType(const std::string& arg_name, const ParamType& type) {
    Register(&outputs, arg_name, type);
  }

 private:
  void Register(std::map<std::string, ParamType>* ts,
                const std::string& arg_name, ParamType type) {
    (*ts)[arg_name] = type;
  }
};

/*
 * The ParamTypeRegistry help register the input and output data types for all
 * the kernels. It is made singleton so that all the objects of the same kernel
 * can share the same information.
 *
 * Usage:
 * for register a kernel for FC operator.
 * ParamTypeRegistry::Global().Register(
 *        "fc", {TARGET(kCUDA), PRECISION(kFloat)}, 0,
 *        {typeid(Tensor), {TARGET(kCUDA)}});
 */
class ParamTypeRegistry {
 public:
  enum class IO : int { kInput = 0, kOutput };

  template <TargetType target, PrecisionType precision,
            DataLayoutType layout = DataLayoutType::kNCHW>
  /*
   * Helper class for registering a ParamType for a Kernel.
   * Usage:
   *
   * NewInstance<TARGET(kHost), PRECISION(kFloat)>("fc")
   *   .BindInput(0, {typeid(Tensor).hash_code(), {TARGET(kHost)})
   *   .BindInput(1, {typeid(Tensor).hash_code(), {TARGET(kHost),
   *                                               PRECISION(kFloat)});
   */
  struct NewInstance {
    explicit NewInstance(const std::string& kernel_type)
        : kernel_type_(kernel_type) {}

    NewInstance& BindInput(const std::string& arg_name,
                           const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kInput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }
    NewInstance& BindOutput(const std::string& arg_name,
                            const ParamType& ptype) {
      ParamTypeRegistry::Global().Register<IO::kOutput>(
          kernel_type_, Place{target, precision, layout}, arg_name, ptype);
      return *this;
    }

    bool Finalize() { return true; }

   private:
    std::string kernel_type_;
  };

  template <IO io>
  void Register(const std::string& kernel_type, const Place& place,
                const std::string& arg_name, ParamType data_type) {
    KernelIdTy key{kernel_type, place, io, arg_name};
    types_[key] = data_type;
  }

  ParamType Retrive(const Place& place, int offset);

  static ParamTypeRegistry& Global() {
    static ParamTypeRegistry x;
    return x;
  }

 private:
  ParamTypeRegistry() = default;

 public:
  // Identification for a Kernel.
  struct KernelIdTy {
    std::string kernel_type;
    Place place;
    IO io;
    std::string arg_name;
  };

  using key_t = KernelIdTy;
  struct KeyCmp {
    bool operator()(const key_t& a, const key_t& b) const;
  };

 private:
  std::map<key_t, ParamType, ParamTypeRegistry::KeyCmp> types_;
};

// Light-weight kernel implementation.
// The OpKernel is designed to implement the specific algorithm on a target
// device.
template <TargetType Target, PrecisionType Precision,
          DataLayoutType DataLayout = DataLayoutType::kNCHW>
class OpKernel : public KernelBase {
 public:
  // Set runtime context.
  void SetContext(std::unique_ptr<KernelContext>&& ctx) { ctx_ = ctx; }

  // Run the kernel.
  virtual void Run() { CHECK(false) << "Not Implemented"; }

  TargetType target() const override { return Target; }
  PrecisionType precision() const override { return Precision; }
  DataLayoutType layout() const override { return DataLayout; }
  std::string name() const override {
    return op_type() + ":" + TargetToStr(Target) + "/" +
           PrecisionToStr(Precision) + "/" + DataLayoutToStr(DataLayout);
  }

  void Touch() {}

  OpKernel() = default;
  virtual ~OpKernel() = default;

 protected:
  std::unique_ptr<KernelContext> ctx_;
};

}  // namespace lite
}  // namespace paddle
