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

#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/model_parser/cpp/op_desc.h"

namespace paddle {
namespace lite {

// For registry factory.
struct Registry {
  void Touch() {}
};

namespace mir {
class Node;
class SSAGraph;
}

class OpInfo;

/**
 * The base class of an light-weight operators, currently just used in inference
 * to eliminate overhead of some operations in current framework.
 *
 * The Operator are designed as follows:
 * - it can has some members to hold the argument and some other computation
 * resources,
 * - it should act like a function call, no more logic included.
 */
class OpLite : public Registry {
 public:
  OpLite() = default;
  explicit OpLite(const std::string &type) : op_type_(type) {}
  explicit OpLite(const std::vector<Place> &valid_places)
      : valid_places_(valid_places) {}

  void SetValidPlaces(const std::vector<Place> &places) {
    VLOG(3) << "valid places " << valid_places_.size();
    valid_places_ = places;
  }
  const std::vector<Place> &valid_places() const { return valid_places_; }
  // Check the shape.
  virtual bool CheckShape() const { return true; }
  // Inference the outputs' shape.
  virtual bool InferShape() const { return true; }
  // Run this operator.
  virtual bool Run();

  // Link the external execution environ to internal context.
  bool Attach(const cpp::OpDesc &opdesc, lite::Scope *scope);

  const OpInfo *op_info() const { return op_info_.get(); }
  OpInfo *mutable_op_info() { return op_info_.get(); }

  // Human-readable information.
  virtual std::string DebugString() const = 0;

  const Place &kernel_place() const { return kernel_place_; }

  // Create all the kernels for the valid targets.
  std::vector<std::unique_ptr<KernelBase>> CreateKernels(
      const std::vector<Place> &places, const std::string &kernel_type = "");

  lite::Scope *scope() { return scope_; }

  // Assign op param to kernel.
  virtual void AttachKernel(KernelBase *kernel) = 0;

  virtual ~OpLite() = default;

 protected:
  // Attach it with the runtime environment.
  virtual bool AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) = 0;

  // Specify the kernel to run by default. This will specify the value of
  // `kernel_place_`.
  virtual void StaticPickKernel(const std::vector<Place> &valid_targets) {
    auto kernels = CreateKernels(valid_targets);
    kernel_ = std::move(kernels.front());
  }

  // Wait until all the inputs' events are ready.
  void SyncInputEvents() {}

  // Record the output events, and that will tell all the dependent operators
  // some inputs are ready.
  void RecordOutputEvents() {}

  const Tensor *GetTensor(lite::Scope *scope, const std::string &name) const;
  Tensor *GetMutableTensor(lite::Scope *scope, const std::string &name) const;

  friend class mir::Node;
  friend class mir::SSAGraph;

 protected:
  // some helper functions.
  template <typename T>
  const T *GetVar(Scope *scope, const std::string &name) {
    auto *var = scope->FindVar(name);
    CHECK(var) << "No var found for " << name;
    return &var->Get<T>();
  }
  template <typename T>
  T *GetMutableVar(Scope *scope, const std::string &name) {
    auto *var = scope->FindVar(name);
    CHECK(var) << "No var found for " << name;
    return var->GetMutable<T>();
  }

 protected:
  lite::Scope *scope_{};
  std::unique_ptr<KernelBase> kernel_;
  std::string op_type_;
  std::vector<Place> valid_places_;
  Place kernel_place_{TARGET(kHost), PRECISION(kFloat)};
  std::unique_ptr<OpInfo> op_info_;
};

/*
 * Operator Information, such as some description. It will be shared by all the
 * kernels of the same operator.
 */
class OpInfo : public cpp::OpDesc {
 public:
  OpInfo(const OpInfo &) = default;
  explicit OpInfo(const cpp::OpDesc &other) : cpp::OpDesc(other) {}

  // Collect all the input variable's name.
  std::vector<std::string> input_names() const {
    std::vector<std::string> res;
    for (auto &param : InputArgumentNames()) {
      for (auto &x : Input(param)) {
        res.push_back(x);
      }
    }
    return res;
  }

  // Collect all the output variable's name.
  std::vector<std::string> output_names() const {
    std::vector<std::string> res;
    for (auto &param : OutputArgumentNames()) {
      for (auto &x : Output(param)) {
        res.push_back(x);
      }
    }
    return res;
  }

  std::vector<std::string> input_argnames() const {
    return InputArgumentNames();
  }

  std::vector<std::string> output_argnames() const {
    return OutputArgumentNames();
  }

  bool GetInputArgname(const std::string &value_name, std::string *out) const {
    for (auto &item : inputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = item.first;
        return true;
      }
    }
    return false;
  }
  bool GetOutputArgname(const std::string &value_name, std::string *out) const {
    for (auto &item : outputs_) {
      auto it = std::find(item.second.begin(), item.second.end(), value_name);
      if (it != item.second.end()) {
        *out = item.first;
        return true;
      }
    }
    return false;
  }

  void UpdateAllInputs(const std::string &from, const std::string &to) {
    for (auto &item : inputs_) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }

  void UpdateAllOutputs(const std::string &from, const std::string &to) {
    for (auto &item : outputs_) {
      for (auto &var : item.second) {
        if (var == from) var = to;
      }
    }
  }
};

}  // namespace lite
}  // namespace paddle
