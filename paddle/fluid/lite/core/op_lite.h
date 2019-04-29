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

#include <glog/logging.h>
#include <boost/variant.hpp>
#include <map>
#include <memory>
#include <string>
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/lite/core/context.h"
#include "paddle/fluid/lite/core/kernel.h"
#include "paddle/fluid/lite/core/scope.h"
#include "paddle/fluid/lite/model_parser/compatible_pb.h"

namespace paddle {
namespace lite {

using any_t = boost::variant<int, float, framework::Variable *>;
using anys_t = std::map<std::string, any_t>;

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
  // The strategies to pick a kernel from candidates.
  enum class KernelStrategy {
    // Return the user specified one.
    kStatic = 0,
    // Specify the expected kernel externally.
    kSpecified,
    // Run each kernel to evaluate and get the best kernel.
    kRuntime,
  };

  OpLite() = default;
  OpLite(const std::string &type) : op_type_(type) {}
  OpLite(const std::vector<Place> &valid_places)
      : valid_places_(valid_places) {}

  void SetValidPlaces(const std::vector<Place> &places) {
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
  bool Attach(const OpDesc &opdesc, lite::Scope *scope);

  const OpInfo *op_info() const { return op_info_.get(); }
  OpInfo *mutable_op_info() { return op_info_.get(); }

  // Human-readable information.
  virtual std::string DebugString() const = 0;

  const Place &kernel_place() const { return kernel_place_; }

  // NOTE This might be discarded.
  void PickKernel(const std::vector<Place> &valid_places,
                  KernelStrategy kernel_strategy = KernelStrategy::kStatic);

  // Create all the kernels for the valid targets.
  std::vector<std::unique_ptr<KernelBase>> CreateKernels(
      const std::vector<Place> &places, const std::string &kernel_type = "");

  lite::Scope *scope() { return scope_; }

  // Assign op param to kernel.
  virtual void AttachKernel(KernelBase *kernel) = 0;

  virtual ~OpLite() = default;

 protected:
  // Attach it with the runtime environment.
  virtual bool AttachImpl(const OpDesc &opdesc, lite::Scope *scope) = 0;

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
class OpInfo {
 public:
  // To avoid the bugs from legancy framework::OpDesc, we use the ProtoBuf
  // message instead.
  void Build(const framework::proto::OpDesc &desc) {
    ExtractInputsAndOutputs(desc);
    CollectInputAndOutputArgnames(desc);
    CollectArguments(desc);
    desc_.reset(new framework::proto::OpDesc(desc));
  }

  const framework::proto::OpDesc &desc() const {
    CHECK(desc_) << "desc has't set";
    return *desc_;
  }
  framework::proto::OpDesc *mutable_desc() { return desc_.get(); }
  const std::list<std::string> &input_names() const { return input_names_; }
  const std::list<std::string> &output_names() const { return output_names_; }
  const std::map<std::string, std::list<std::string>> &input_argument() const {
    return input_argument_;
  }
  const std::map<std::string, std::list<std::string>> &output_argument() const {
    return output_argument_;
  }
  bool GetInputArgname(const std::string &value_name, std::string *out) const;
  bool GetOutputArgname(const std::string &value_name, std::string *out) const;

  const std::list<std::string> &input_argnames() const {
    return input_argnames_;
  }
  const std::list<std::string> &output_argnames() const {
    return output_argnames_;
  }

 private:
  void ExtractInputsAndOutputs(const framework::proto::OpDesc &opdesc) {
    for (const auto &item : opdesc.inputs()) {
      for (const auto &x : item.arguments()) {
        input_names_.push_back(x);
      }
    }
    for (const auto &item : opdesc.outputs()) {
      for (const auto &x : item.arguments()) {
        output_names_.push_back(x);
      }
    }
  }

  void CollectInputAndOutputArgnames(const framework::proto::OpDesc &opdesc) {
    for (const auto &item : opdesc.inputs()) {
      input_argnames_.push_back(item.parameter());
    }
    for (const auto &item : opdesc.outputs()) {
      output_argnames_.push_back(item.parameter());
    }
  }

  void CollectArguments(const framework::proto::OpDesc &opdesc) {
    for (const auto &item : opdesc.inputs()) {
      for (auto &x : item.arguments()) {
        input_argument_[item.parameter()].push_back(x);
      }
    }
    for (const auto &item : opdesc.outputs()) {
      for (auto &x : item.arguments()) {
        output_argument_[item.parameter()].push_back(x);
      }
    }
  }

 private:
  std::list<std::string> input_names_;
  std::list<std::string> output_names_;
  std::list<std::string> input_argnames_;
  std::list<std::string> output_argnames_;
  std::map<std::string, std::list<std::string>> input_argument_;
  std::map<std::string, std::list<std::string>> output_argument_;
  // NOTE too heavy.
  std::unique_ptr<framework::proto::OpDesc> desc_;
};

}  // namespace lite
}  // namespace paddle
