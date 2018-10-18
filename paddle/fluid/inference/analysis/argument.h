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

/*
 * This file defines the class Argument, which is the input and output of the
 * analysis module. All the fields that needed either by Passes or PassManagers
 * are contained in Argument.
 *
 * TODO(Superjomn) Find some way better to contain the fields when it grow too
 * big.
 */

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ir::Graph;

/*
 * The argument definition of both Pass and PassManagers.
 *
 * All the fields should be registered here for clearness.
 */
struct Argument {
  Argument() = default;
  explicit Argument(const std::string& model_dir) { SetModelDir(model_dir); }

// Declare an argument field with getter and setter.
#define DECL_ARGUMENT_FIELD(field__, Field, type__) \
  DECL_ARGUMENT_GETTER(field__, type__);            \
  void Set##Field(const type__& x) {                \
    if (!Has(#field__)) {                           \
      Set<type__>(#field__, new type__(x));         \
    } else {                                        \
      *field__() = x;                               \
    }                                               \
  }
#define DECL_ARGUMENT_GETTER(field__, type__) \
  type__* field__() { return Has(#field__) ? &Get<type__>(#field__) : nullptr; }

#define DECL_ARGUMENT_UNIQUE_FIELD(field__, Field, type__) \
  DECL_ARGUMENT_GETTER(field__, type__)                    \
  void Set##Field(std::unique_ptr<type__>&& x) {           \
    if (Has(#field__)) {                                   \
      Release<type__>(#field__);                           \
    }                                                      \
    Set<type__>(#field__, x.release());                    \
  }

  // Model path
  DECL_ARGUMENT_FIELD(model_dir, ModelDir, std::string);
  // Model specified with program and parameters files.
  DECL_ARGUMENT_FIELD(model_program_path, ModelProgramPath, std::string);
  DECL_ARGUMENT_FIELD(model_params_path, ModelParamsPath, std::string);

  // The overall graph to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(main_graph, MainGraph, Graph);
  // The overall Scope to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(scope, Scope, framework::Scope);

  DECL_ARGUMENT_UNIQUE_FIELD(main_program, MainProgram, framework::ProgramDesc);

  // The ir passes to perform in analysis phase.
  DECL_ARGUMENT_FIELD(ir_analysis_passes, IrAnalysisPasses,
                      std::vector<std::string>);

  DECL_ARGUMENT_FIELD(use_gpu, UseGPU, bool);
  DECL_ARGUMENT_FIELD(use_tensorrt, UseTensorRT, bool);
  DECL_ARGUMENT_FIELD(tensorrt_node_teller, TensorRtNodeTeller,
                      std::function<bool(const framework::ir::Node*)>);

  // The graph that process by the Passes or PassManagers.
  // std::unique_ptr<Graph> main_graph;

  // The original program desc.
  std::unique_ptr<framework::proto::ProgramDesc> original_program_desc;

  // The processed program desc.
  std::unique_ptr<framework::proto::ProgramDesc> transformed_program_desc;

  // Support for any other attributes.
  template <typename T>
  void Set(const std::string& key, T* data) {
    PADDLE_ENFORCE_NOT_NULL(data);
    PADDLE_ENFORCE(!attrs_.count(key), "Duplicate set Argument's attr [%s]",
                   key);
    attrs_[key] = data;
    attr_deleters_[key] = [data, key]() {
      VLOG(3) << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
      VLOG(3) << "argument delete attr: " << key;
      delete data;
    };
  }

  bool Has(const std::string& name) const { return attrs_.count(name); }

  template <typename T>
  T* Release(const std::string& key) {
    PADDLE_ENFORCE(attrs_.count(key));
    auto* res = boost::any_cast<T*>(attrs_.at(key));
    attrs_.erase(key);
    attr_deleters_.erase(key);
    return res;
  }

  template <typename T>
  T& Get(const std::string& key) {
    PADDLE_ENFORCE(Has(key));
    return *boost::any_cast<T*>(attrs_.at(key));
  }

  ~Argument() {
    for (auto& item : attr_deleters_) {
      item.second();
    }
  }

 private:
  std::unordered_map<std::string, boost::any> attrs_;
  std::unordered_map<std::string, std::function<void()>> attr_deleters_;
};

#define ARGUMENT_CHECK_FIELD(argument__, fieldname__) \
  PADDLE_ENFORCE(argument__->fieldname__(),           \
                 "the argument field [%s] should be set", #fieldname__);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
