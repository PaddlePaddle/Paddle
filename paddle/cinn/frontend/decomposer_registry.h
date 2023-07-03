// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <functional>
#include <string>
#include <unordered_map>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {

class Decomposer;

class DecomposerContext {
 public:
  explicit DecomposerContext(
      NetBuilder* builder, absl::flat_hash_map<std::string, Variable>* var_map)
      : builder_(builder), var_map_(var_map) {}

  NetBuilder* builder() const { return builder_; }

  // Map the new var to the original var.
  void MapOutToOrigin(const Variable& new_var, const Variable& ori_var) const {
    if (new_var->shape != ori_var->shape) {
      LOG(FATAL)
          << "The output shape should be equal to the original. But received : "
          << new_var->id << ".shape=[" << utils::Join(new_var->shape, ", ")
          << "] and the original var " << ori_var->id << ".shape=["
          << utils::Join(ori_var->shape, ", ") << "].";
    }
    if (new_var->type != ori_var->type) {
      LOG(FATAL)
          << "The output type shoule be equal to the original. But received : "
          << new_var->id << ".type=" << new_var->type
          << " and the original var " << ori_var->id
          << ".type=" << ori_var->type;
    }
    (*var_map_)[new_var->id] = ori_var;
  }

 private:
  NetBuilder* builder_{nullptr};
  absl::flat_hash_map<std::string, Variable>* var_map_{nullptr};
};

class InstrDecomposerRegistry : public Registry<Decomposer> {
 public:
  static InstrDecomposerRegistry* Global() {
    static InstrDecomposerRegistry x;
    return &x;
  }

  inline const Decomposer* Get(const std::string& op_name,
                               const common::Target& target) {
    const Decomposer* decomposer = Find(op_name, target);
    CHECK(decomposer) << "Decomposer for [" << op_name << ", " << target
                      << "] is not registered";
    return decomposer;
  }

  inline const Decomposer* Find(const std::string& name,
                                const common::Target& target) {
    return Registry<Decomposer>::Find(name + "_" + target.arch_str());
  }

 private:
  InstrDecomposerRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(InstrDecomposerRegistry);
};

class Decomposer {
 public:
  using DecomposerKernel =
      std::function<void(const Instruction& instr, const DecomposerContext&)>;

  Decomposer& SetBody(const DecomposerKernel& kernel) {
    kernel_ = kernel;
    return *this;
  }

  void Run(const Instruction& instr, const DecomposerContext& context) const {
    kernel_(instr, context);
  }

  std::string name;

 private:
  DecomposerKernel kernel_;
};

#define CINN_DECOMPOSER_REGISTER_CORE(name, target, kernel)        \
  ::cinn::frontend::InstrDecomposerRegistry::Global()              \
      ->__REGISTER__(std::string(#name) + "_" + target.arch_str()) \
      .SetBody(kernel)

#define CINN_DECOMPOSER_REGISTER_ALL(name, kernel)                   \
  static std::vector<::cinn::common::Target> all_targets = {         \
      ::cinn::common::DefaultHostTarget(),                           \
      ::cinn::common::DefaultNVGPUTarget()};                         \
  for (auto& target : all_targets) {                                 \
    ::cinn::frontend::InstrDecomposerRegistry::Global()              \
        ->__REGISTER__(std::string(#name) + "_" + target.arch_str()) \
        .SetBody(kernel);                                            \
  }

/**
 * @def CINN_DECOMPOSER_REGISTER
 * \brief Register a decomposer kernel
 *
 * Register a decomposer on the specific target:
 * \code
 *  CINN_DECOMPOSER_REGISTER(name, target, kernel);
 * \endcode
 *
 * Register a decomposer on all default targets:
 * \code
 * CINN_DECOMPOSER_REGISTER(name, kernel);
 * \endcode
 */
#define GET_MACRO(_0, _1, _2, FUNC, ...) FUNC
#define CINN_DECOMPOSER_REGISTER(...)      \
  GET_MACRO(__VA_ARGS__,                   \
            CINN_DECOMPOSER_REGISTER_CORE, \
            CINN_DECOMPOSER_REGISTER_ALL)  \
  (__VA_ARGS__)

}  // namespace frontend
}  // namespace cinn
