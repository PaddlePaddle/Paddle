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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/paddle/cpp/op_desc.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/utils/registry.h"

namespace cinn {
namespace frontend {

namespace paddle {
// NOTE: Please Ensure the GradVarName function's definition is the
// same as Paddle's!!! The definition ref to
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L97
inline std::string GradVarName(const std::string& var_name) {
  constexpr char kGradVarSuffix[] = "@GRAD";
  constexpr size_t kGradVarSuffixSize = 5U;

  std::string result;
  result.reserve(var_name.size() + kGradVarSuffixSize);
  result += var_name;
  result += kGradVarSuffix;
  return result;
}

// Only used for rename the output of inplace var!
// Trick solution to fix CINN not support inplace op problem.
constexpr char InplaceOutSuffix[] = "@InplaceOut";
}  // namespace paddle

class OpMapperContext {
 public:
  OpMapperContext(
      const hlir::framework::Scope& scope,
      const common::Target& target,
      NetBuilder* builder,
      std::unordered_map<std::string, Variable>* var_map,
      std::unordered_map<std::string, std::string>* var_model_to_program_map,
      std::unordered_set<std::string>* fetch_var_names)
      : scope_(scope),
        target_(target),
        builder_(builder),
        var_map_(var_map),
        var_model_to_program_map_(var_model_to_program_map),
        fetch_var_names_(fetch_var_names) {
    CHECK_NOTNULL(builder_);
    CHECK_NOTNULL(var_map_);
    CHECK_NOTNULL(var_model_to_program_map_);
    CHECK_NOTNULL(fetch_var_names_);
  }

  const auto& Scope() const { return scope_; }

  const auto& Target() const { return target_; }

  NetBuilder* Builder() const { return builder_; }

  // add Variable into local var_map
  void AddVar(const std::string& name,
              const Variable& var,
              bool can_inplace = true) const;

  // get Variable from local var_map or scope
  Variable GetVar(const std::string& name) const;

  // add map from paddle name to cinn name into var_model_to_program_map
  void AddVarModelToProgram(const std::string& name,
                            const std::string& id,
                            bool can_inplace = true) const;

  void AddFetchVarName(const std::string& name) const;

  struct FeedInfo {
    std::vector<int> shape;
    common::Type type;
  };

  void AddFeedInfo(const std::string& name, const FeedInfo& info);

  const FeedInfo& GetFeedInfo(const std::string& name) const;

 private:
  const hlir::framework::Scope& scope_;
  const common::Target& target_;
  NetBuilder* builder_{nullptr};

  std::unordered_map<std::string, Variable>* var_map_{nullptr};
  // map from var in Paddle model to var name in program.
  std::unordered_map<std::string, std::string>* var_model_to_program_map_{
      nullptr};
  // fetch var names used in Paddle
  std::unordered_set<std::string>* fetch_var_names_{nullptr};

  std::unordered_map<std::string, FeedInfo> feed_info_map_;
};

class OpMapper {
 public:
  using OpMapperFunc =
      std::function<void(const paddle::cpp::OpDesc&, const OpMapperContext&)>;

  OpMapper() = default;

  OpMapper& Set(const OpMapperFunc& kernel) {
    kernel_ = kernel;
    return *this;
  }
  void Run(const paddle::cpp::OpDesc& op_desc,
           const OpMapperContext& ctx) const {
    kernel_(op_desc, ctx);
  }

  std::string name;

 private:
  OpMapperFunc kernel_;
};

class OpMapperRegistry : public Registry<OpMapper> {
 public:
  OpMapperRegistry() = default;

 private:
  CINN_DISALLOW_COPY_AND_ASSIGN(OpMapperRegistry);
};

#define UNIQUE_OPMAPPER_NAME(OpName) \
  static ::cinn::frontend::OpMapper& __op_mapper_registrar_##OpName

#define CINN_REGISTER_OP_MAPPER(OpName, Kernel)                \
  CINN_STR_CONCAT(UNIQUE_OPMAPPER_NAME(OpName), __COUNTER__) = \
      ::cinn::frontend::OpMapperRegistry::Global()             \
          ->__REGISTER_OR_GET__(#OpName)                       \
          .Set(Kernel);

}  // namespace frontend
}  // namespace cinn
