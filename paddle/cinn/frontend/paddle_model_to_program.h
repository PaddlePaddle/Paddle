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

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <absl/types/variant.h>
#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/type.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/paddle/cpp/program_desc.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/scope.h"

namespace cinn {
namespace frontend {

class PaddleModelToProgram {
 public:
  explicit PaddleModelToProgram(
      hlir::framework::Scope* scope,
      std::unordered_map<std::string, std::vector<int>> input_shape_map,
      const common::Target& target)
      : scope_(scope),
        input_shape_map_(input_shape_map),
        target_(target),
        net_builder_(new NetBuilder("paddle_model_netbuilder")) {
    CHECK(scope_);

    AddOpMapper_feed();
    AddOpMapper_fetch();
    AddOpMapper_mul();
    AddOpMapper_scale();
    AddOpMapper_relu();
    AddOpMapper_elementwise_add();
    AddOpMapper_elementwise_mul();
    AddOpMapper_elementwise_div();
    AddOpMapper_elementwise_sub();
    AddOpMapper_conv2d();
    AddOpMapper_batchnorm();
    AddOpMapper_pool2d();
    AddOpMapper_softmax();
    AddOpMapper_relu6();
    AddOpMapper_depthwise_conv2d();
    AddOpMapper_sigmoid();
    AddOpMapper_slice();
    AddOpMapper_dropout_infer();
    AddOpMapper_matmul();
    AddOpMapper_reshape2();
    AddOpMapper_concat();
    AddOpMapper_assign();
    AddOpMapper_fill_constant();
    AddOpMapper_transpose2();
    AddOpMapper_exp();
  }

  std::unique_ptr<Program> operator()(const std::string& model_dir,
                                      bool is_combined);

  // Add an Instruction to a program given a Paddle-format \p op_desc.
  void AddOp(const paddle::cpp::OpDesc& op_desc);

  // @{
  void AddOpMapper_feed();
  void AddOpMapper_fetch();
  void AddOpMapper_scale();
  void AddOpMapper_mul();
  void AddOpMapper_relu();
  void AddOpMapper_elementwise_add();
  void AddOpMapper_elementwise_mul();
  void AddOpMapper_elementwise_div();
  void AddOpMapper_elementwise_sub();
  void AddOpMapper_conv2d();
  void AddOpMapper_batchnorm();
  void AddOpMapper_pool2d();
  void AddOpMapper_softmax();
  void AddOpMapper_relu6();
  void AddOpMapper_depthwise_conv2d();
  void AddOpMapper_sigmoid();
  void AddOpMapper_slice();
  void AddOpMapper_dropout_infer();
  void AddOpMapper_matmul();
  void AddOpMapper_reshape2();
  void AddOpMapper_concat();
  void AddOpMapper_assign();
  void AddOpMapper_fill_constant();
  void AddOpMapper_transpose2();
  void AddOpMapper_exp();
  // @}

  const absl::flat_hash_map<std::string, Variable>& var_map() const {
    return var_map_;
  }
  const absl::flat_hash_map<std::string, std::string>&
  var_model_to_program_map() {
    return var_model_to_program_map_;
  }
  const absl::flat_hash_set<std::string>& fetch_names() { return fetch_names_; }

 protected:
  void AddVar(const std::string& name,
              const Variable& var,
              bool replace = false);

  Variable GetVar(const std::string& name);

  void TransposeVar(const std::string& name);

  void ReverseHWVar(const std::string& name);

 private:
  // op mapper
  absl::flat_hash_map<std::string,
                      std::function<void(const paddle::cpp::OpDesc&)>>
      op_mappers_;
  std::unordered_map<std::string, std::vector<int>> input_shape_map_;
  // net builder
  std::unique_ptr<NetBuilder> net_builder_;

  absl::flat_hash_map<std::string, Variable> var_map_;
  absl::flat_hash_set<std::string> fetch_names_;
  // map from var in Paddle model to var name in program.
  absl::flat_hash_map<std::string, std::string> var_model_to_program_map_;
  hlir::framework::Scope* scope_{};
  common::Target target_;
};

}  // namespace frontend
}  // namespace cinn
