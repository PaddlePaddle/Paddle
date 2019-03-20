/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace operators {

enum class OpState {                /* nGraph support state on ops          */
                     FULL_TRAIN,    /* Support full ops for train           */
                     PARTIAL_TRAIN, /* Support partial ops for train        */
                     FULL_TEST,     /* Support full list of ops for test    */
                     PARTIAL_TEST,  /* Support partial list of ops for test */
                     UNKNOWN        /* Output all for debug purpose         */
};

// cache engine repetitives
struct EngineCache {
  std::shared_ptr<ngraph::Function> ngraph_function;
  std::set<std::string> persistables;
  std::vector<std::string> var_in;
  std::vector<std::string> var_out;
  std::vector<size_t> var_in_updates;
  bool is_test = true;
};

// perform graph build through bridge and execute computation
class NgraphEngine {
 public:
  explicit NgraphEngine(const framework::Scope& scope,
                        const platform::Place& place,
                        const framework::ExecutionContext& ctx);

  void Run(const framework::Scope& scope, const platform::Place& place) const;

  static const framework::BlockDesc* p_bdesc;
  static std::vector<std::string> feed_vars, fetch_vars;

  static void FuseNgraphOps(
      const framework::BlockDesc& prog,
      std::vector<std::unique_ptr<framework::OperatorBase>>* ops);

 private:
  static std::unordered_map<std::string, EngineCache> engine_cache;
  static std::unordered_map<
      std::string, std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>
      t_in_cache_;
  static framework::Variable* pre_var_ptr;

  const framework::Scope& scope_;
  const platform::Place& place_;
  std::vector<std::shared_ptr<framework::OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::set<std::string> persistables_;
  std::unordered_set<std::string> post_op_inputs_;
  OpState op_state_ = OpState::UNKNOWN;
  bool is_test_{true};
  std::string func_cache_key_;

  // ngraph backend eg. CPU
  static std::shared_ptr<ngraph::runtime::Backend> backend_;
  // ngraph function to call and execute
  std::shared_ptr<ngraph::Function> ngraph_function_;
  // var_name of inputs
  std::vector<std::string> var_in_;
  // var_name of outputs from  fetch in order
  std::vector<std::string> var_out_;
  // non-persitable var_in
  std::vector<size_t> var_in_updates_;
  // map input vars to nodes
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_in_node_map_;
  // map each var name with a ngraph node
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_node_map_;
  // prepare info for ngraph engine need
  void Prepare(const std::vector<int>& interval);
  // get ngraph engine input and output list
  void BuildNgIO(const std::vector<framework::OpDesc*>& op_descs,
                 const std::vector<int>& interval);
  // get ngraph input and define ngraph input parameters
  void GetNgInputShape();
  // Call ngraph bridge to map ops
  void BuildNgNodes();
  // run paddle RuntimeInferShape to get the tensor shape
  void RunInferShape();
  // build ngraph function call
  void BuildNgFunction(const std::vector<int>& interval);
  // Check cache for ngraph function or otherwise build the function
  void GetNgFunction(std::string engine_key, const std::vector<int>& interval);
};

}  // namespace operators
}  // namespace paddle
