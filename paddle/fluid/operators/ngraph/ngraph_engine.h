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

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/operator.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace operators {

typedef enum {                /* nGraph support state on ops          */
               FULL_TRAIN,    /* Support full ops for train           */
               PARTIAL_TRAIN, /* Support partial ops for train        */
               FULL_TEST,     /* Support full list of ops for test    */
               PARTIAL_TEST   /* Support partial list of ops for test */
} op_state;

// perform graph build through bridge and execute computation
class NgraphEngine {
 public:
  explicit NgraphEngine(
      const framework::Scope& scope, const platform::Place& place,
      const std::vector<std::shared_ptr<framework::OperatorBase>>& ops,
      const std::unordered_map<std::string, ngraph::element::Type>&
          var_type_map,
      const std::unordered_set<std::string>& persist,
      const std::unordered_set<std::string>& fetches,
      const std::unordered_set<std::string>& post_op_inputs,
      op_state ng_op_state)
      : scope_(scope),
        place_(place),
        fused_ops_(ops),
        var_type_map_(var_type_map),
        persistables_(persist),
        fetches_(fetches),
        post_op_inputs_(post_op_inputs),
        ng_op_state_(ng_op_state) {
    var_in_node_map_ = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

    var_node_map_ = std::make_shared<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>();

    BuildNgIO();

    GetNgFunction();
  }

  void Run(const framework::Scope& scope, const platform::Place& place) const;

 private:
  static std::unordered_map<std::string, std::shared_ptr<ngraph::Function>>
      func_cache_;
  const framework::Scope& scope_;
  const platform::Place& place_;
  std::vector<std::shared_ptr<framework::OperatorBase>> fused_ops_;
  std::unordered_map<std::string, ngraph::element::Type> var_type_map_;
  std::unordered_set<std::string> persistables_;
  std::unordered_set<std::string> fetches_;
  std::unordered_set<std::string> post_op_inputs_;
  op_state ng_op_state_;

  // ngraph backend eg. CPU
  static std::shared_ptr<ngraph::runtime::Backend> backend_;
  // ngraph function to call and execute
  std::shared_ptr<ngraph::Function> ngraph_function_;
  // var_name of inputs
  std::vector<std::string> var_in_;
  // var_name of outputs from  fetch in order
  std::vector<std::string> var_out_;
  // map input vars to nodes
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_in_node_map_;
  // map each var name with a ngraph node
  std::shared_ptr<
      std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
      var_node_map_;
  // cache key to check if function is cached
  std::shared_ptr<std::string> GetCacheKey();
  // get ngraph input and define ngraph input parameters
  void GetNgInputShape(std::shared_ptr<framework::OperatorBase> op);
  // Call ngraph bridge to map ops
  void BuildNgNodes();
  // get the ngraph input and output var list
  void BuildNgIO();
  // build ngraph function call
  void BuildNgFunction();
  // Check cache for ngraph function or otherwise build the function
  void GetNgFunction();
};

}  // namespace operators
}  // namespace paddle
