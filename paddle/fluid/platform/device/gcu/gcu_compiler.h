/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/gcu_info.h"
#include "paddle/fluid/platform/device/gcu/gcu_strategy.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/device/gcu/utils/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace platform {
namespace gcu {

class GcuCompiler {
 public:
  GcuCompiler() = default;
  ~GcuCompiler() = default;
  void ConvertGraph(const Graph *graph,
                    const std::vector<std::string> &feed_list,
                    const std::vector<std::string> &fetch_list,
                    std::vector<uint64_t> &input_sizes,    // NOLINT
                    std::vector<uint64_t> &output_sizes);  // NOLINT
  void Compile(const std::vector<const Graph *> &graph_list,
               const std::vector<std::vector<std::string>> &all_feeds,
               const std::vector<std::vector<std::string>> &all_fetches,
               const std::string &program_key = "default_program_key");
  void PostProcess(const std::vector<const Graph *> &before_graph_list,
                   const Graph *post_graph);
  ResourceReflection GetReflectionInfo() { return reflection_; }
  std::map<std::string, std::vector<int64_t>> GetVarFixedMap();

 private:
  std::vector<std::set<std::string>> GetOptimizerLinkageVar(const NodePtr node);
  std::vector<std::string> GetInOrOutVarByArcheType(
      const NodePtr node, const std::string &archetype);
  void Preprocess(const Graph *&graph,  // NOLINT
                  const std::vector<std::string> &feed_list,
                  const std::vector<std::string> &fetch_list);
  GcuOpPtr CreateInput(const Node *node,
                       const GraphType &graph_type,
                       int64_t &size);  // NOLINT
  GcuOpPtr AddGteOp(const Node *node, const GcuOpPtr &input);
  void SetOutput(const std::vector<std::string> &fetch_list,
                 const GraphType &graph_type,
                 std::vector<uint64_t> &output_sizes);  // NOLINT
  bool IsRefNode(const Node *node);
  bool IsTrainingGraph(const std::vector<const Graph *> &graph_list);
  int32_t SetOutputWithFetchList(
      const std::vector<std::string> &fetch_list,
      const GraphType &graph_type,
      std::vector<std::vector<int64_t>> &tuple_shape,      // NOLINT
      std::vector<::builder::PrimitiveType> &tuple_dtype,  // NOLINT
      std::vector<GcuOp> &outputs);                        // NOLINT
  bool SetOutputWithLeafNodes(
      const GraphType &graph_type,
      std::vector<std::vector<int64_t>> &tuple_shape,      // NOLINT
      std::vector<::builder::PrimitiveType> &tuple_dtype,  // NOLINT
      std::vector<GcuOp> &outputs);                        // NOLINT
  void RecordWeightsToTrans();
  void Reset();

 private:
  std::set<Node *> feed_list_;
  GcuBuilderPtr builder_ = nullptr;
  std::map<std::string, NodePtr> var_node_cache_;
  std::map<std::string, GcuOpPtr> gcu_node_cache_;
  std::set<std::string> leaf_var_nodes_;
  std::set<std::string> transed_var_nodes_;
  ResourceReflection reflection_;
  // for training param update
  std::map<std::string, GcuOpPtr> weights_name_to_gcuop_;
  // {input idx, input size}
  std::map<std::string, std::pair<size_t, int64_t>> var_name_to_input_;
  std::map<std::string, PaddleVarDesc> var_to_pd_var_desc_;
  bool is_training_graph_ = false;
  size_t counter_ = 0;
  std::map<std::string, GcuTransInfo> trans_infos_;
  std::string running_mode_ = RunningMode::SERIAL;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
