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

#include <queue>
#include "graph_viz_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

const static char kFeedsAttr[] = "__feed_names__";
const static char kFetchesAttr[] = "__fetch_names__";
/*
 * This Pass is used to automatically prune the graph and insert feed and fetch
 * ops. Currently, it is used in inference scenarios.
 * NOTE It also suffers from the naming bug, that is the operator's output
 * variable have the same name of input, so the prune algorithm can't tell
 * whether to keep the input or output.
 * TODO(Superjomn) Fix that bug.
 */
class FeedFetchAutoPrunePass final : public Pass {
 public:
  ~FeedFetchAutoPrunePass() = default;

 protected:
  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override;

 private:
  // Remove the original feed fetch operators.
  bool RemoveFeedFetchOps(Graph *graph) const;

  std::set<Node *> DFS(
      const std::vector<Node *> &starts, const Graph &graph,
      std::function<const std::vector<Node *> &(Node *)> &&get_nexts) const;

  std::set<Node *> ExtractNodesByName(
      const std::unordered_set<std::string> &names, const Graph &graph) const;

  // Extract the nodes that starts from `feeds` and retches `fetches`.
  bool Prune(Graph *graph, const std::vector<std::string> &feeds,
             const std::vector<std::string> &fetches) const;

  std::set<Node *> ExtractDependedNodes(const std::vector<std::string> &feeds,
                                        const std::vector<std::string> &fetches,
                                        Graph *graph) const;

  bool InsertFeedFetchOps(Graph *graph, const std::vector<std::string> &feeds,
                          const std::vector<std::string> &fetches) const;

  bool DetectFeedFetchOps(Graph *graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
