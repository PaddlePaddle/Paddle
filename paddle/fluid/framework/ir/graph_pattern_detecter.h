#pragma once

#include <numeric>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

// Some basic torminolygies:
//   - Pattern: a pattern defined as a graph.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;

  uint64_t id{std::numeric_limits<uint64_t>::max()};
  std::vector<PDNode*> inlinks;
  std::vector<PDNode*> outlinks;
  teller_t teller;
};

class PDPattern {
 public:
  using edge_t = std::pair<PDNode*, PDNode*>;

  void AddEdge(PDNode* a, PDNode* b);

  PDNode* NewNode();

  const std::vector<PDNode>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

 private:
  std::vector<PDNode> nodes_;
  std::vector<edge_t> edges_;
};

/*
 * GraphPatternDetecter helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 */
class GraphPatternDetecter {
 public:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  using subgraph_t = std::vector<hit_rcd_t>;

  // Operate on the detected pattern.
  using handle_t = std::function<void(
      const std::vector<hit_rcd_t>& /*hitted pattern*/, Graph*)>;

  void operator()(handle_t handler);

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const ir::Graph& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

 private:
  PDPattern pattern_;
  std::vector<hit_rcd_t> marked_records_;
  std::unordered_map<const PDNode*, std::unordered_set<Node*>> pdnodes2nodes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
