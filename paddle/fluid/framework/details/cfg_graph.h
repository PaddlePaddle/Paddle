#pragma once
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <list>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"


namespace paddle {
namespace framework {
namespace details {

class ControlFlowGraph{
public:
  explicit ControlFlowGraph(std::unique_ptr<ir::Graph> graph) ;
  void DataAnalysis();
  void UpdateGraph(ir::Node* old_node, ir::Node* new_node, int beign_idx);

  const std::unordered_set<ir::Node*>& LiveIn(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& LiveOut(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Def(ir::Node* op) const;
  const std::unordered_set<ir::Node*>& Use(ir::Node* op) const;
  const std::vector<ir::Node*>& Ops() const;
private:
  typedef std::unordered_map<ir::Node*, std::list<ir::Node*>> NodeListType;
  typedef std::unordered_map<ir::Node*, std::unordered_set<ir::Node*>> NodeSetType;

  std::vector<ir::Node*> ops_; // topology sort ops
  NodeListType successors_;
  NodeListType predecessors_;
  NodeSetType live_in_;
  NodeSetType live_out_;
  NodeSetType uses_;
  NodeSetType defs_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
