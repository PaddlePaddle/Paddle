#pragma once
#include <unordered_map>
#include <unordered_set>
#include <string>

#include "paddle/fluid/framework/"


namespace paddle {
namespace framework {
namespace details {

class ControlFlowGraph : public ir::Graph {
public:
  void ComputeLiveRange()
private:
  typedef std::unordered_map<std::string, std::list<std::string>> DictType;
  typedef std::unordered_map<std::unordered_set<std::string>> SetType;
  DictType successors_;
  DictType predecessors_;
  SetType live_in_;
  SetType live_out_;
  SetType uses_;
  SetType defs_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
