#pragma once
#include <string>
#include <algorithm>
#include <list>
#include <iterator>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

// O(1) insert, delete . sorted by node size.
class OrderedReusedNodePairPool {
public:
  using Iter = typename std::list<std::pair<ir::Node*, ir::Node*>>::iterator;
  using ConstIter = typename std::list<std::pair<ir::Node*, ir::Node*>>::const_iterator;
  struct VarInBytesComparator {
    using NodePair = std::pair<ir::Node*, ir::Node*>;
    bool operator()(const NodePair& lhs, ir::Node* rhs) {
      auto get_node_size = [&](ir::Node* n) {
        auto* desc = n->Var();
        auto shape = desc->GetShape();
        size_t type_size =
        framework::SizeOfType(framework::ToTypeIndex(desc->GetDataType()));
        int size = 1;
        for(auto& s: shape) { size *= s; }
        return type_size * std::abs(size);
      };
      return get_node_size(lhs.first) < get_node_size(rhs);
    }
  };
public:
  void Insert(ir::Node* var, ir::Node* op) {
    PADDLE_ENFORCE(var->IsVar() && !var->IsCtrlVar());
    PADDLE_ENFORCE(op->IsOp());
    Iter it = std::lower_bound(nodes_.begin(), nodes_.end(), var, VarInBytesComparator());
    it = nodes_.insert(it, std::make_pair(var, op));
    mark_table_[var->Name()] = it;
  }
  void Erase(ir::Node* var) {
    PADDLE_ENFORCE(mark_table_.count(var->Name()));
    nodes_.erase(mark_table_[var->Name()]);
    mark_table_.erase(var->Name());
  }
  Iter begin() { return nodes_.begin();}
  Iter end() { return nodes_.end();}
  ConstIter begin() const { return nodes_.begin();}
  ConstIter end() const { return nodes_.end();}

private:
  // for searching.
  std::unordered_map<std::string, Iter> mark_table_;
  // node swap pairs. var -> cache var
  std::list<std::pair<ir::Node*, ir::Node*> > nodes_;
};

constexpr char kFetchedVars[] = "fetched_vars";
constexpr char kUnlivedNodePool[] = "unused_node_pool";
constexpr char kReusedNodePairMap[] = "reused_nodepair_map";
constexpr char kGraphOpsReused[] = "graph_ops_reused";
constexpr char kGraphEarlyDeleteOpsDeps[] = "graph_early_delete_ops_deps";
// insert early delete op after these ops

using UnlivedNodePool = std::vector<std::pair<std::string, /*var node*/
                                              int> /*the last op which use var node id*/>;  // order matters
using ReusedNodePairMap =
    std::unordered_map<int /*op order id*/,
                       std::pair<std::string /*var*/, std::string /*reused var*/>>;
using GraphOpsReused = std::vector<int/*op order id*/>;
using GraphEarlyDeleteOpsDeps = std::vector<std::vector<int/*op order id*/>>;


}  // namespace details
}  // namespace framework
}  // namespace paddle
