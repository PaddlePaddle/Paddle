#include <set>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/details/cfg_graph.h"

namespace paddle {
namespace framework {
namespace details {

class MemoryOptimizePass : public Pass {
public:
  bool IsValidVar(ir::Node* node) const;
  const ir::Node* SearchMatch(ir::Node* var) const;
  const std::string ToString(ir::Node* var) const;
protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
private:
  std::unique_ptr<ControlFlowGraph> cfg_;
  // order matters, use set instead unordered
  std::set<ir::Node*> pool_;
  std::unordered_set<ir::Node*> skip_set_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
