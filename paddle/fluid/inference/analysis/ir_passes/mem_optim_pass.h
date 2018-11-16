#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class MemOptimPass : public framework::ir::Pass {
 public:
  virtual ~MemOptimPass() = default;

 protected:
  std::unique_ptr<framework::ir::Graph> ApplyImpl(
      std::unique_ptr<framework::ir::Graph> graph) const;

 private:
  using lifecycle_t = std::pair<int, int>;
  void CollectLifeCycle(
      std::unordered_map<std::string, lifecycle_t> *lifecycles);

  void CollectShapes(std::unordered_map<std::string, framework::ir::Node *> *tensor_nodes);

  void MakeReusePlan();

  framework::ir::Graph *graph_{nullptr};
  int max_lifecycle_{-1};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
