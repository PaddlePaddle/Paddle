/*
 * This file defines TensorRTSubgraphNodeMarkPass which helps to mark the ops
 * that supported by TensorRT engine.
 */
#include "paddle/fluid/inference/analysis/pass.h"
#include "paddle/fluid/inference/analysis/subgraph_splitter.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Mark the operators that TensorRT engine supports.
 */
class TensorRTSubgraphNodeMarkPass : public DataFlowGraphPass {
 public:
  using teller_t = SubGraphSplitter::NodeInsideSubgraphTeller;

  TensorRTSubgraphNodeMarkPass(const teller_t& teller) : teller_(teller) {}

  bool Initialize(Argument* argument) override { return true; }

  // This class get a sub-graph as input and determine whether to transform this
  // sub-graph into TensorRT.
  void Run(DataFlowGraph* graph) override;

  std::string repr() const { return "tensorrt-sub-subgraph-mark"; }
  std::string description() const { return "tensorrt sub-graph mark pass"; }

  Pass* CreateGraphvizDebugerPass() const override;
  bool Finalize() override;

 private:
  teller_t teller_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
