/*
 * This file defines IRPassManager, it helps control the passes in IR. Inference
 * phrase will load the model program and parameters from disk, that is quite
 * different from the training phase.
 * This manager will control the Passes and make the passes in IR work smoothly
 * for inference.
 */

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace inference {
namespace analysis {
using framework::ProgramDesc;

class IRPassManager final {
 public:
  IRPassManager(const ProgramDesc& program);

  void Apply(const std::vector<std::string>& passes);

  framework::ir::Graph& graph() const { return *graph_; }

 private:
  std::unique_ptr<framework::ir::Graph> graph_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
