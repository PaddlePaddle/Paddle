/*
 * This file defines the class Argument, which is the input and output of the
 * analysis module. All the fields that needed either by Passes or PassManagers
 * are contained in Argument.
 *
 * TODO(Superjomn) Find some way better to contain the fields when it grow too
 * big.
 */

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * The argument definition of both Pass and PassManagers.
 *
 * All the fields should be registered here for clearness.
 */
struct Argument {
  // The graph that process by the Passes or PassManagers.
  std::unique_ptr<DataFlowGraph> main_dfg;

  // The original program desc.
  std::unique_ptr<framework::proto::ProgramDesc> origin_program_desc;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
