#pragma once
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Load program and parameter to memory from the disk.
 */
class IrGraphBuildPass : public AnalysisPass {
 public:

};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
