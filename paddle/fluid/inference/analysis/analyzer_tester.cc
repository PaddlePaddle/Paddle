#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST_F(DFG_Tester, main) {
  Analyzer analyser;
  analyser.Run(&argument);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
