#include "paddle/fluid/inference/analysis/pretty_log.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace analysis {

TEST(PLOG, INFO) {
  PLOG(INFO) << "this is INFO";
  std::cerr << "\033[32m hello" << std::endl;
}

TEST(PLOG, WARNING) { PLOG(INFO) << "this is WARNING"; }

}  // namespace analysis
}  // namespace framework
}  // namespace paddle
