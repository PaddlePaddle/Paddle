#include "paddle/fluid/inference/analysis/pretty_log.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace analysis {

using logging::info;
using logging::warn;
using logging::suc;
using logging::PrettyLog;

TEST(PLOG, INFO) { PrettyLog({{warn(), "warn"}, {info(), "info"}, {suc(), "suc"}}); }

}  // namespace analysis
}  // namespace framework
}  // namespace paddle
