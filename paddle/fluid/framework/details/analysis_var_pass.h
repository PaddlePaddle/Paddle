#pragma once

#include <set>
#include <string>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

class AnalysisVarPass : public ir::Pass {
public:

protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
                                       std::unique_ptr<ir::Graph> graph) const override;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
