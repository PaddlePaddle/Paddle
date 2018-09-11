#pragma once

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

class ConvElementwiseAddMKLDNNFusePass : public FusePassBase {
 public:
  virtual ~ConvElementwiseAddMKLDNNFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;

  const std::string name_scope_{"conv_elementwise_add_mkldnn_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
