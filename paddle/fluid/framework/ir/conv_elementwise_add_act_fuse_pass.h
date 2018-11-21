#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

class ConvElementwiseAddActFusePass : public FusePassBase {
 public:
  virtual ~ConvElementwiseAddActFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
