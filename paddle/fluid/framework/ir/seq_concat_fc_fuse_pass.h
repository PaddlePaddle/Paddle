#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class SeqConcatFcFusePass : public Pass {
 public:
  virtual ~SeqConcatFcFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(std::unique_ptr<ir::Graph> graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
