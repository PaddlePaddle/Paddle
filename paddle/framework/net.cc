#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

PlainNet::PlainNet(const NetDesc& def) {}

void PlainNet::InferShape(const ScopePtr& scope) const {
  for (auto& op : ops_) {
    op.InferShape();
  }
}

void PlainNet::Run(const ScopePtr& scope, const DeviceContext& ctx) const {
  for (auto& op : ops_) {
    op.Run(ctx);
  }
}
}  // namespace framework
}  // namespace paddle
