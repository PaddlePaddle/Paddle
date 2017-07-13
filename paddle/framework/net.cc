#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

void PlainNet::AddOp(const OpDesc& desc) {
  ops_.push_back(OpRegistry::CreateOp(desc));
}

void PlainNet::InferShape(const ScopePtr& scope) const {
  for (auto& op : ops_) {
    op->InferShape(scope);
  }
}

void PlainNet::Run(const ScopePtr& scope, const DeviceContext& ctx) const {
  for (auto& op : ops_) {
    op->Run(scope, ctx);
  }
}

// REGISTER_OP(plainnet_operator, PlainNet, PlainNetOpProtoAndCheckerMaker);
}  // namespace framework
}  // namespace paddle
