#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

void PlainNet::AddOp(const OpDesc& desc) {
  ops_.push_back(OpRegistry::CreateOp(desc));
}

void PlainNet::AddOp(const OperatorPtr& op) { ops_.push_back(op); }

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

class PlainNetOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  PlainNetOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddComment("This is test op");
  }
};
}  // namespace framework
}  // namespace paddle

REGISTER_OP(plainnet_operator, paddle::framework::PlainNet,
            paddle::framework::PlainNetOpProtoAndCheckerMaker);