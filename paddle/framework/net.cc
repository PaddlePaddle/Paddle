#include "paddle/framework/net.h"

namespace paddle {
namespace framework {

PlainNet::PlainNet(const NetDesc& def) {}

Error PlainNet::InferShape(Scope* scope) {
  for (auto& op : ops_) {
    // wrong shape
    auto err = op.InferShape();
    if (!err) return err;
  }
  // ok
  return Error();
}

Error PlainNet::Run(Scope* scope, OpContext* context, OpIndex begin,
                    OpIndex end) const {
  // TODO Add implementation here.
  return Error();
}

}  // namespace framework
}  // namespace paddle
